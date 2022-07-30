from pathlib import Path
from typing import Callable

import torch
from joblib import Parallel, delayed
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

node_encode = {
    "ala": 0,
    "arg": 1,
    "asn": 2,
    "asp": 3,
    "cys": 4,
    "gln": 5,
    "glu": 6,
    "gly": 7,
    "his": 8,
    "ile": 9,
    "leu": 10,
    "lys": 11,
    "met": 12,
    "phe": 13,
    "pro": 14,
    "ser": 15,
    "thr": 16,
    "trp": 17,
    "tyr": 18,
    "val": 19,
    "unk": 20,
}


class Residue:
    """Residue class"""

    def __init__(self, line: str) -> None:
        self.name = line[17:20].strip()
        self.num = int(line[22:26].strip())
        self.chainID = line[21].strip()
        self.x = float(line[30:38].strip())
        self.y = float(line[38:46].strip())
        self.z = float(line[46:54].strip())


class Structure:
    """Structure class"""

    def __init__(self, filename: str) -> None:
        self.residues = {}
        self.parse_file(filename)

    def parse_file(self, filename: str) -> None:
        """Parse PDB file"""
        for line in open(filename, "r"):
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                res = Residue(line)
                self.residues[res.num] = res

    def get_coords(self) -> torch.Tensor:
        """Get coordinates of all atoms"""
        coords = [[res.x, res.y, res.z] for res in self.residues.values()]
        return torch.tensor(coords)

    def get_nodes(self) -> torch.Tensor:
        """Get features of all nodes of a graph"""
        return torch.tensor([node_encode[res.name.lower()] for res in self.residues.values()])

    def get_edges(self, threshold: float) -> torch.Tensor:
        """Get edges of a graph using threshold as a cutoff"""
        coords = self.get_coords()
        dist = torch.cdist(coords, coords)
        edges = torch.where(dist < threshold)
        edges = torch.cat([arr.view(-1, 1) for arr in edges], axis=1)
        edges = edges[edges[:, 0] != edges[:, 1]]
        return edges.t()

    def get_graph(self, threshold: float) -> dict:
        """Get a graph using threshold as a cutoff"""
        nodes = self.get_nodes()
        edges = self.get_edges(threshold)
        coords = self.get_coords()
        return dict(x=nodes, edge_index=edges, pos=coords)


def run(input_dir: str, output_dir: str, threads: int = 1, threshold: int = 5):
    """Run the pipeline"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    def save_graph(pdb_id: str) -> None:
        """Calculate a single graph from a file"""
        graph: dict = Structure(input_dir / f"{pdb_id}.pdb").get_graph(threshold)
        data = Data(**graph)
        torch.save(data, output_dir / f"{pdb_id}.pt")

    pdbs = [f.stem for f in input_dir.glob("*.pdb")]
    Parallel(n_jobs=threads)(delayed(save_graph)(i) for i in tqdm(pdbs))


class LargePreTrainDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Callable = None,
        pre_transform: Callable = None,
        pre_filter: Callable = None,
        threshold: int = 7,
        threads: int = 1,
    ):
        self.input_dir = Path(root)
        self.threshold = threshold
        self.threads = threads
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        """All generated filenames."""
        return [f"data_{x}.pt" for x in range(self.len())]

    def process(self):
        """Convert pdbs into graphs."""
        idx = 0
        for raw_path in self.input_dir.glob("*.pdb"):
            s = Structure(raw_path)
            data = Data(**s.get_graph(self.threshold))

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, Path(self.processed_dir) / f"data_{idx}.pt")
            idx += 1

    def len(self):
        """Number of pdb structures == number of graphs."""
        return len([x for x in self.input_dir.glob("*.pdb")])

    def get(self, idx: int):
        """Load a single graph."""
        return torch.load(Path(self.processed_dir) / f"data_{idx}.pt")
