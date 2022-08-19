from pathlib import Path
from typing import Callable, Union

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
        self.plddt = float(line[60:66].strip())


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

    def get_graph(self) -> dict:
        """Get a graph using threshold as a cutoff"""
        nodes = []
        pos = []
        plddt = []
        for res in self.residues.values():
            nodes.append(node_encode[res.name.lower()])
            pos.append([res.x, res.y, res.z])
            plddt.append(res.plddt)
        return dict(x=torch.tensor(nodes), pos=torch.tensor(pos), plddt=torch.tensor(plddt))


class LectinPreTrainDataset(Dataset):
    def __init__(
        self,
        root: str,
        data_dir: str,
        transform: Callable = None,
        pre_transform: Callable = None,
        pre_filter: Callable = None,
        threads: int = 1,
    ):
        self.input_dir = Path(data_dir)
        self.threads = threads
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        """All generated filenames."""
        return ["data_0.pt"]

    def process(self):
        """Convert pdbs into graphs."""

        for idx, i in tqdm(enumerate(self.input_dir.glob("*.pdb"))):
            s = Structure(i)
            data = Data(**s.get_graph())
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data, Path(self.processed_dir) / f"data_{idx}.pt")

    def len(self):
        """Number of pdb structures == number of graphs."""
        return 727

    def get(self, idx: int):
        """Load a single graph."""
        return torch.load(Path(self.processed_dir) / f"data_{idx}.pt")
