import torch
from encd import encd
from utils import onehot_encode

node_encoding = encd["prot"]["node"]


def encode_residue(residue: str, node_feats: str):
    """Encode a residue"""
    residue = residue.lower()
    if node_feats == "label":
        if residue not in node_encoding:
            return node_encoding["unk"]
        return node_encoding[residue] + 1
    elif node_feats == "onehot":
        return onehot_encode(node_encoding[residue], len(node_encoding))
    else:
        raise ValueError("Unknown node_feats type!")


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

    def __init__(self, filename: str, node_feats: str) -> None:
        self.residues = {}
        self.parse_file(filename)
        self.node_feats = node_feats

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
        return torch.tensor([encode_residue(res.name, self.node_feats) for res in self.residues.values()])

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


if __name__ == "__main__":
    import pandas as pd
    from joblib import Parallel, delayed
    from tqdm import tqdm

    if "snakemake" in globals():
        all_structures = snakemake.input.pdbs
        threshold = snakemake.params.threshold

        def get_graph(filename: str) -> dict:
            """Single function to be run in parallel."""
            return Structure(filename, snakemake.params.node_feats).get_graph(threshold)

        data = Parallel(n_jobs=snakemake.threads)(delayed(get_graph)(i) for i in tqdm(all_structures))
        df = pd.DataFrame(pd.Series(data, name="data"))
        df["filename"] = all_structures
        df["ID"] = df["filename"].apply(lambda x: x.split("/")[-1].split(".")[0])
        df.set_index("ID", inplace=True)
        df.drop("filename", axis=1, inplace=True)
        df = df.to_pickle(snakemake.output.pickle)
    else:
        from pathlib import Path

        from jsonargparse import CLI

        def run(input_dir: str, output_dir: str, threads: int = 1, threshold: int = 5, node_feats: str = "label"):
            """Run the pipeline"""
            input_dir = Path(input_dir)
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)

            def save_graph(pdb_id: str) -> None:
                """Calculate a single graph from a file"""
                graph: dict = Structure(input_dir / f"{pdb_id}.pdb", node_feats).get_graph(threshold)
                torch.save(graph, output_dir / f"{pdb_id}.pt")

            pdbs = [f.stem for f in input_dir.glob("*.pdb")]
            Parallel(n_jobs=threads)(delayed(save_graph)(i) for i in tqdm(pdbs))

        cli = CLI(run)
