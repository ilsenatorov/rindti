import os.path as osp

import torch
from encd import encd
from utils import onehot_encode

node_encoding = encd["prot"]["node"]


def encode_residue(residue: str, node_feats: str):
    """Encode a residue.

    Non-standard residues (modified amino acids, ligands, nucleotides) are treated
    as unknown: label 0, which is the padding/unknown index reserved by the
    embedding, or an all-zero one-hot vector.
    """
    residue = residue.lower()
    label = node_encoding.get(residue)
    if node_feats == "label":
        return 0 if label is None else label + 1
    elif node_feats == "onehot":
        return onehot_encode(label, len(node_encoding))
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
        """Parse PDB file.

        Residues are keyed by (chain, number): residue numbering restarts per chain,
        so keying on the number alone silently collapsed every chain onto the first.
        """
        for line in open(filename):
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                res = Residue(line)
                self.residues[(res.chainID, res.num)] = res

    def get_coords(self) -> torch.Tensor:
        """Get coordinates of all atoms"""
        coords = [[res.x, res.y, res.z] for res in self.residues.values()]
        return torch.tensor(coords)

    def get_nodes(self) -> torch.Tensor:
        """Get features of all nodes of a graph"""
        return torch.tensor([encode_residue(res.name, self.node_feats) for res in self.residues.values()])

    def get_edges(self, threshold: float) -> tuple:
        """Get edges of a graph using threshold as a cutoff.

        Returns both the edge index and the CA-CA distance for each edge, so the
        distance can be kept as an edge attribute instead of being thrown away
        once it has been thresholded.
        """
        coords = self.get_coords()
        dist = torch.cdist(coords, coords)
        edges = torch.where(dist < threshold)
        distances = dist[edges]
        edges = torch.cat([arr.view(-1, 1) for arr in edges], axis=1)
        keep = edges[:, 0] != edges[:, 1]
        return edges[keep].t(), distances[keep]

    def get_graph(self, threshold: float, edge_feats: str = "none") -> dict:
        """Get a graph using threshold as a cutoff.

        Args:
            threshold: contact cutoff in Angstrom.
            edge_feats: ``distance`` keeps the CA-CA distance as a continuous edge
                attribute; ``none`` produces an unlabelled contact graph. Note that
                only the ``transformer`` node module consumes continuous edge
                attributes - GIN/GAT/Cheb ignore edges, and FiLM expects discrete
                relation types.
        """
        nodes = self.get_nodes()
        edges, distances = self.get_edges(threshold)
        graph = dict(x=nodes, edge_index=edges)
        if edge_feats == "distance":
            graph["edge_feats"] = distances.unsqueeze(1).float()
        return graph


if __name__ == "__main__":
    import pandas as pd
    from joblib import Parallel, delayed
    from tqdm import tqdm

    if "snakemake" in globals():
        all_structures = snakemake.input.pdbs
        threshold = snakemake.params.threshold

        edge_feats = snakemake.params.edge_feats

        def get_graph(filename: str) -> dict:
            """Single function to be run in parallel."""
            return Structure(filename, snakemake.params.node_feats).get_graph(threshold, edge_feats)

        data = Parallel(n_jobs=snakemake.threads)(delayed(get_graph)(i) for i in tqdm(all_structures))
        df = pd.DataFrame(pd.Series(data, name="data"))
        df["filename"] = all_structures
        # splitext, not split("."): target IDs may contain dots, and truncating
        # there produced IDs that no longer joined against the interaction table.
        df["ID"] = df["filename"].apply(lambda x: osp.splitext(osp.basename(x))[0])
        df.set_index("ID", inplace=True)
        df.drop("filename", axis=1, inplace=True)
        df = df.to_pickle(snakemake.output.pickle)
    else:
        import os

        from jsonargparse import CLI

        def run(
            pdb_dir: str,
            output: str,
            threads: int = 1,
            threshold: float = 5,
            node_feats: str = "label",
        ):
            """Run the pipeline"""

            def get_graph(filename: str) -> dict:
                """Calculate a single graph from a file"""
                return Structure(filename, node_feats).get_graph(threshold)

            pdbs = [osp.join(pdb_dir, x) for x in os.listdir(pdb_dir)]
            data = Parallel(n_jobs=threads)(delayed(get_graph)(i) for i in tqdm(pdbs))
            df = pd.DataFrame(pd.Series(data, name="data"))
            df["filename"] = pdbs
            df["ID"] = df["filename"].apply(lambda x: osp.splitext(osp.basename(x))[0])
            df.set_index("ID", inplace=True)
            df.drop("filename", axis=1, inplace=True)
            df = df.to_dict("index")
            df.to_pickle(output)

        cli = CLI(run)
