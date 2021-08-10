import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pandas.core.frame import DataFrame
from torch_geometric.utils import to_undirected

aa_encoding = {
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
}

edge_type_encoding = {"cnt": 0, "combi": 1, "hbond": 2, "pept": 3, "ovl": 4}


def onehot_encode(position: int, count: Optional[int] = 20):
    """One-hot encode position
    Args:
        position (int): Which entry to set to 1
        count (Optional[int], optional): Max number of entries. Defaults to 20.
    Returns:
        [type]: [description]
    """
    t = [0] * count
    t[position] = 1
    return t


class ProteinEncoder:
    def __init__(self, config: dict):
        self.features = config["prepare_proteins"]["node_features"]
        self.edge_features = config["prepare_proteins"]["edge_features"]

    def encode_residue(self, residue: str) -> np.array:
        """Fully encode residue - one-hot and features

        Args:
            residue (str): One-letter residue name

        Returns:
            np.array: Concatenated features and one-hot encoding of residue name
        """
        if residue.lower() not in aa_encoding:
            return None
        elif self.features == "label":
            return aa_encoding[residue.lower()]
        elif self.features == "onehot":
            return onehot_encode(aa_encoding[residue.lower()])
        else:
            raise ValueError("Unknown features type!")

    def parse_sif(self, filename: str) -> Tuple[DataFrame, DataFrame]:
        """Parse a single sif file

        Args:
            filename (str): SIF file location

        Returns:
            Tuple[DataFrame, DataFrame]: nodes, edges DataFrames
        """
        nodes = []
        edges = []
        with open(filename, "r") as file:
            for line in file:
                line = line.strip()
                splitline = line.split()
                if len(splitline) != 3:
                    continue
                node1, edgetype, node2 = splitline
                node1split = node1.split(":")
                node2split = node2.split(":")
                if len(node1split) != 4:
                    continue
                if len(node2split) != 4:
                    continue
                chain1, resn1, x1, resaa1 = node1split
                chain2, resn2, x2, resaa2 = node2split
                if x1 != "_" or x2 != "_":
                    continue
                if resaa1.lower() not in aa_encoding or resaa2.lower() not in aa_encoding:
                    continue
                resn1 = int(resn1)
                resn2 = int(resn2)
                if resn1 == resn2:
                    continue
                edgesplit = edgetype.split(":")
                if len(edgesplit) != 2:
                    continue
                node1 = {"chain": chain1, "resn": resn1, "resaa": resaa1}
                node2 = {"chain": chain2, "resn": resn2, "resaa": resaa2}
                edgetype, _ = edgesplit
                edge = {
                    "resn1": resn1,
                    "resn2": resn2,
                    "type": edgetype,
                }
                nodes.append(node1)
                nodes.append(node2)
                edges.append(edge)
        nodes = pd.DataFrame(nodes).drop_duplicates()
        nodes = nodes.sort_values("resn").reset_index(drop=True).reset_index().set_index("resn")
        edges = pd.DataFrame(edges).drop_duplicates()
        node_idx = nodes["index"].to_dict()
        edges["node1"] = edges["resn1"].apply(lambda x: node_idx[x])
        edges["node2"] = edges["resn2"].apply(lambda x: node_idx[x])
        return nodes, edges

    def encode_nodes(self, nodes: pd.DataFrame) -> torch.Tensor:
        """Given dataframe of nodes create node features

        Args:
            nodes (pd.DataFrame): nodes dataframe from parse_sif

        Returns:
            torch.Tensor: Tensor of node features [n_nodes, *]
        """
        nodes.drop_duplicates(inplace=True)
        node_attr = [self.encode_residue(x) for x in nodes["resaa"]]
        node_attr = [x for x in node_attr if x is not None]
        node_attr = np.asarray(node_attr)
        if self.features == "label":
            node_attr = torch.tensor(node_attr, dtype=torch.long)
        else:
            node_attr = torch.tensor(node_attr, dtype=torch.float32)
        return node_attr

    def encode_edges(self, edges: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given dataframe of edges, create edge index and edge attributes

        Args:
            edges (pd.DataFrame): edges dataframe from parse_sif

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: edge index [2,n_edges], edge attributes [n_edges, *]
        """
        edges.drop_duplicates(inplace=True)
        edge_index = edges[["node1", "node2"]].astype(int).values
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_index = edge_index.t().contiguous()
        if self.edge_features == "none":
            return edge_index, None
        edge_features = edges["type"].apply(lambda x: edge_type_encoding[x])
        if self.edge_features == "label":
            edge_features = torch.tensor(edge_features, dtype=torch.long)
            edge_index, edge_features = to_undirected(edge_index, edge_features)
            return edge_index, edge_features
        elif self.edge_features == "onehot":
            edge_features = edge_features.apply(onehot_encode, count=len(edge_type_encoding))
            edge_features = torch.tensor(edge_features, dtype=torch.float)
            edge_index, edge_features = to_undirected(edge_index, edge_features)
            return edge_index, edge_features

    def __call__(self, protein_sif: str) -> dict:
        """Fully process the protein

        Args:
            protein_sif (str): File location for sif file

        Returns:
            dict: standard format with x for node features, edge_index for edges etc
        """
        nodes, edges = self.parse_sif(protein_sif)
        node_attr = self.encode_nodes(nodes)
        edge_index, edge_attr = self.encode_edges(edges)
        return dict(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, index_mapping=nodes["index"].to_dict())


def extract_name(protein_sif: str) -> str:
    """Extract the protein name from the sif filename"""
    return protein_sif.split("/")[-1].split("_")[0]


if __name__ == "__main__":
    proteins = pd.Series(list(snakemake.input.rins), name="sif")
    proteins = pd.DataFrame(proteins)
    proteins["ID"] = proteins["sif"].apply(extract_name)
    proteins.set_index("ID", inplace=True)
    prot_encoder = ProteinEncoder(snakemake.config)
    proteins["data"] = proteins["sif"].apply(prot_encoder)
    with open(snakemake.output.protein_pickle, "wb") as file:
        pickle.dump(proteins, file)
