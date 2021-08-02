import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
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

aa3to1 = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
}

aa1to3 = {v: k for (k, v) in aa3to1.items()}

pro_res_aliphatic_table = ["A", "I", "L", "M", "V"]
pro_res_aromatic_table = ["F", "W", "Y"]
pro_res_polar_neutral_table = ["C", "N", "Q", "S", "T"]
pro_res_acidic_charged_table = ["D", "E"]
pro_res_basic_charged_table = ["H", "K", "R"]

res_weight_table = {
    "A": 71.08,
    "C": 103.15,
    "D": 115.09,
    "E": 129.12,
    "F": 147.18,
    "G": 57.05,
    "H": 137.14,
    "I": 113.16,
    "K": 128.18,
    "L": 113.16,
    "M": 131.20,
    "N": 114.11,
    "P": 97.12,
    "Q": 128.13,
    "R": 156.19,
    "S": 87.08,
    "T": 101.11,
    "V": 99.13,
    "W": 186.22,
    "Y": 163.18,
}

res_pka_table = {
    "A": 2.34,
    "C": 1.96,
    "D": 1.88,
    "E": 2.19,
    "F": 1.83,
    "G": 2.34,
    "H": 1.82,
    "I": 2.36,
    "K": 2.18,
    "L": 2.36,
    "M": 2.28,
    "N": 2.02,
    "P": 1.99,
    "Q": 2.17,
    "R": 2.17,
    "S": 2.21,
    "T": 2.09,
    "V": 2.32,
    "W": 2.83,
    "Y": 2.32,
}

res_pkb_table = {
    "A": 9.69,
    "C": 10.28,
    "D": 9.60,
    "E": 9.67,
    "F": 9.13,
    "G": 9.60,
    "H": 9.17,
    "I": 9.60,
    "K": 8.95,
    "L": 9.60,
    "M": 9.21,
    "N": 8.80,
    "P": 10.60,
    "Q": 9.13,
    "R": 9.04,
    "S": 9.15,
    "T": 9.10,
    "V": 9.62,
    "W": 9.39,
    "Y": 9.62,
}

res_pkx_table = {
    "A": 0.00,
    "C": 8.18,
    "D": 3.65,
    "E": 4.25,
    "F": 0.00,
    "G": 0,
    "H": 6.00,
    "I": 0.00,
    "K": 10.53,
    "L": 0.00,
    "M": 0.00,
    "N": 0.00,
    "P": 0.00,
    "Q": 0.00,
    "R": 12.48,
    "S": 0.00,
    "T": 0.00,
    "V": 0.00,
    "W": 0.00,
    "Y": 0.00,
}

res_pl_table = {
    "A": 6.00,
    "C": 5.07,
    "D": 2.77,
    "E": 3.22,
    "F": 5.48,
    "G": 5.97,
    "H": 7.59,
    "I": 6.02,
    "K": 9.74,
    "L": 5.98,
    "M": 5.74,
    "N": 5.41,
    "P": 6.30,
    "Q": 5.65,
    "R": 10.76,
    "S": 5.68,
    "T": 5.60,
    "V": 5.96,
    "W": 5.89,
    "Y": 5.96,
}

res_hydrophobic_ph2_table = {
    "A": 47,
    "C": 52,
    "D": -18,
    "E": 8,
    "F": 92,
    "G": 0,
    "H": -42,
    "I": 100,
    "K": -37,
    "L": 100,
    "M": 74,
    "N": -41,
    "P": -46,
    "Q": -18,
    "R": -26,
    "S": -7,
    "T": 13,
    "V": 79,
    "W": 84,
    "Y": 49,
}

res_hydrophobic_ph7_table = {
    "A": 41,
    "C": 49,
    "D": -55,
    "E": -31,
    "F": 100,
    "G": 0,
    "H": 8,
    "I": 99,
    "K": -23,
    "L": 97,
    "M": 74,
    "N": -28,
    "P": -46,
    "Q": -10,
    "R": -14,
    "S": -5,
    "T": 13,
    "V": 76,
    "W": 97,
    "Y": 63,
}


def minmax_normalise(data: dict) -> dict:
    """Normalise data in the dict

    Args:
        data (dict): Input data

    Returns:
        dict: Normalised data
    """
    min_val = min(data.values())
    max_val = max(data.values())
    interval = max_val - min_val
    return {k: (v - min_val) / interval for (k, v) in data.items()}


res_weight_table = minmax_normalise(res_weight_table)
res_pka_table = minmax_normalise(res_pka_table)
res_pkb_table = minmax_normalise(res_pkb_table)
res_pkx_table = minmax_normalise(res_pkx_table)
res_pl_table = minmax_normalise(res_pl_table)
res_hydrophobic_ph2_table = minmax_normalise(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = minmax_normalise(res_hydrophobic_ph7_table)


def residue_features(residue: str) -> np.ndarray:
    """Calculate array of residue features

    Args:
        residue (str): One-letter residue name

    Returns:
        np.array: Concatenated features
    """
    residue = aa3to1[residue.upper()]
    res_property1 = [
        1 if residue in pro_res_aliphatic_table else 0,
        1 if residue in pro_res_aromatic_table else 0,
        1 if residue in pro_res_polar_neutral_table else 0,
        1 if residue in pro_res_acidic_charged_table else 0,
        1 if residue in pro_res_basic_charged_table else 0,
    ]
    res_property2 = [
        res_weight_table[residue],
        res_pka_table[residue],
        res_pkb_table[residue],
        res_pkx_table[residue],
        res_pl_table[residue],
        res_hydrophobic_ph2_table[residue],
        res_hydrophobic_ph7_table[residue],
    ]
    # print(np.array(res_property1 + res_property2).shape)
    return np.asarray(res_property1 + res_property2)


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


def encode_residue(residue: str, features="label") -> np.array:
    """Fully encode residue - one-hot and features

    Args:
        residue (str): One-letter residue name

    Returns:
        np.array: Concatenated features and one-hot encoding of residue name
    """
    if residue.lower() not in aa_encoding:
        return None
    elif features == "label":
        return aa_encoding[residue.lower()]
    elif features == "custom":
        res_features = residue_features(residue)
        onehot = onehot_encode(aa_encoding[residue.lower()])
        return np.concatenate([onehot, res_features])
    elif features == "onehot":
        return onehot_encode(aa_encoding[residue.lower()])
    else:
        raise ValueError("Unknown features type!")


contact_type1_encoding = {"cnt": 0, "combi": 1, "hbond": 2, "pept": 3, "ovl": 4}

contact_type2_encoding = {"all_all": 0, "mc_mc": 1, "mc_sc": 2, "sc_sc": 3}


def parse_sif(filename: str):
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
            edgetype1, edgetype2 = edgesplit
            edge = {
                "resn1": resn1,
                "resn2": resn2,
                "type1": edgetype1,
                "type2": edgetype2,
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


def process_graph(nodes: pd.DataFrame, edges: pd.DataFrame, features="label") -> Tuple[np.array, np.array]:
    # sourcery skip: extract-duplicate-method, remove-redundant-if, split-or-ifs
    """Given nodes and edges for a graph, process it to respective matrices for nodes and edges

    Args:
        nodes (pd.DataFrame): Data for each node
        edges (pd.DataFrame): Data for each edge
        onehot (Optional[bool], optional): Whether to one-hot encode nodes. Defaults to False.

    Returns:
        Tuple[np.array, np.array]: nodes, edges as a matrix
    """
    edges.drop("type1", axis=1, inplace=True)
    edges.drop("type2", axis=1, inplace=True)
    edges.drop_duplicates(inplace=True)
    edge_list = edges[["node1", "node2"]].astype(int).values
    node_attr = [encode_residue(x, features=features) for x in nodes["resaa"]]
    node_attr = [x for x in node_attr if x is not None]
    node_attr = np.asarray(node_attr)
    if features == "label":
        node_attr = torch.tensor(node_attr, dtype=torch.long)
    else:
        node_attr = torch.tensor(node_attr, dtype=torch.float32)
    return node_attr, np.unique(edge_list, axis=0)


def process_protein(protein_sif: str, features: str = "label") -> dict:
    """Fully process the protein

    Args:
        protein_sif (str): File location for sif file
        config (dict): Snakemake config

    Returns:
        dict: standard format with x for node features, edge_index for edges etc
    """
    nodes, edges = parse_sif(protein_sif)
    node_attr, edge_list = process_graph(nodes, edges, features)
    edge_list = torch.tensor(edge_list, dtype=torch.long)
    edge_index = edge_list.t().contiguous()
    edge_index = to_undirected(edge_index)
    return dict(x=node_attr, edge_index=edge_index, index_mapping=nodes["index"].to_dict())


def summary(filename: str) -> dict:
    """Summarise a sif file, used for plotting

    Args:
        filename (str): File location for sif file

    Returns:
        dict: Summary of RIN
    """
    nodes, edges = parse_sif(filename)
    res = {"type1_count": edges["type1"].value_counts()}
    res["type2_count"] = edges["type2"].value_counts()
    res["nnodes"] = nodes.shape[0]
    res["nedges"] = edges.shape[0]
    res["aa_count"] = nodes["resaa"].value_counts()
    combined_edges = edges["type1"] + "_" + edges["type2"]
    res["type_count"] = combined_edges.value_counts()
    return res


def extract_name(protein_sif):
    return protein_sif.split("/")[-1].split("_")[0]


def encode_sequence(sequence: str, maxlen: int, onehot: Optional[bool] = False) -> torch.Tensor:
    """Encode the protein sequence

    Args:
        sequence (str): FASTA sequence
        maxlen (int): Max allowed length
        onehot (Optional[bool], optional): Whether to use one-hot encoding. Defaults to False.

    Returns:
        torch.Tensor: Tensor of final data
    """
    sequence = sequence[:maxlen]
    n = len(sequence)
    lenc = np.asarray([aa_encoding[aa1to3[i.upper()].lower()] for i in sequence])
    lenc = lenc + 1
    if onehot:
        ohenc = onehot.fit_transform(lenc.reshape(-1, 1))
        padded = np.pad(ohenc, ((0, maxlen - n), (0, 0)))
        return torch.tensor(padded, dtype=torch.float32)
    else:
        padded = np.pad(lenc, (0, maxlen - n)).reshape(1, -1)
        return torch.tensor(padded, dtype=torch.long)


if __name__ == "__main__":
    if snakemake.config["prepare_proteins"]["data"] == "graph":
        proteins = pd.Series(list(snakemake.input.rins), name="sif")
        proteins = pd.DataFrame(proteins)
        proteins["ID"] = proteins["sif"].apply(extract_name)
        proteins.set_index("ID", inplace=True)
        proteins["data"] = proteins["sif"].apply(
            process_protein, features=snakemake.config["prepare_proteins"]["features"]
        )
    elif snakemake.config["prepare_proteins"]["data"] == "sequence":
        proteins = pd.read_csv(snakemake.input.targ)
        features = snakemake.config["prepare_proteins"]["features"]
        maxlen = snakemake.config["prepare_proteins"]["sequence"]["maxlen"]
        if features == "onehot":
            onehot = OneHotEncoder(sparse=False)
        proteins["data"] = proteins["FASTA Sequence"].apply(lambda x: {"x": encode_sequence(x, maxlen, onehot)})
        proteins.set_index("UniProt ID", inplace=True)
    else:
        raise ValueError("Unkown protein encoding method")
    with open(snakemake.output.protein_pickle, "wb") as file:
        pickle.dump(proteins, file)
