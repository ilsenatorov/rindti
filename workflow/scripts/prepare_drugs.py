import pickle
from typing import Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
from torch_geometric.utils import to_undirected
from utils import list_to_dict, onehot_encode

node_encoding = list_to_dict(["other", 6, 7, 8, 9, 16, 17, 35, 15, 53, 5, 11, 14, 34])
edge_encoding = list_to_dict(["SINGLE", "DOUBLE", "AROMATIC"])


class DrugEncoder:
    def __init__(self, node_feats: str, edge_feats: str, max_num_atoms: int = 150):
        """Drug encoder, goes from SMILES to dictionary of torch data

        Args:
            node_feats (str): 'label' or 'onehot'
            edge_feats (str): 'label' or 'onehot
            max_num_atoms (int, optional): filter out molecules that are too big. Defaults to 150.
        """
        self.node_feats = node_feats
        self.edge_feats = edge_feats
        self.max_num_atoms = max_num_atoms

    def encode_node(self, atom_num):
        """Encode single atom"""
        if atom_num not in node_encoding.keys():
            atom_num = "other"
        label = node_encoding[atom_num]
        if self.node_feats == "onehot":
            return onehot_encode(label, len(node_encoding))
        elif self.node_feats == "label":
            return label + 1
        else:
            raise ValueError("Unknown node encoding!")

    def encode_edge(self, edge):
        """Encode single edge"""
        label = edge_encoding[edge]
        if self.edge_feats == "onehot":
            return onehot_encode(label, len(edge_encoding))
        elif self.edge_feats == "label":
            return label
        else:
            raise ValueError("Unknown edge encoding!")

    def __call__(self, smiles: str) -> dict:
        """Generate drug Data from smiles

        Args:
            smiles (str): SMILES

        Returns:
            dict: dict with x, edge_index etc or np.nan for bad entries
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:  # when rdkit fails to read a molecule it returns None
            return np.nan
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
        edges = []
        edge_feats = []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.append([start, end])
            btype = str(bond.GetBondType())
            # If bond type is unknown, remove molecule
            if btype not in edge_encoding.keys():
                return np.nan
            edge_feats.append(self.encode_edge(btype))
        if not edges:  # If no edges (bonds) were found, remove molecule
            return np.nan
        atom_features = []
        for atom in mol.GetAtoms():
            atom_num = atom.GetAtomicNum()
            atom_features.append(self.encode_node(atom_num))
        if len(atom_features) > self.max_num_atoms:
            return np.nan
        if self.node_feats == "label":
            x = torch.tensor(atom_features, dtype=torch.long)
        else:
            x = torch.tensor(atom_features, dtype=torch.float32)
        edge_index = torch.tensor(edges).t().contiguous()
        edge_feats = torch.tensor(edge_feats, dtype=torch.long)
        edge_index, edge_feats = to_undirected(edge_index, edge_feats)
        return dict(x=x, edge_index=edge_index, edge_feats=edge_feats)


if __name__ == "__main__":
    config = snakemake.config["prepare_drugs"]
    drug_enc = DrugEncoder(config["node_feats"], config["edge_feats"], config["max_num_atoms"])
    ligs = pd.read_csv(snakemake.input.lig, sep="\t").drop_duplicates("Drug_ID").set_index("Drug_ID")
    ligs["data"] = ligs["Drug"].apply(drug_enc)
    ligs = ligs[ligs["data"].notna()]

    with open(snakemake.output.drug_pickle, "wb") as file:
        pickle.dump(ligs, file)
