import pickle

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
from rdkit.Chem.rdchem import ChiralType
from torch_geometric.utils import to_undirected
from utils import list_to_dict, onehot_encode

node_encoding = list_to_dict(["other", 6, 7, 8, 9, 16, 17, 35, 15, 53, 5, 11, 14, 34])
edge_encoding = list_to_dict(["SINGLE", "DOUBLE", "AROMATIC"])
glycan_encoding = {
    "other": [0, 0, 0],
    6: [1, 0, 0],  # carbon
    7: [0, 1, 0],  # nitrogen
    8: [0, 0, 1],  # oxygen
}

chirality_encoding = {
    ChiralType.CHI_OTHER: [0, 0, 0],
    ChiralType.CHI_TETRAHEDRAL_CCW: [
        1,
        1,
        0,
    ],  # counterclockwise rotation of polarized light -> rotate light to the left
    ChiralType.CHI_TETRAHEDRAL_CW: [1, 0, 1],  # clockwise rotation of polarized light -> rotate light to the right
    ChiralType.CHI_UNSPECIFIED: [0, 0, 0],
}


class DrugEncoder:
    """Drug encoder, goes from SMILES to dictionary of torch data

    Args:
        node_feats (str): 'label' or 'onehot'
        edge_feats (str): 'label' or 'onehot
        max_num_atoms (int, optional): filter out molecules that are too big. Defaults to 150.
    """

    def __init__(self, node_feats: str, edge_feats: str, max_num_atoms: int = 150):
        assert node_feats in {"label", "onehot", "glycan", "glycanone", "IUPAC"}
        assert edge_feats in {"label", "onehot", "none"}
        self.node_feats = node_feats
        self.edge_feats = edge_feats
        self.max_num_atoms = max_num_atoms

    def encode_node(self, atom_num, atom):
        """Encode single atom"""
        if atom_num not in node_encoding.keys():
            atom_num = "other"

        if self.node_feats == "glycan":
            if atom_num in glycan_encoding:
                return glycan_encoding[atom_num] + chirality_encoding[atom.GetChiralTag()]
            else:
                return glycan_encoding["other"] + chirality_encoding[atom.GetChiralTag()]

        label = node_encoding[atom_num]
        if self.node_feats == "onehot":
            return onehot_encode(label, len(node_encoding))
        return label + 1

    def encode_edge(self, edge):
        """Encode single edge"""
        label = edge_encoding[edge]
        if self.edge_feats == "onehot":
            return onehot_encode(label, len(edge_encoding))
        elif self.edge_feats == "label":
            return label
        else:
            raise ValueError("This shouldn't be called for edge type none")

    def __call__(self, smiles: str) -> dict:
        """Generate drug Data from smiles

        Args:
            smiles (str): SMILES

        Returns:
            dict: dict with x, edge_index etc or np.nan for bad entries
        """
        if smiles != smiles:  # check for nans, i.e. missing smiles strings in dataset
            return np.nan
        mol = Chem.MolFromSmiles(smiles)
        if not mol:  # when rdkit fails to read a molecule it returns None
            return np.nan
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
        edges = []
        edge_feats = [] if self.edge_feats != "none" else None
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.append([start, end])
            btype = str(bond.GetBondType())
            # If bond type is unknown, remove molecule
            if btype not in edge_encoding.keys():
                return np.nan
            if self.edge_feats != "none":
                edge_feats.append(self.encode_edge(btype))
        if not edges:  # If no edges (bonds) were found, remove molecule
            return np.nan
        atom_features = []
        for atom in mol.GetAtoms():
            atom_num = atom.GetAtomicNum()
            atom_features.append(self.encode_node(atom_num, atom))
        if len(atom_features) > self.max_num_atoms:
            return np.nan
        if self.node_feats == "label":
            x = torch.tensor(atom_features, dtype=torch.long)
        else:
            x = torch.tensor(atom_features, dtype=torch.float32)
        edge_index = torch.tensor(edges).t().contiguous()
        if self.edge_feats == "onehot":
            edge_feats = torch.tensor(edge_feats, dtype=torch.float32)
        elif self.edge_feats == "label":
            edge_feats = torch.tensor(edge_feats, dtype=torch.long)
        elif self.edge_feats == "none":
            edge_feats = None
        else:
            raise ValueError("Unknown edge encoding!")
        if self.edge_feats != "none":
            edge_index, edge_feats = to_undirected(edge_index, edge_feats)
        else:
            edge_index = to_undirected(edge_index)
        return dict(x=x, edge_index=edge_index, edge_feats=edge_feats)


if __name__ == "__main__":
    drug_enc = DrugEncoder(snakemake.params.node_feats, snakemake.params.edge_feats, snakemake.params.max_num_atoms)
    ligs = pd.read_csv(snakemake.input.lig, sep="\t", index_col=0)
    ligs["data"] = ligs["Drug"].apply(drug_enc)
    ligs = ligs[ligs["data"].notna()]
    ligs = ligs.to_pickle(snakemake.output.pickle)
