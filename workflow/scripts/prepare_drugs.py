import pickle

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
from torch_geometric.utils import to_undirected
from utils import list_to_dict

node_encoding = list_to_dict(["other", 6, 7, 8, 9, 16, 17, 35, 15, 53, 5, 11, 14, 34])
edge_encoding = list_to_dict(["SINGLE", "DOUBLE", "AROMATIC"])


def featurize(smiles: str, max_num_atoms=150) -> dict:
    """Generate drug Data from smiles

    Args:
        smiles (str): SMILES

    Returns:
        dict: dict with x, edge_index etc
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
        edge_feats.append(edge_encoding[btype])
    if not edges:  # If no edges (bonds) were found, remove molecule
        return np.nan
    atom_features = []
    for atom in mol.GetAtoms():
        atom_num = atom.GetAtomicNum()
        if atom_num not in node_encoding.keys():
            atom_features.append(node_encoding["other"])
        else:
            atom_features.append(node_encoding[atom_num])
    if len(atom_features) > max_num_atoms:
        return np.nan
    x = torch.tensor(atom_features, dtype=torch.long)
    edge_index = torch.tensor(edges).t().contiguous()
    edge_feats = torch.tensor(edge_feats, dtype=torch.long)
    edge_index, edge_feats = to_undirected(edge_index, edge_feats)
    return dict(x=x, edge_index=edge_index, edge_feats=edge_feats)


if __name__ == "__main__":

    ligs = pd.read_csv(snakemake.input.lig, sep="\t").drop_duplicates("Drug_ID").set_index("Drug_ID")
    ligs["data"] = ligs["Drug"].apply(featurize, max_num_atoms=snakemake.config["prepare_drugs"]["max_num_atoms"])
    ligs = ligs[ligs["data"].notna()]

    with open(snakemake.output.drug_pickle, "wb") as file:
        pickle.dump(ligs, file)
