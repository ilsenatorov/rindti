import pickle

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

# atom_num_mapping = {0: 'padding',
#                     1: 6,
#                     2: 8,
#                     3: 7,
#                     4: 16,
#                     5: 9,
#                     6: 17,
#                     7: 35,
#                     8: 15,
#                     9: 53,
#                     10: 11,
#                     11: 14,
#                     12: 5,
#                     13: 19,
#                     14: 34,
#                     15: 33,
#                     16: 30,
#                     17: 51,
#                     18: 3,
#                     19: 13,
#                     20: 20,
#                     21: 12,
#                     22: 52,
#                     23: 47,
#                     24: 56,
#                     25: 1,
#                     26: 38,
#                     27: 23,
#                     28: 'other'}

atom_num_mapping = {0: "padding", 1: 6, 2: 8, 3: 7, 4: 16, 5: 9, 6: 17, 7: 35, 8: "other"}

atom_num_mapping = {v: k for (k, v) in atom_num_mapping.items()}  # Reverse the mapping dict


def featurize(smiles: str) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:  # when rdkit fails to read a molecule it returns None
        return np.nan
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)

    edges = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([start, end])
    if not edges:  # If no edges (bonds) were found, exit (single ion etc)
        return np.nan
    atom_features = []
    for atom in mol.GetAtoms():
        atom_num = atom.GetAtomicNum()
        if atom_num not in atom_num_mapping.keys():
            atom_features.append(atom_num_mapping["other"])
        else:
            atom_features.append(atom_num_mapping[atom_num])

    x = torch.tensor(atom_features, dtype=torch.long)
    edge_index = torch.tensor(edges).t().contiguous()

    return dict(x=x, edge_index=to_undirected(edge_index, num_nodes=x.size(0)))


if __name__ == "__main__":

    ligs = pd.read_csv(snakemake.input.lig).drop_duplicates("InChI Key").set_index("InChI Key")
    ligs["data"] = ligs["Canonical SMILES"].apply(featurize)
    ligs = ligs[ligs["data"].notna()]

    with open(snakemake.output.drug_pickle, "wb") as file:
        pickle.dump(ligs, file)
