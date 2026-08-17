import numpy as np
import pandas as pd
import torch
from encd import encd
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
from torch_geometric.utils import to_undirected
from utils import onehot_encode

# Buckets for the `rich` atom featuriser. Values outside a bucket land in a
# trailing "other" slot, so every list below encodes len(bucket) + 1 features.
DEGREES = [0, 1, 2, 3, 4, 5]
CHARGES = [-2, -1, 0, 1, 2]
NUM_HS = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
CHIRALITIES = [
    "CHI_UNSPECIFIED",
    "CHI_TETRAHEDRAL_CW",
    "CHI_TETRAHEDRAL_CCW",
    "CHI_OTHER",
]


def _bucket(value, options: list) -> list:
    """One-hot over ``options`` with a trailing slot for anything unlisted."""
    encoded = [0] * (len(options) + 1)
    encoded[options.index(value) if value in options else len(options)] = 1
    return encoded


class DrugEncoder:
    """Drug encoder, goes from SMILES to dictionary of torch data

    Args:
        node_feats (str): 'label', 'onehot', 'rich' or 'glycan'
        edge_feats (str): 'label' or 'onehot
        max_num_atoms (int, optional): filter out molecules that are too big. Defaults to 150.
    """

    def __init__(self, node_feats: str, edge_feats: str, max_num_atoms: int = 150):
        assert node_feats in {"label", "onehot", "rich", "glycan", "glycanone", "IUPAC"}
        assert edge_feats in {"label", "onehot", "none"}
        self.node_feats = node_feats
        self.edge_feats = edge_feats
        self.max_num_atoms = max_num_atoms

    def encode_node_rich(self, atom_num, atom) -> list:
        """Standard atom featurisation.

        ``label``/``onehot`` describe an atom by its element alone, which is far
        leaner than any published DTI baseline uses. This adds the usual
        OGB/DeepChem descriptors: connectivity, charge, hydrogens, hybridisation,
        aromaticity, ring membership and chirality.
        """
        return (
            _bucket(atom_num, list(encd["drug"]["node"]))
            + _bucket(atom.GetDegree(), DEGREES)
            + _bucket(atom.GetFormalCharge(), CHARGES)
            + _bucket(atom.GetTotalNumHs(), NUM_HS)
            + _bucket(str(atom.GetHybridization()), HYBRIDIZATIONS)
            + _bucket(str(atom.GetChiralTag()), CHIRALITIES)
            + [int(atom.GetIsAromatic()), int(atom.IsInRing())]
        )

    def encode_node(self, atom_num, atom):
        """Encode single atom"""
        if atom_num not in encd["drug"]["node"].keys():
            atom_num = "other"

        if self.node_feats == "glycan":
            if atom_num in encd["glycan"]:
                return encd["glycan"][atom_num] + encd["chirality"][atom.GetChiralTag()]
            else:
                return encd["glycan"]["other"] + encd["chirality"][atom.GetChiralTag()]

        if self.node_feats == "rich":
            return self.encode_node_rich(atom_num, atom)

        label = encd["drug"]["node"][atom_num]
        if self.node_feats == "onehot":
            return onehot_encode(label, len(encd["drug"]["node"]))
        return label + 1

    def encode_edge(self, edge):
        """Encode single edge"""
        label = encd["drug"]["edge"][edge]
        if self.edge_feats == "onehot":
            return onehot_encode(label, len(encd["drug"]["edge"]))
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
            if btype not in encd["drug"]["edge"].keys():
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
    drug_enc = DrugEncoder(
        snakemake.params.node_feats,
        snakemake.params.edge_feats,
        snakemake.params.max_num_atoms,
    )
    ligs = pd.read_csv(snakemake.input.lig, sep="\t").set_index("Drug_ID")
    ligs["data"] = ligs["Drug"].apply(drug_enc)
    ligs = ligs[ligs["data"].notna()]
    ligs = ligs.to_pickle(snakemake.output.pickle)
