import pickle
import random
from copy import deepcopy
from hashlib import new
from math import ceil
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .data import TwoGraphData
from .utils import add_arg_prefix


class GnomadTransformer:
    """Transformer of TwoGraphData entries

    Args:
        gnomad (dict): Dict of gnomad DataFrames for each protein
        index_mapping (dict): Dict of index mappings for each protein
        encoded_residues (dict): Dict of encoded tensors for each residue
        max_num_mut (int): Upper limit of number of mutations
    """

    def __init__(
        self,
        gnomad: dict,
        index_mapping: dict,
        encoded_residues: dict,
        max_num_mut: int = 50,
    ):
        self.gnomad = gnomad
        self.index_mapping = index_mapping
        self.encoded_residues = encoded_residues
        self.max_num_mut = max_num_mut

    def __call__(self, data: TwoGraphData) -> TwoGraphData:
        """Called within the Dataset class during loading the data

        Args:
            data (TwoGraphData): Prot + Drug entry

        Returns:
            (TwoGraphData): entry with modified protein features
        """
        prot_id = data["prot_id"]
        if prot_id not in self.gnomad:
            return data
        x = data["prot_x"]
        mutations = self.gnomad[prot_id]
        mutations = mutations.sample(frac=1).drop_duplicates("mut_pos")
        num_mut = min(np.random.randint(0, self.max_num_mut), mutations.shape[0])
        if num_mut == 0:
            return data
        mutations = mutations.sample(num_mut)

        new_x = x.detach().clone()
        for i, row in mutations.iterrows():
            position_index = self.index_mapping[prot_id][row["mut_pos"]]
            new_x_row = self.encoded_residues[row["mut_to"].lower()].detach().clone()
            new_x[position_index, :] = new_x_row
        data["prot_x"] = new_x
        return data

    @staticmethod
    def from_pickle(filename: str, max_num_mut=50):
        """Load transformer from pickle

        Args:
            filename (str): Pickle file location
            max_num_mut (int, optional): Maximal number of mutation to apply. Defaults to 50.

        Returns:
            GnomadTransformer: Transformer
        """
        with open(filename, "rb") as file:
            all_data = pickle.load(file)
        return GnomadTransformer(
            all_data["gnomad"],
            all_data["index_mapping"],
            all_data["encoded_residues"],
            max_num_mut=max_num_mut,
        )


class PfamTransformer:
    """For given protein uniprot id, find a negative or positive match

    Args:
        pfams (dict): dictionary, which protein is in which pfam
        pos_balance (float, optional): How often to pick positive matches. Defaults to 0.5.
        min_fam_count (int. optional): Lower threshold for family to pick positives
    """

    def __init__(self, pfams: dict, pos_balance: float = 0.7, min_fam_count: int = 5):
        assert pos_balance >= 0 and pos_balance <= 1, "pos_balance not between 0 and 1!"
        self.pos_balance = pos_balance
        self.pfams = pfams
        self.min_fam_count = min_fam_count

    def __call__(self, data: Data) -> TwoGraphData:
        """Find a matching graph pair for this protein

        Args:
            data (Data): Protein data (has to contain id field for this to work)

        Returns:
            TwoGraphData: all features of original graph become 'a_<feature>', all features of the pair are 'b_<feature>'
        """
        fams = data["fam"].rstrip(";").split(";")
        fam = random.choice(fams)
        new_data = add_arg_prefix("a_", data)
        if len(self.pfams[fam]) <= self.min_fam_count:  # If from a small family only negative samples are allowed
            label = -1
        else:
            label = np.random.choice([1, -1], size=1, p=[self.pos_balance, 1 - self.pos_balance])[0]
        if label == 1:
            new_data.update(add_arg_prefix("b_", self._get_pos_sample(data["id"], fam)))
        else:
            new_data.update(add_arg_prefix("b_", self._get_neg_sample(data["id"], fam)))
        new_data["label"] = torch.tensor(label, dtype=torch.long)
        new_data = TwoGraphData(**new_data)
        new_data.num_nodes = 0
        return new_data

    def _get_pos_sample(self, prot_id: str, fam_id: str) -> dict:
        """Get a positive match for given family (protein from same family)

        Args:
            prot_id (str): Protein id
            fam_id (str): Family id

        Returns:
            dict: data of protein from same family
        """
        pair_prot = random.choice(self.pfams[fam_id])
        if pair_prot["id"] == prot_id:
            return self._get_pos_sample(prot_id, fam_id)
        else:
            return pair_prot

    def _get_neg_sample(self, prot_id: str, fam_id: str) -> dict:
        """Get a negative match for given family (protein from different family)

        Args:
            prot_id (str): Protein id
            fam_id (str): Family id

        Returns:
            dict: data of protein from different family
        """
        other_fam_id = random.choice(list(self.pfams.keys()))
        if other_fam_id == fam_id:
            return self._get_neg_sample(prot_id, fam_id)
        pair_prot = random.choice(self.pfams[other_fam_id])
        return pair_prot


class SizeFilter:
    """Filters out graph that are too big/small"""

    def __init__(self, max_nnodes: int, min_nnodes: int = 0):
        self.max_nnodes = max_nnodes
        self.min_nnodes = min_nnodes

    def __call__(self, data: Data) -> bool:
        """Returns True if number of nodes in given graph is within required values else False"""
        nnodes = data.x.size(0)
        return nnodes > self.min_nnodes and nnodes < self.max_nnodes


class DataCorruptor:
    """Corrupt or mask the nodes in a graph (or graph pair)

    Args:
        frac (Dict[str, float]): dict of which attributes to corrupt ({'x' : 0.05} or {'prot_x' : 0.1, 'drug_x' : 0.2})
        type (str, optional): 'corrupt' or 'mask'. Corrupt puts new values sampled from old, mask puts zeroes. Defaults to 'mask'.
    """

    def __init__(self, frac: Dict[str, float], type: str = "mask"):
        self.type = type
        self.frac = {k: v for k, v in frac.items() if v > 0}
        self._set_corr_func()

    def _set_corr_func(self):
        """Sets the necessary corruption function"""
        if self.type == "mask":
            self.corr_func = mask_features
        elif self.type == "corrupt":
            self.corr_func = corrupt_features
        else:
            raise ValueError("Unknown corruption function type, should be 'mask' or 'corrupt'!")

    def __call__(self, orig_data: Union[Data, TwoGraphData]) -> TwoGraphData:
        """Apply corruption

        Args:
            orig_data (Union[Data, TwoGraphData]): data, has to have attributes that match ones from self.frac

        Returns:
            TwoGraphData: Data with corrupted features
        """
        data = deepcopy(orig_data)
        for k, v in self.frac.items():
            new_feat, idx = self.corr_func(data[k], v)
            data[k] = new_feat
            data[k[:-1] + "idx"] = idx
        return data


def corrupt_features(features: torch.Tensor, frac: float) -> Tuple[torch.Tensor, list]:
    """Corrupt the features

    Args:
        features (torch.Tensor): Node features
        frac (float): Fraction of nodes to corrupt

    Returns:
        torch.Tensor, list: New corrupt features, idx of masked nodes
    """
    assert frac >= 0 and frac <= 1, "frac has to between 0 and 1!"
    num_nodes = features.size(0)
    num_node_types = int(features.max() + 1)
    num_corrupt_nodes = ceil(num_nodes * frac)
    idx = np.random.choice(range(num_nodes), num_corrupt_nodes, replace=False)
    features[idx] = torch.randint_like(features[idx], low=1, high=num_node_types)
    return features, idx


def mask_features(features: torch.Tensor, frac: float) -> Tuple[torch.Tensor, list]:
    """Mask the features

    Args:
        features (torch.Tensor): Node features
        frac (float): Fraction of nodes to mask

    Returns:
        torch.Tensor, list: New masked features, idx of masked nodes
    """
    assert frac >= 0 and frac <= 1, "frac has to between 0 and 1!"
    num_nodes = features.size(0)
    num_corrupt_nodes = ceil(num_nodes * frac)
    idx = np.random.choice(range(num_nodes), num_corrupt_nodes, replace=False)
    features[idx] = torch.zeros_like(features[idx])
    return features, idx
