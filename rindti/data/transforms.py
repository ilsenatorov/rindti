import pickle
from copy import deepcopy
from math import ceil
from typing import Dict, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data

from .data import TwoGraphData


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
        for _, row in mutations.iterrows():
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
