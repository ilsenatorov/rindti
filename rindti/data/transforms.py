import copy
import pickle
from copy import deepcopy
from math import ceil
from typing import Dict, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data

from .data import TwoGraphData


class SizeFilter:
    """Filters out graph that are too big/small."""

    def __init__(self, min_nnodes: int, max_nnodes: int = 0):
        self.min_nnodes = min_nnodes
        self.max_nnodes = max_nnodes

    def __call__(self, data: Data) -> bool:
        """Returns True if number of nodes in given graph is within required values else False."""
        nnodes = data.num_nodes
        return nnodes > self.min_nnodes and nnodes < self.max_nnodes


class NeighborhoodMasker:
    def __init__(self, spots=1, k=1, graphs=None, mode=None):
        self.k = k
        self.spots = spots
        self.graphs = ["prot"] if graphs is None else graphs

    def __call__(self, data: Union[Data, TwoGraphData]):
        for graph in self.graphs:
            mask_ids = []
            candidates = set(np.random.choice(range(len(data[graph + "_x"])), self.spots, replace=False))
            for i in range(self.k):
                new_candidates = set()
                for c in candidates:
                    mask_ids.append(c)
                    for x in data[graph + "_edge_index"][0, (data[graph + "_edge_index"][1] == 170)]:
                        new_candidates.add(x.item())
                candidates = new_candidates
            for c in candidates:
                mask_ids.append(c)

            data[graph + "_masked"] = mask_ids
            data[graph + "_x_orig"] = data[graph + "_x"].clone()
            data[graph + "_x"][mask_ids] = 0

        return data


class DataCorruptor:
    """Corrupt or mask the nodes in a graph (or graph pair).

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

    def __call__(self, data: Union[Data, TwoGraphData]) -> TwoGraphData:
        """Apply corruption.

        Args:
            orig_data (Union[Data, TwoGraphData]): data, has to have attributes that match ones from self.frac

        Returns:
            TwoGraphData: Data with corrupted features
        """
        for k, v in self.frac.items():
            new_feat, idx = self.corr_func(data[k], v)
            data[k + "_orig"] = data[k][idx].detach().clone()
            data[k + "_idx"] = idx
            data[k][idx] = new_feat
        return data


def corrupt_features(features: torch.Tensor, frac: float) -> Tuple[torch.Tensor, list]:
    """Return corrupt features.

    Args:
        features (torch.Tensor): Node features
        frac (float): Fraction of nodes to corrupt

    Returns:
        torch.Tensor, list: New corrupt features, idx of masked nodes
    """
    assert frac >= 0 and frac <= 1, "frac has to between 0 and 1!"
    num_nodes = features.size(0)
    num_corrupt_nodes = ceil(num_nodes * frac)
    idx = list(np.random.choice(range(num_nodes), num_corrupt_nodes, replace=False))
    new = np.random.choice(range(num_nodes), num_corrupt_nodes, replace=False)
    return features[new], idx


def mask_features(features: torch.Tensor, frac: float) -> Tuple[torch.Tensor, list]:
    """Return masked features.

    Args:
        features (torch.Tensor): Node features
        frac (float): Fraction of nodes to mask

    Returns:
        torch.Tensor, list: New masked features, idx of masked nodes
    """
    assert frac >= 0 and frac <= 1, "frac has to between 0 and 1!"
    num_nodes = features.size(0)
    num_corrupt_nodes = ceil(num_nodes * frac)
    idx = list(np.random.choice(range(num_nodes), num_corrupt_nodes, replace=False))
    features = torch.zeros_like(features[idx])
    return features, idx
