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


class PosNoise:
    """Add Gaussian noise to the coordinates of the nodes in a graph."""

    def __init__(self, sigma: float = 0.5):
        self.sigma = sigma

    def __call__(self, batch) -> torch.Tensor:
        noise = torch.randn_like(batch.pos) * self.sigma
        batch.pos += noise
        batch.noise = noise
        return batch


class MaskType:
    """Masks the type of the nodes in a graph."""

    def __init__(self, prob: float):
        self.prob = prob

    def __call__(self, batch) -> torch.Tensor:
        mask = torch.rand_like(batch.x, dtype=torch.float32) < self.prob
        batch.orig_x = batch.x[mask]
        batch.x[mask] = 20
        batch.mask = mask
        return batch
