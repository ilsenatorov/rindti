from argparse import ArgumentParser, _ArgumentGroup
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
from matplotlib.figure import Figure
from torch import FloatTensor, LongTensor, Tensor
from torch.utils.data import random_split


def remove_arg_prefix(prefix: str, kwargs: dict) -> dict:
    """Removes the prefix from all the args
    Args:
        prefix (str): prefix to remove (`drug_`, `prot_` or `mlp_` usually)
        kwargs (dict): dict of arguments

    Returns:
        dict: Sub-dict of arguments
    """
    new_kwargs = {}
    prefix_len = len(prefix)
    for key, value in kwargs.items():
        if key.startswith(prefix):
            new_key = key[prefix_len:]
            if new_key == "x_batch":
                new_key = "batch"
            new_kwargs[new_key] = value
    return new_kwargs


def add_arg_prefix(prefix: str, kwargs: dict) -> dict:
    """Adds the prefix to all the args. Removes None values and "index_mapping"

    Args:
        prefix (str): prefix to add (`drug_`, `prot_` or `mlp_` usually)
        kwargs (dict): dict of arguments

    Returns:
        dict: Sub-dict of arguments
    """
    return {prefix + k: v for (k, v) in kwargs.items() if k != "index_mapping" and v is not None}


class _MyArgumentGroup(_ArgumentGroup):
    """Custom arguments group

    Args:
        prefix (str, optional): Prefix to begin arguments from. Defaults to "".

    """

    def __init__(self, *args, prefix="", **kwargs):
        self.prefix = prefix
        super().__init__(*args, **kwargs)

    def add_argument(self, name: str, **kwargs):
        """Add argument with prefix before it

        Args:
            name (str): [description]
        """
        name = self.prefix + name
        super().add_argument(name, **kwargs)


class MyArgParser(ArgumentParser):
    """Custom argument parser"""

    def add_argument_group(self, *args, prefix="", **kwargs) -> _MyArgumentGroup:
        """Adds an ArgumentsGroup with every argument starting with the prefix

        Args:
            prefix (str, optional): Prefix to begin arguments from. Defaults to "".

        Returns:
            _MyArgumentGroup: group
        """
        group = _MyArgumentGroup(self, *args, prefix=prefix, conflict_handler="resolve", **kwargs)
        self._action_groups.append(group)
        return group

    def get_arg_group(self, group_title: str) -> _MyArgumentGroup:
        """Get arg group under this title"""
        for group in self._action_groups:
            if group.title == group_title:
                return group


def split_random(dataset, train_frac: float = 0.8):
    """Randomly split dataset"""
    tot = len(dataset)
    train = int(tot * train_frac)
    val = int(tot * (1 - train_frac))
    return random_split(dataset, [train, val])


def minmax_normalise(s: pd.Series) -> pd.Series:
    """MinMax normalisation of a pandas series"""
    return (s - s.min()) / (s.max() - s.min())


def plot_loss_count_dist(losses: dict) -> Figure:
    """Plot distribution of times sampled vs avg loss of families"""
    fig = plt.figure()
    plt.xlabel("Times sampled")
    plt.ylabel("Avg loss")
    plt.title("Prot statistics")
    count = [len(x) for x in losses.values()]
    mean = [np.mean(x) for x in losses.values()]
    plt.scatter(x=count, y=mean)
    return fig


def to_prob(s: pd.Series) -> pd.Series:
    """Convert to probabilities"""
    return s / s.sum()


def get_type(data: dict, key: str) -> str:
    """Check which type of data we have

    Args:
        data (dict): TwoGraphData or Data
        key (str): "x" or "prot_x" or "drug_x" usually

    Raises:
        ValueError: If not FloatTensor or LongTensor

    Returns:
        str: "label" for LongTensor, "onehot" for FloatTensor
    """
    feat = data.get(key)
    if isinstance(feat, LongTensor):
        return "label"
    if isinstance(feat, FloatTensor):
        return "onehot"
    if feat is None:
        return "none"
    raise ValueError("Unknown data type {}".format(type(data[key])))


def get_node_loss(
    x: Tensor,
    pred_x: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Calculate cross-entropy loss for node prediction"""
    x = x if isinstance(x, LongTensor) else x.argmax(dim=1)
    return F.cross_entropy(pred_x, x)
