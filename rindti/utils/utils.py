from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, _ArgumentGroup
from random import randint
from typing import Iterable, Union

import torch

from rindti.utils.data import TwoGraphData


def remove_arg_prefix(prefix: str, kwargs: Union[dict, TwoGraphData]) -> dict:
    """Removes the prefix from all the args
    Args:
        prefix (str): prefix to remove (`drug_`, `prot_` or `mlp_` usually)
        kwargs (dict): dict of arguments

    Returns:
        dict: Sub-dict of arguments
    """
    new_kwargs = {}
    if not isinstance(kwargs, dict):
        kwargs = kwargs.__dict__
    prefix_len = len(prefix)
    for key, value in kwargs.items():
        if key.startswith(prefix):
            new_key = key[prefix_len:]
            if new_key == "x_batch":
                new_key = "batch"
            new_kwargs[new_key] = value
    return new_kwargs


def add_arg_prefix(prefix: str, kwargs: dict) -> dict:
    """Adds the prefix to all the args. Removes None values and 'index_mapping'

    Args:
        prefix (str): prefix to add (`drug_`, `prot_` or `mlp_` usually)
        kwargs (dict): dict of arguments

    Returns:
        dict: Sub-dict of arguments
    """
    if not isinstance(kwargs, dict):
        kwargs = kwargs.__dict__
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
