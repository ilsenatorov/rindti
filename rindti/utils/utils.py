from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, _ArgumentGroup
from random import randint
from typing import Iterable

import torch


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


class MyArgParser(ArgumentParser):
    """Custom argument parser"""

    def add_argument_group(self, *args, prefix="", **kwargs) -> _ArgumentGroup:
        """Adds an ArgumentsGroup with every argument startin with the prefix

        Args:
            prefix (str, optional): Prefix to begin arguments from. Defaults to "".

        Returns:
            _ArgumentGroup: group
        """
        group = _MyArgumentGroup(self, *args, prefix=prefix, conflict_handler="resolve", **kwargs)
        self._action_groups.append(group)
        return group


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


def fake_data() -> Iterable[torch.Tensor]:
    """Create fake entry for forward function of DTI prediction models

    Returns:
        Iterable[torch.Tensor]: prot and drug features, edge indices and batches
    """
    return [
        torch.randint(low=0, high=5, size=(15,)),
        torch.randint(low=0, high=5, size=(15,)),
        torch.randint(low=0, high=5, size=(2, 10)),
        torch.randint(low=0, high=5, size=(2, 10)),
        torch.zeros((15,), dtype=torch.long),
        torch.zeros((15,), dtype=torch.long),
    ]
