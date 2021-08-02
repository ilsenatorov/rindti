from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, _ArgumentGroup
from random import randint

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


def get_name(**kwargs):
    """
    Create name for run from kwargs
    """
    prot = "PROT={prot_embed}:{prot_embed_dim}:{prot_hidden_dim}:{prot_dropout}".format(**kwargs)
    drug = "DRUG={drug_embed}:{drug_embed_dim}:{drug_hidden_dim}:{drug_dropout}".format(**kwargs)
    mlp = "MLP={mlp_num_layers}:{mlp_hidden_dim}:{mlp_dropout}:{feat_method}".format(**kwargs)
    return "_".join([prot, drug, mlp])


def create_numeric_mapping(node_properties):
    """
    Create node feature map.
    :param node_properties: List of features sorted.
    :return : Feature numeric map.
    """
    return {value: i for i, value in enumerate(node_properties)}


class MyArgParser(ArgumentParser):
    def add_argument_group(self, *args, prefix="", **kwargs):
        group = _MyArgumentGroup(self, *args, prefix=prefix, conflict_handler="resolve", **kwargs)
        self._action_groups.append(group)
        return group


class _MyArgumentGroup(_ArgumentGroup):
    def __init__(self, *args, prefix="", **kwargs):
        self.prefix = prefix
        super().__init__(*args, **kwargs)

    def add_argument(self, name, **kwargs):
        name = self.prefix + name
        super().add_argument(name, **kwargs)


def combine_parameters(params):
    return torch.cat([param.view(-1) for param in params])


def fake_data():
    return [
        torch.randint(low=0, high=5, size=(15,)),
        torch.randint(low=0, high=5, size=(15,)),
        torch.randint(low=0, high=5, size=(2, 10)),
        torch.randint(low=0, high=5, size=(2, 10)),
        torch.zeros((15,), dtype=torch.long),
        torch.zeros((15,), dtype=torch.long),
    ]
