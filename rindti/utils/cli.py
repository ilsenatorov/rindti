import itertools
from argparse import ArgumentParser, _ArgumentGroup
from typing import List

import yaml


def remove_arg_prefix(prefix: str, kwargs: dict) -> dict:
    """Removes the prefix from all the args.

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
    """Adds the prefix to all the args. Removes None values and "index_mapping".

    Args:
        prefix (str): prefix to add (`drug_`, `prot_` or `mlp_` usually)
        kwargs (dict): dict of arguments

    Returns:
        dict: Sub-dict of arguments
    """
    return {prefix + k: v for (k, v) in kwargs.items() if k != "index_mapping" and v is not None}


def read_config(filename: str) -> dict:
    """Read in yaml config for training."""
    with open(filename, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def hparams_config(hparams: dict) -> List[dict]:
    """Get all possible combinations of hyperparameters.

    If any entry is a list, it will be expanded to all possible combinations.

    Args:
        hparams (dict): Hyperparameters

    Returns:
        list: List of hyperparameter configurations
    """
    configs = []
    hparams_small = {k: v for k, v in hparams.items() if isinstance(v, list)}
    if hparams_small == {}:
        return [hparams]
    keys, values = zip(*hparams_small.items())
    for v in itertools.product(*values):
        config = hparams.copy()
        config.update(dict(zip(keys, v)))
        configs.append(config)
    return configs
