from __future__ import annotations

import collections
import itertools
from collections.abc import Callable
from typing import Any

import git
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
    with open(filename) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def write_config(filename: str, config: dict) -> None:
    """Write a config to a file."""
    with open(filename, "w") as file:
        yaml.dump(config, file)


def _plain(node):
    """Convert the nested defaultdict tree back to plain dicts.

    A defaultdict leaks into whatever consumes the config - Lightning writes it
    into hparams.yaml as a `!!python/object/apply:collections.defaultdict` tag,
    which yaml.safe_load then refuses to read back.
    """
    if isinstance(node, dict):
        return {k: _plain(v) for k, v in node.items()}
    return node


def _tree():
    """Defaultdict of defaultdicts"""
    return collections.defaultdict(_tree)


class IterDict:
    """Returns a list of dicts with all possible combinations of hyperparameters."""

    def __init__(self):
        self.current_path = []
        self.flat = {}

    def _flatten(self, d: dict):
        for k, v in d.items():
            self.current_path.append(k)
            if isinstance(v, dict):
                self._flatten(v)
            else:
                self.flat[",".join(self.current_path)] = v
            self.current_path.pop()

    def _get_variants(self):
        configs = []
        hparams_small = {k: v for k, v in self.flat.items() if isinstance(v, list)}
        if hparams_small == {}:
            return [self.flat]
        keys, values = zip(*hparams_small.items(), strict=True)
        for v in itertools.product(*values):
            config = self.flat.copy()
            config.update(dict(zip(keys, v, strict=True)))
            configs.append(config)
        return configs

    def _unflatten(self, d: dict):
        root = _tree()
        for k, v in d.items():
            parts = k.split(",")
            curr = root
            for part in parts[:-1]:
                curr = curr[part]
            part = parts[-1]
            curr[part] = v
        return _plain(root)

    def __call__(self, d: dict):
        # Reset: the instance keeps state between calls, so reusing one would
        # accumulate keys from every config it had ever seen.
        self.current_path = []
        self.flat = {}
        self._flatten(d)
        variants = self._get_variants()
        return [self._unflatten(v) for v in variants]


def recursive_apply(ob: dict | Any, func: Callable) -> dict | Any:
    """Apply a function to the nested dict recursively."""
    if isinstance(ob, dict):
        return {k: recursive_apply(v, func) for k, v in ob.items()}
    else:
        return func(ob)


def get_git_hash() -> str:
    """Get the git hash of the current repository.

    Returns "unknown" when not run from a checkout, which is the normal case
    for an installed package.
    """
    try:
        return git.Repo(search_parent_directories=True).head.object.hexsha
    except (git.InvalidGitRepositoryError, ValueError):
        return "unknown"
