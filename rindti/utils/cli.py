import collections
import itertools

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
    with open(filename, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def tree():
    """Defaultdict of defaultdicts"""
    return collections.defaultdict(tree)


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
        keys, values = zip(*hparams_small.items())
        for v in itertools.product(*values):
            config = self.flat.copy()
            config.update(dict(zip(keys, v)))
            configs.append(config)
        return configs

    def _unflatten(self, d: dict):
        root = tree()
        for k, v in d.items():
            parts = k.split(",")
            curr = root
            for part in parts[:-1]:
                curr = curr[part]
            part = parts[-1]
            curr[part] = v
        return root

    def __call__(self, d: dict):
        self._flatten(d)
        variants = self._get_variants()
        return [self._unflatten(v) for v in variants]


def get_git_hash():
    """Get the git hash of the current repository."""
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha
