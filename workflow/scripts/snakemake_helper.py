import hashlib
import json
import os
import os.path as osp

import pandas as pd


def flatten_config(config: dict) -> dict:
    """Flatten a config dictionary."""
    df = pd.json_normalize(config).T.sort_index()
    return df[0].to_dict()


class Namer:
    """Assist in naming the files"""

    def __init__(self, cutoff: int = None):
        self.cutoff = cutoff

    def hash_config(self, config: dict) -> str:
        """Hash a config dictionary."""
        as_json = json.dumps(config, sort_keys=True).encode("utf-8")
        return hashlib.md5(as_json).hexdigest()[: self.cutoff]

    def get_name(self, config: dict) -> str:
        """Get the name of a config.
        All the string entries are concatenated and the hash is appended.
        """
        flat = flatten_config(config)
        res = ""
        for k, v in flat.items():
            if k in ["source", "target"]:
                continue
            elif isinstance(v, str):
                res += v[0]
        return res + "_" + self.hash_config(config)

    def explain_name(self, config: dict) -> str:
        """Explain config name"""
        print(f"{'Letter'.center(10)} # {'Value'.center(10)} # {'Key'.center(30)}")
        print("#" * 56)
        flat = flatten_config(config)
        res = ""
        for k, v in flat.items():
            if k in ["source", "target"]:
                continue
            elif isinstance(v, str):
                res += v[0]
                print(f"{v[0].center(10)} # {v.center(10)} # {k.center(30)}")
        return res + "_" + self.hash_config(config)

    def __call__(self, config: dict) -> str:
        return self.get_name(config)


class SnakemakeHelper:
    """Helper class for Snakemake."""

    def __init__(self, config: dict, namer_cutoff: int = None):
        self.namer = Namer(namer_cutoff)
        self.source = config["source"]
        self.config = config
        self.target = "/".join(self.source.split("/")[:-1] + ["results"])
        self.prot_ids = [x.split(".")[0] for x in os.listdir(self.source + "/structures") if x.endswith(".pdb")]
        self.raw_structs = [os.path.join(self.source, "structures", x + ".pdb") for x in self.prot_ids]

    def _source(self, *args) -> str:
        return os.path.join(self.source, *args)

    def _target(self, *args) -> str:
        return os.path.join(self.target, *args)
