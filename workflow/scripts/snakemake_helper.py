import hashlib
import json
import os

import pandas as pd


def flatten_config(config: dict) -> dict:
    """Flatten a config dictionary."""
    df = pd.json_normalize(config).T
    return df[0].to_dict()


class Namer:
    """Assist in naming the files"""

    def __init__(self, cutoff: int = None):
        self.cutoff = cutoff

    def hash_config(self, config: dict) -> str:
        """Hash a config dictionary."""
        as_json = json.dumps(config).encode("utf-8")
        return hashlib.md5(as_json).hexdigest()[: self.cutoff]

    def get_name(self, config: dict) -> str:
        """Get the name of a config.
        All the string entries are concatenated and the hash is appended.
        """
        flat = flatten_config(config)
        res = ""
        for k, v in flat.items():
            if k == "source":
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
        self.config = config
        self._set_inputs()

    def _set_inputs(self):
        self.source_dir = self.config["source"]
        self.target_dir = "/".join(self.source_dir.split("/")[:-1] + ["results"])
        structures = self._source("structures")
        if not os.path.isdir(structures):
            raise FileNotFoundError(f"Missing the structures directory {structures}")
        # splitext, not split("."): Davis target IDs contain dots
        # (e.g. "RSK1(KinDom.1-N-terminal)"), and truncating at the first one
        # produced ids whose files do not exist. The distance_based rule then had
        # unsatisfiable inputs and snakemake silently fell back to the esm rule,
        # quietly changing the protein featurisation method.
        self.prot_ids = [os.path.splitext(x)[0] for x in os.listdir(structures) if x.endswith(".pdb")]
        if not self.prot_ids:
            nested = [d for d in os.listdir(structures) if os.path.isdir(os.path.join(structures, d))]
            hint = (
                f" It contains the subdirector{'y' if len(nested) == 1 else 'ies'} "
                f"{nested} - the PDBs are probably one level too deep."
                if nested
                else ""
            )
            raise ValueError(f"No .pdb files found in {structures}.{hint}")
        self.raw_structs = [self._source("structures", x + ".pdb") for x in self.prot_ids]
        self.tables = {k: self._source("tables", k + ".tsv") for k in ["inter", "lig", "prot"]}

    def _source(self, *args) -> str:
        return os.path.join(self.source_dir, *args)

    def _target(self, *args) -> str:
        return os.path.join(self.target_dir, *args)
