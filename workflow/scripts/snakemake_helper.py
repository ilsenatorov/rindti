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

    # Keys that name a location rather than describe the data, so they must not
    # influence the name: the same dataset built from a different directory is the
    # same dataset.
    IGNORED_KEYS = ("source", "target")

    def _letters(self, config: dict) -> list[tuple[str, str]]:
        """The (key, value) pairs that contribute a letter to a name, in a stable order.

        Sorted, not insertion-ordered: YAML preserves the order keys were written in,
        so reordering two lines in a config with no semantic change would otherwise
        produce a different name and orphan every artifact built before the reorder.
        """
        flat = flatten_config(config)
        return [(k, v) for k, v in sorted(flat.items()) if k not in self.IGNORED_KEYS and isinstance(v, str)]

    def hash_config(self, config: dict) -> str:
        """Hash a config dictionary.

        ``sort_keys=True`` for the same reason ``_letters`` sorts: without it the hash
        follows the order the keys happen to be written in.
        """
        as_json = json.dumps(config, sort_keys=True).encode("utf-8")
        return hashlib.md5(as_json).hexdigest()[: self.cutoff]

    def get_name(self, config: dict) -> str:
        """Get the name of a config.
        All the string entries are concatenated and the hash is appended.
        """
        return "".join(v[0] for _, v in self._letters(config)) + "_" + self.hash_config(config)

    def explain_name(self, config: dict) -> str:
        """Explain config name"""
        print(f"{'Letter'.center(10)} # {'Value'.center(10)} # {'Key'.center(30)}")
        print("#" * 56)
        for k, v in self._letters(config):
            print(f"{v[0].center(10)} # {v.center(10)} # {k.center(30)}")
        return self.get_name(config)

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
