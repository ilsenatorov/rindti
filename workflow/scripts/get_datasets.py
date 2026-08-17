# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "PyTDC>=1.1",
#     # PyTDC still imports pkg_resources, which setuptools 81 removed.
#     "setuptools<81",
#     "pandas>=2.2",
#     "numpy>=1.26,<2",
#     "requests>=2.32",
#     "tqdm>=4.66",
#     "jsonargparse>=4.28",
# ]
# ///
"""Download DTI datasets and their AlphaFold structures.

PyTDC pins ``numpy<2``, which is incompatible with the modern torch stack that
the rest of RINDTI needs, so this script declares its own dependencies and is
meant to be run in an isolated environment::

    uv run workflow/scripts/get_datasets.py davis --min_num_aa 100

It is a one-off preprocessing step; nothing at training time imports it.
"""

import os
import shutil
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tdc.multi_pred import DTI
from tqdm import tqdm
from urllib3.util.retry import Retry


def get_float(entry: str):
    """Try to convert value to float."""
    try:
        return float(entry)
    except Exception:
        return np.nan


def count_residues(pdb: str) -> int:
    """Count num of residues in a PDB file."""
    count = 0
    for line in pdb.split("\n"):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            count += 1
    return count


class DatasetFetcher:
    """Download the dataset and pdb files."""

    def __init__(
        self,
        dataset_name: Literal[
            "davis",
            "kiba",
            "glass",
            "BindingDB",
            "BindingDB_Kd",
            "BindingDB_Ki",
            "BindingDB_IC50",
        ],
        dataset_dir: str = "datasets",
        min_num_aa: int = 0,
        max_num_aa: int = float("inf"),
        download_structures: bool = False,
        structure_cache: str = None,
    ):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.min_num_aa = min_num_aa
        self.max_num_aa = max_num_aa
        self.download_structures = download_structures
        # Datasets overlap heavily in their targets (the three BindingDB variants
        # especially), so a shared cache avoids re-fetching the same AlphaFold
        # model once per dataset.
        self.structure_cache = Path(structure_cache) if structure_cache else None
        if self.structure_cache:
            self.structure_cache.mkdir(parents=True, exist_ok=True)
        self.session = self._make_session()
        # Lowercase the directory: the TDC dataset name is "BindingDB" but every
        # dataset directory and the configs that point at them are lowercase, and
        # a case mismatch is invisible until the pipeline finds no tables.
        self.dataset_folder = f"{dataset_dir}/{dataset_name.lower()}/resources"
        self.structures_folder = f"{self.dataset_folder}/structures"
        self.tables_folder = f"{self.dataset_folder}/tables"
        self._create_dirs()

    @staticmethod
    def _make_session() -> requests.Session:
        """Session that retries on the transient failures a few thousand
        AlphaFold requests will inevitably hit."""
        session = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
        return session

    def _create_dirs(self):
        """Create necessary directories."""
        Path(self.tables_folder).mkdir(parents=True, exist_ok=True)
        Path(self.structures_folder).mkdir(parents=True, exist_ok=True)

    def _get_glass(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Download the GLASS dataset."""
        colnames = {
            "UniProt ID": "Target_ID",
            "InChI Key": "Drug_ID",
            "Canonical SMILES": "Drug",
            "Value": "Y",
            "FASTA Sequence": "Target",
        }
        inter = pd.read_csv("https://zhanggroup.org/GLASS/downloads/interactions_total.tsv", sep="\t")
        lig = pd.read_csv("https://zhanggroup.org/GLASS/downloads/ligands.tsv", sep="\t")
        prot = pd.read_csv("https://zhanggroup.org/GLASS/downloads/targets.tsv", sep="\t")
        inter = inter[inter["Parameter"].isin(["Ki", "IC50", "EC50"])]
        inter = inter.rename(
            colnames,
            axis=1,
        )[["Drug_ID", "Target_ID", "Y"]]
        lig = lig.rename(
            colnames,
            axis=1,
        )[["Drug_ID", "Drug"]]
        prot = prot.rename(colnames, axis=1)[["Target_ID", "Target"]]
        inter["Y"] = inter["Y"].apply(get_float)
        return inter, lig, prot

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load the necessary dataset."""
        if self.dataset_name == "BindingDB":
            # Merges three different assay quantities into one Y column, which
            # run() then medians per pair. IC50 is assay-dependent and not directly
            # comparable to Kd/Ki, and it dominates by volume. Use BindingDB_Kd (or
            # _Ki / _IC50) for a single-quantity dataset.
            data = pd.concat([DTI(name=f"BindingDB_{x}").get_data() for x in ["IC50", "Kd", "Ki"]])
        elif self.dataset_name == "glass":
            return self._get_glass()
        elif self.dataset_name.lower() == "davis":
            data = DTI("Davis").get_data()
        else:
            data = DTI(name=self.dataset_name).get_data()
        return (
            data[["Drug_ID", "Target_ID", "Y"]],
            data[["Drug", "Drug_ID"]].drop_duplicates(),
            data[["Target", "Target_ID"]].drop_duplicates(),
        )

    def get_pdb(self, pdb_id: str) -> bool:
        """Download a predicted structure from AlphaFoldDB.

        The file URL is resolved through the API rather than built by hand, so
        this keeps working as AlphaFoldDB bumps its model version.

        Returns:
            True if a structure is now present or the target definitively has
            none; False if the attempt failed transiently and is worth retrying.
        """
        target = Path(f"{self.structures_folder}/{pdb_id}.pdb")
        if target.exists():
            return True

        cached = self.structure_cache / f"{pdb_id}.pdb" if self.structure_cache else None
        if cached is not None and cached.exists():
            # A zero-byte marker records a target AlphaFold has no model for, or
            # one that failed the length filter, so it is not requested again.
            if cached.stat().st_size:
                shutil.copyfile(cached, target)
            return True

        try:
            meta = self.session.get(f"https://alphafold.ebi.ac.uk/api/prediction/{pdb_id}", timeout=30)
            text = None
            if meta.ok and meta.json():
                response = self.session.get(meta.json()[0]["pdbUrl"], timeout=60)
                response.raise_for_status()
                if self.min_num_aa <= count_residues(response.text) <= self.max_num_aa:
                    text = response.text
        except (requests.RequestException, ValueError):
            # Transient: never cache a negative marker here, or a network blip
            # would permanently blacklist a target that does have a model.
            return False

        if text is not None:
            target.write_text(text)
        if cached is not None:
            cached.write_text(text or "")
        return True

    def run(self):
        """Run the script."""
        inter, lig, prot = self.load_data()
        inter = inter[inter["Y"].notna()]
        inter = inter.groupby(["Drug_ID", "Target_ID"]).agg("median").reset_index()
        if self.download_structures:
            pending = list(inter["Target_ID"].unique())
            for attempt in range(3):
                failed = [t for t in tqdm(pending, desc=f"structures (pass {attempt + 1})") if not self.get_pdb(t)]
                if not failed:
                    break
                print(f"{len(failed)} targets failed transiently; retrying")
                pending = failed
            else:
                print(f"Giving up on {len(pending)} targets after 3 passes: {pending[:5]}")
            # splitext, not split("."): target IDs may contain dots, and truncating
            # there drops the targets whose structures were just downloaded.
            available_structures = [os.path.splitext(x)[0] for x in os.listdir(self.structures_folder)]
            inter = inter[inter["Target_ID"].isin(available_structures)]
            prot = prot[prot["Target_ID"].isin(available_structures)]
        lig = lig[lig["Drug_ID"].isin(inter["Drug_ID"].unique())]

        inter.to_csv(f"{self.tables_folder}/inter.tsv", sep="\t", index=False)
        lig.to_csv(f"{self.tables_folder}/lig.tsv", sep="\t", index=False)
        prot.to_csv(f"{self.tables_folder}/prot.tsv", sep="\t", index=False)


if __name__ == "__main__":
    from jsonargparse import CLI

    def run(
        dataset_name: Literal[
            "davis",
            "kiba",
            "glass",
            "BindingDB",
            "BindingDB_Kd",
            "BindingDB_Ki",
            "BindingDB_IC50",
        ],
        dataset_dir: str = "datasets",
        min_num_aa: int = 0,
        max_num_aa: int | float = float("inf"),
        download_structures: bool = False,
        structure_cache: str = None,
    ):
        """Run the script.

        Args:
            structure_cache: directory shared between datasets so overlapping
                targets are fetched from AlphaFold only once.
        """
        DatasetFetcher(
            dataset_name,
            dataset_dir,
            min_num_aa,
            max_num_aa,
            download_structures,
            structure_cache,
        ).run()

    cli = CLI(run)
