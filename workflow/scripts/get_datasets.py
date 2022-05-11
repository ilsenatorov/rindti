import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import requests
from tdc.multi_pred import DTI
from tqdm import tqdm


def get_float(entry: str):
    """Try to convert value to float."""
    try:
        return float(entry)
    except Exception:
        return np.nan


def count_residues(pdb: str) -> int:
    """Count number of residues in a PDB file."""
    count = 0
    for line in pdb.split("\n"):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            count += 1
    return count


class DatasetFetcher:
    def __init__(
        self,
        dataset_name: str,
        dataset_dir: str = "datasets",
        min_number_aa: int = 0,
        max_number_aa: int = float("inf"),
    ):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.min_number_aa = min_number_aa
        self.max_number_aa = max_number_aa
        self.dataset_folder = f"{dataset_dir}/{dataset_name}/resources"
        self.structures_folder = f"{self.dataset_folder}/structures"
        self.drugs_folder = f"{self.dataset_folder}/drugs"
        self._create_dirs()

    def _create_dirs(self):
        """Create necessary directories."""
        Path(self.drugs_folder).mkdir(parents=True, exist_ok=True)
        Path(self.structures_folder).mkdir(parents=True, exist_ok=True)

    def _get_glass(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Download the GLASS dataset."""
        inter = pd.read_csv("https://zhanggroup.org/GLASS/downloads/interactions_total.tsv", sep="\t")
        lig = pd.read_csv("https://zhanggroup.org/GLASS/downloads/ligands.tsv", sep="\t")
        inter = inter[inter["Parameter"].isin(["Ki", "IC50", "EC50"])]
        inter = inter.rename(
            {
                "UniProt ID": "Target_ID",
                "InChI Key": "Drug_ID",
                "Value": "Y",
            },
            axis=1,
        )[["Drug_ID", "Target_ID", "Y"]]
        lig = lig.rename(
            {
                "UniProt ID": "Target_ID",
                "InChI Key": "Drug_ID",
                "Value": "Y",
                "Canonical SMILES": "Drug",
            },
            axis=1,
        )[["Drug_ID", "Drug"]]
        inter["Y"] = inter["Y"].apply(get_float)
        return inter, lig

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load the necessary dataset."""
        if self.dataset_name == "BindingDB":
            data = pd.concat([DTI(name=f"BindingDB_{x}").get_data() for x in ["IC50", "Kd", "Ki"]])
        elif self.dataset_name == "glass":
            return self._get_glass()
        elif self.dataset_name.lower() == "davis":
            raise NotImplementedError("Davis dataset is not available yet.")
        else:
            data = DTI(name=self.dataset_name).get_data()
        return data[["Drug_ID", "Target_ID", "Y"]], data.drop_duplicates()[["Drug", "Drug_ID"]]

    def get_pdb(self, pdb_id: str) -> None:
        """Download PDB structure from AlphaFoldDB."""
        if not os.path.exists(f"{self.structures_folder}/{pdb_id}.pdb"):
            response = requests.get(f"https://alphafold.ebi.ac.uk/files/AF-{pdb_id}-F1-model_v2.pdb")
            if response:
                n_res = count_residues(response.text)
                if n_res >= self.min_number_aa and n_res <= self.max_number_aa:
                    with open(f"{self.structures_folder}/{pdb_id}.pdb", "w") as file:
                        file.write(response.text)

    def run(self):
        """Run the script."""
        inter, lig = self.load_data()
        inter = inter[inter["Y"].notna()]
        inter = inter.groupby(["Drug_ID", "Target_ID"]).agg("median").reset_index()
        for i in tqdm(inter["Target_ID"].unique()):
            self.get_pdb(i)
        available_structures = [x.split(".")[0] for x in os.listdir(self.structures_folder)]
        inter = inter[inter["Target_ID"].isin(available_structures)]
        lig = lig[lig["Drug_ID"].isin(inter["Drug_ID"].unique())]

        inter.to_csv(f"{self.drugs_folder}/inter.tsv", sep="\t", index=False)
        lig.to_csv(f"{self.drugs_folder}/lig.tsv", sep="\t", index=False)


if __name__ == "__main__":
    from jsonargparse import CLI

    def run(
        dataset_name: str,
        dataset_dir: str = "datasets",
        min_number_aa: int = 0,
        max_number_aa: int = float("inf"),
    ):
        """Run the script."""
        DatasetFetcher(dataset_name, dataset_dir, min_number_aa, max_number_aa).run()

    cli = CLI(run)
