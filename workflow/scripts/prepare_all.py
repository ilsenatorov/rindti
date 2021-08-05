import pickle
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame


def process(row: pd.Series) -> dict:
    """Process each interaction, drugs encoded as graphs


    Args:
        row (pd.Series): Row of the DataFrame

    Returns:
        dict: dict with 'label', 'split', 'prot_id', 'drug_id'
    """
    split = row["split"]
    threshold = snakemake.config["prepare_all"]["threshold"]
    if threshold:
        label = 1 if float(row["Value"]) < threshold else 0
    else:
        label = -np.log10(row["Value"]) / 1e9
    return {
        "label": label,
        "split": split,
        "prot_id": row["UniProt ID"],
        "drug_id": row["InChI Key"],
    }


def process_df(df: DataFrame) -> Iterable[dict]:
    """Apply process() function to each row of the DataFrame

    Args:
        df (DataFrame): input df

    Returns:
        Iterable[dict]: list of dicts (outputs of process)
    """
    return [process(row) for (_, row) in df.iterrows()]


def del_index_mapping(x: dict) -> dict:
    """Delete 'index_mapping' entry from the dict

    Args:
        x (dict): dict of values

    Returns:
        dict: dict without 'index_mapping'
    """
    del x["index_mapping"]
    return x


if __name__ == "__main__":

    threshold = snakemake.config["prepare_all"]["threshold"]

    interactions = pd.read_csv(snakemake.input.inter)

    with open(snakemake.input.drugs, "rb") as file:
        drugs = pickle.load(file)

    with open(snakemake.input.proteins, "rb") as file:
        prots = pickle.load(file)
    interactions = interactions[interactions["UniProt ID"].isin(prots.index)]
    interactions = interactions[interactions["InChI Key"].isin(drugs.index)]
    drug_count = interactions["InChI Key"].value_counts()
    prot_count = interactions["UniProt ID"].value_counts()
    prots["count"] = prot_count
    drugs["count"] = drug_count
    prots["data"] = prots["data"].apply(del_index_mapping)
    prots = prots[prots.index.isin(interactions["UniProt ID"])]
    drugs = drugs[drugs.index.isin(interactions["InChI Key"])]
    full_data = process_df(interactions)

    final_data = {
        "data": full_data,
        "config": snakemake.config,
        "prots": prots[["data", "count"]],
        "drugs": drugs[["data", "count"]],
    }
    with open(snakemake.output.combined_pickle, "wb") as file:
        pickle.dump(final_data, file, protocol=-1)
