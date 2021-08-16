import pickle
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame


def process(row: pd.Series) -> dict:
    """Process each interaction, drugs encoded as graphs"""
    split = row["split"]
    return {
        "label": row["Y"],
        "split": split,
        "prot_id": row["Target_ID"],
        "drug_id": row["Drug_ID"],
    }


def process_df(df: DataFrame) -> Iterable[dict]:
    """Apply process() function to each row of the DataFrame"""
    return [process(row) for (_, row) in df.iterrows()]


def del_index_mapping(x: dict) -> dict:
    """Delete 'index_mapping' entry from the dict"""
    del x["index_mapping"]
    return x


if __name__ == "__main__":

    interactions = pd.read_csv(snakemake.input.inter, sep="\t")

    with open(snakemake.input.drugs, "rb") as file:
        drugs = pickle.load(file)

    with open(snakemake.input.proteins, "rb") as file:
        prots = pickle.load(file)
    interactions = interactions[interactions["Target_ID"].isin(prots.index)]
    interactions = interactions[interactions["Drug_ID"].isin(drugs.index)]
    drug_count = interactions["Drug_ID"].value_counts()
    prot_count = interactions["Target_ID"].value_counts()
    prots["count"] = prot_count
    drugs["count"] = drug_count
    prots["data"] = prots["data"].apply(del_index_mapping)
    prots = prots[prots.index.isin(interactions["Target_ID"])]
    drugs = drugs[drugs.index.isin(interactions["Drug_ID"])]
    full_data = process_df(interactions)

    final_data = {
        "data": full_data,
        "config": snakemake.config,
        "prots": prots[["data", "count"]],
        "drugs": drugs[["data", "count"]],
    }
    with open(snakemake.output.combined_pickle, "wb") as file:
        pickle.dump(final_data, file, protocol=-1)
