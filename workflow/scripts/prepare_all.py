import pickle
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from utils import get_config


def process(row: pd.Series) -> dict:
    """Process each interaction."""
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
    if "index_mapping" in x:
        del x["index_mapping"]
    return x


if __name__ == "__main__":

    interactions = pd.read_csv(snakemake.input.inter, sep="\t")

    with open(snakemake.input.drugs, "rb") as file:
        drugs = pickle.load(file)

    with open(snakemake.input.prots, "rb") as file:
        prots = pickle.load(file)

    interactions = interactions[interactions["Target_ID"].isin(prots.index)]
    # interactions = interactions[interactions["Drug_ID"].isin(drugs.index)]

    prots = prots[prots.index.isin(interactions["Target_ID"].unique())]
    drugs = drugs[drugs.index.isin(interactions["Drug_ID"].unique())]

    prot_count = interactions["Target_ID"].value_counts()
    drug_count = interactions["Drug_ID"].value_counts()

    prots["data"] = prots.apply(lambda x: {**x["data"], "count": prot_count[x.name]}, axis=1)
    # drugs["data"] = drugs.apply(lambda x: {**x["data"], "count": drug_count[x.name]}, axis=1)

    full_data = process_df(interactions)
    snakemake.config["data"] = {}
    snakemake.config["data"]["prot"] = get_config(prots, "prot")
    if "sequence_only" not in snakemake.config or not snakemake.config["sequence_only"]["drugs"]:
        snakemake.config["data"]["drug"] = get_config(drugs, "drug")

    final_data = {
        "data": full_data,
        "config": snakemake.config,
        "prots": prots,
        "drugs": drugs,
    }
    print(len(full_data))
    print(drugs.shape)
    print(prots.shape)

    with open(snakemake.output.combined_pickle, "wb") as file:
        pickle.dump(final_data, file, protocol=-1)
