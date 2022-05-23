import os
import pickle
from typing import Iterable

import pandas as pd
from pandas.core.frame import DataFrame
from prepare_drugs import edge_encoding as drug_edge_encoding
from prepare_drugs import node_encoding as drug_node_encoding
from utils import prot_edge_encoding, prot_node_encoding


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
    if "index_mapping" in x:
        del x["index_mapping"]
    return x


def update_config(config: dict, prot_size=None, drug_size=None) -> dict:
    """Updates config with dims of everything"""
    config["prot_feat_dim"] = len(prot_node_encoding) if prot_size is None else prot_size
    config["drug_feat_dim"] = len(drug_node_encoding) if drug_size is None else drug_size
    config["prot_edge_dim"] = len(prot_edge_encoding)
    config["drug_edge_dim"] = len(drug_edge_encoding)
    return config


if __name__ == "__main__":

    interactions = pd.read_csv(snakemake.input.inter, sep="\t")

    with open(snakemake.input.drugs, "rb") as file:
        drugs = pickle.load(file)

    with open(snakemake.input.prots, "rb") as file:
        prots = pickle.load(file)

    interactions = interactions[interactions["Target_ID"].isin(prots.index)]
    interactions = interactions[interactions["Drug_ID"].isin(drugs.index)]
    prots = prots[prots.index.isin(interactions["Target_ID"])]
    drugs = drugs[drugs.index.isin(interactions["Drug_ID"])]

    full_data = process_df(interactions)
    config = update_config(
        snakemake.config,
        prot_size=prots.iloc[0]["data"]["x"].size(0),
        drug_size=drugs.iloc[0]["data"]["x"].size(0),
    )

    final_data = {
        "data": full_data,
        "config": config,
        "prots": prots[["data"]],
        "drugs": drugs[["data"]],
    }

    with open(snakemake.output.combined_pickle, "wb") as file:
        pickle.dump(final_data, file, protocol=-1)
