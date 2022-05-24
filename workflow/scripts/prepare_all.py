import pickle
from typing import Iterable

import pandas as pd
from pandas.core.frame import DataFrame
from prepare_drugs import edge_encoding as drug_edge_encoding
from prepare_drugs import node_encoding as drug_node_encoding
from torch import FloatTensor, LongTensor
from utils import prot_edge_encoding, prot_node_encoding

encodings = dict(
    prot=dict(node=prot_node_encoding, edge=prot_edge_encoding),
    drug=dict(node=drug_node_encoding, edge=drug_edge_encoding),
)


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


def get_type(data: dict, key: str) -> str:
    """Check which type of data we have."""
    feat = data.get(key)
    if isinstance(feat, LongTensor):
        return "label"
    if isinstance(feat, FloatTensor):
        return "onehot"
    if feat is None:
        return "none"
    raise ValueError("Unknown data type {}".format(type(data[key])))


def max_nodes(df: pd.DataFrame) -> int:
    """Return the maximum number of nodes in the dataset."""
    nnodes = df["data"].apply(lambda x: x["x"].size(0))
    return nnodes.max()


def feat_type(df: pd.DataFrame) -> str:
    """Return the type of the features."""
    return get_type(df["data"][0], "x")


def edge_type(df: pd.DataFrame) -> str:
    """Return the type of the edges."""
    return get_type(df["data"][0], "edge_feats")


def feat_dim(df: pd.DataFrame, which: str) -> int:
    """Return the dimension of the features."""
    ftype = feat_type(df)
    if ftype == "label":
        return len(encodings[which]["node"])
    else:
        return df["data"][0]["x"].size(1)


def edge_dim(df: pd.DataFrame, which: str) -> int:
    """Return the dimension of the edges."""
    ftype = edge_type(df)
    if ftype == "label":
        return len(encodings[which]["edge"])
    elif ftype == "onehot":
        return df["data"][0]["edge_feats"].size(1)
    else:
        return 0


def get_config(df: pd.DataFrame, which: str) -> dict:
    """Return the config of the features."""
    return {
        "max_nodes": max_nodes(df),
        "edge_type": edge_type(df),
        "feat_type": feat_type(df),
        "feat_dim": feat_dim(df, which),
        "edge_dim": edge_dim(df, which),
    }


if __name__ == "__main__":

    interactions = pd.read_csv(snakemake.input.inter, sep="\t")

    with open(snakemake.input.drugs, "rb") as file:
        drugs = pickle.load(file)

    with open(snakemake.input.prots, "rb") as file:
        prots = pickle.load(file)

    interactions = interactions[interactions["Target_ID"].isin(prots.index)]
    interactions = interactions[interactions["Drug_ID"].isin(drugs.index)]
    prots = prots[prots.index.isin(interactions["Target_ID"].unique())]
    drugs = drugs[drugs.index.isin(interactions["Drug_ID"].unique())]
    prot_count = interactions["Target_ID"].value_counts()
    drug_count = interactions["Drug_ID"].value_counts()
    prots["data"] = prots.apply(lambda x: {**x["data"], "count": prot_count[x.name]}, axis=1)
    drugs["data"] = drugs.apply(lambda x: {**x["data"], "count": drug_count[x.name]}, axis=1)

    full_data = process_df(interactions)
    snakemake.config["data"] = {
        "prot": get_config(prots, "prot"),
        "drug": get_config(drugs, "drug"),
    }

    final_data = {
        "data": full_data,
        "config": snakemake.config,
        "prots": prots,
        "drugs": drugs,
    }

    with open(snakemake.output.combined_pickle, "wb") as file:
        pickle.dump(final_data, file, protocol=-1)
