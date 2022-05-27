import pandas as pd
from encd import encd
from torch import FloatTensor, LongTensor


def onehot_encode(position: int, count: int) -> list:
    """One-hot encode position
    Args:
        position (int): Which entry to set to 1
        count (int): Max number of entries.
    Returns:
        list: list with zeroes and 1 in <position>
    """
    t = [0] * (count)
    t[position - 1] = 1
    return t


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
        return len(encd[which]["node"])
    else:
        return df["data"][0]["x"].shape[0]


def edge_dim(df: pd.DataFrame, which: str) -> int:
    """Return the dimension of the edges."""
    ftype = edge_type(df)
    if ftype == "label":
        return len(encd[which]["edge"])
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
