import pandas as pd
from torch import FloatTensor, Generator, LongTensor
from torch.utils.data.dataset import random_split


def split_random(dataset, fracs: list) -> list:
    """Randomly split dataset."""
    assert abs(sum(fracs) - 1) < 1e-6, "Sum of fractions must be 1"
    tot = len(dataset)
    values = []
    for i in fracs[1:]:
        values.append(int(i * tot))
    values = [tot - sum(values)] + values
    return random_split(dataset, values)


def minmax_normalise(s: pd.Series) -> pd.Series:
    """MinMax normalisation of a pandas series."""
    return (s - s.min()) / (s.max() - s.min())


def to_prob(s: pd.Series) -> pd.Series:
    """Convert to probabilities."""
    return s / s.sum()


def get_type(data: dict, key: str) -> str:
    """Check which type of data we have.

    Args:
        data (dict): TwoGraphData or Data
        key (str): "x" or "prot_x" or "drug_x" usually

    Raises:
        ValueError: If not FloatTensor or LongTensor

    Returns:
        str: "label" for LongTensor, "onehot" for FloatTensor
    """
    feat = data.get(key)
    if isinstance(feat, LongTensor):
        return "label"
    if isinstance(feat, FloatTensor):
        return "onehot"
    if feat is None:
        return "none"
    raise ValueError("Unknown data type {}".format(type(data[key])))
