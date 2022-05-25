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
