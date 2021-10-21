import pytest

from rindti.utils import split_random


@pytest.fixture
def dataset():
    return list(range(50))


@pytest.mark.parametrize("train_frac", [0.7, 0.0, 2.0])
def test_split(train_frac, dataset):
    train, val = split_random(dataset, train_frac)
    assert len(train) + len(val) == len(dataset)
    assert not set(train).intersection(set(val))
