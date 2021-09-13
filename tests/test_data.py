import random

import pytest
import torch

from rindti.utils.data import TwoGraphData, split_random

label_node = torch.tensor([0, 1, 2], dtype=torch.long)
label_edge = torch.tensor(list(range(16)), dtype=torch.long)
onehot_node = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.float32)
onehot_edge = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.float32)


class TestTwoGraphData:
    """Tests TwoGraphData"""

    @pytest.mark.parametrize("variant", [label_node, onehot_node])
    def test_n_nodes(self, variant):
        """Tests .n_nodes if data is torch.long"""
        tgd = TwoGraphData(a_x=variant)
        assert tgd.n_nodes("a_") == 3

    def test_n_edges(self):
        """Tests .n_edges"""
        tgd = TwoGraphData(a_edge_index=torch.tensor([[0, 1], [1, 0]]), dtype=torch.long)
        assert tgd.n_edges("a_") == 2

    @pytest.mark.parametrize("variant,expected", [(onehot_edge, 4), (label_edge, 16), (None, 1)])
    def test_n_edge_feats(self, variant, expected):
        """Tests .n_edge_feats"""
        tgd = TwoGraphData(a_edge_feats=variant)
        assert tgd.n_edge_feats("a_") == expected


@pytest.mark.parametrize("train_frac,val_frac", [(0.7, 0.2), (0.8, 0.0), (0.0, 0.8)])
def test_split(train_frac, val_frac):
    """Test random data split"""
    dataset = list(range(50))
    train, val, test = split_random(dataset, train_frac, val_frac)
    assert len(train) + len(val) + len(test) == len(dataset)
    assert not set(train).intersection(set(val))
    assert not set(train).intersection(set(test))
    assert not set(val).intersection(set(test))


@pytest.mark.parametrize("train_frac", [random.random() for _ in range(10)] + [0, 1])
def test_split_no_test(train_frac):
    """Test data splitting without test set"""
    val_frac = 1 - train_frac
    dataset = list(range(50))
    train, val = split_random(dataset, train_frac, val_frac)
    assert (len(train) + len(val)) == len(dataset)
    assert not set(train).intersection(set(val))


def test_fail_split():
    """Test split fail if fraction sum > 1"""
    with pytest.raises(AssertionError):
        split_random([], 2, 2)
