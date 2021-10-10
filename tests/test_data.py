from random import random

import pytest
import torch

from rindti.data import TwoGraphData

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
