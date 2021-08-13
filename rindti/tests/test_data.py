import torch
from torch._C import dtype

from ..utils.data import TwoGraphData


class TestTwoGraphData:
    """Tests TwoGraphData"""

    def test_n_nodes_label(self):
        """Tests .n_nodes if data is torch.long"""
        tgd = TwoGraphData(a_x=torch.tensor([0, 1, 2], dtype=torch.long))
        assert tgd.n_nodes("a_") == 3

    def test_n_nodes_onehot(self):
        """Tests.n_nodes if data is torch.float32 (onehot)"""
        tgd = TwoGraphData(a_x=torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.float32))
        assert tgd.n_nodes("a_") == 3

    def test_n_edges(self):
        """Tests .n_edges"""
        tgd = TwoGraphData(a_edge_index=torch.tensor([[0, 1], [1, 0]]), dtype=torch.long)
        assert tgd.n_edges("a_") == 2

    def test_n_edge_feats_label(self):
        """Tests .n_edge_feats if data is torch.long"""
        tgd = TwoGraphData(a_edge_feats=torch.tensor(list(range(16))), dtype=torch.long)
        assert tgd.n_edge_feats("a_") == 16

    def test_n_edge_feats_onehot(self):
        """Tests .n_edge_feats if data is torch.float32 (onehot)"""
        tgd = TwoGraphData(a_edge_feats=torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0]]), dtype=torch.long)
        assert tgd.n_edge_feats("a_") == 4

    def test_n_edge_feats_none(self):
        """Tests .n_edge_feats if data is None"""
        tgd = TwoGraphData(a_edge_feats=None)
        assert tgd.n_edge_feats("a_") == 1

    def test_n_edge_feats_gone(self):
        """Tests .n_edge_feats if data is gone"""
        tgd = TwoGraphData()
        assert tgd.n_edge_feats("a_") == 1
