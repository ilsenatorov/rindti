import torch

from rindti.layers.graphconv import ginconv_net

from ..layers import ChebConvNet, FilmConvNet, GatConvNet, GINConvNet, PNAConvNet

default_config = {
    "input_dim": 16,
    "output_dim": 32,
    "hidden_dim": 64,
    "num_heads": 4,
    "K": 1,
    "dropout": 0.2,
    "edge_dim": 6,
    "deg": torch.randint(low=0, high=5, size=(10,)),
}


fake_data = {
    "x": torch.rand(size=(13, 16)),
    "edge_index": torch.randint(low=0, high=5, size=(2, 10)),
    "edge_feats": torch.randint(low=0, high=5, size=(10,)),
    "batch": torch.zeros((13,), dtype=torch.long),
}


class BaseTestGraphConv:
    def test_init(self):
        """Tests .__init__()"""
        self.module(**default_config)

    def test_forward(self):
        """Tests .forward()"""
        module = self.module(**default_config)
        output = module.forward(**fake_data)
        assert output.size(0) == 13
        assert output.size(1) == 32


class TestGINConv(BaseTestGraphConv):
    """GinConv"""

    module = GINConvNet


class TestGATConv(BaseTestGraphConv):
    """GatConv"""

    module = GatConvNet


class TestChebConv(BaseTestGraphConv):
    """ChebConv"""

    module = ChebConvNet


class TestPNAConv(BaseTestGraphConv):
    """PNAConv"""

    module = PNAConvNet


class TestFilmConv(BaseTestGraphConv):
    """FiLMConv"""

    module = FilmConvNet
