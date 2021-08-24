import torch

from ..layers import ChebConvNet, FilmConvNet, GatConvNet, GINConvNet, PNAConvNet
from ..utils import MyArgParser

default_config = {
    "K": 1,
    "deg": torch.randint(low=0, high=5, size=(10,)),
    "dropout": 0.2,
    "edge_dim": 6,
    "hidden_dim": 64,
    "input_dim": 16,
    "num_heads": 4,
    "output_dim": 32,
}


fake_data = {
    "batch": torch.zeros((13,), dtype=torch.long),
    "edge_feats": torch.randint(low=0, high=5, size=(10,)),
    "edge_index": torch.randint(low=0, high=5, size=(2, 10)),
    "x": torch.rand(size=(13, 16)),
}


class BaseTestGraphConv:
    def test_forward(self):
        """Tests .forward()"""
        module = self.module(**default_config)
        output = module.forward(**fake_data)
        assert output.size(0) == 13
        assert output.size(1) == 32

    def test_args(self):
        """Test static method of adding args"""
        parser = MyArgParser()
        self.module.add_arguments(parser)


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
