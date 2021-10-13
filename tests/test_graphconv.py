import pytest
import torch

from rindti.layers import ChebConvNet, FilmConvNet, GatConvNet, GINConvNet, PNAConvNet
from rindti.utils import MyArgParser


@pytest.fixture
def default_config():
    return {
        "K": 1,
        "deg": torch.randint(low=0, high=5, size=(10,)),
        "dropout": 0.2,
        "edge_dim": 6,
        "hidden_dim": 64,
        "input_dim": 16,
        "num_heads": 4,
        "output_dim": 32,
    }


@pytest.fixture
def fake_data():
    return {
        "batch": torch.zeros((13,), dtype=torch.long),
        "edge_feats": torch.randint(low=0, high=5, size=(10,)),
        "edge_index": torch.randint(low=0, high=5, size=(2, 10)),
        "x": torch.rand(size=(13, 16)),
    }


class BaseTestGraphConv:
    def test_forward(self, default_config, fake_data):
        module = self.module(**default_config)
        output = module.forward(**fake_data)
        assert output.size(0) == 13
        assert output.size(1) == 32

    def test_args(self):
        parser = MyArgParser()
        self.module.add_arguments(parser)


class TestGINConv(BaseTestGraphConv):
    module = GINConvNet


class TestGATConv(BaseTestGraphConv):

    module = GatConvNet


class TestChebConv(BaseTestGraphConv):

    module = ChebConvNet


@pytest.mark.skip
class TestPNAConv(BaseTestGraphConv):

    module = PNAConvNet


class TestFilmConv(BaseTestGraphConv):

    module = FilmConvNet
