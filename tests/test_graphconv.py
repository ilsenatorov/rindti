from pprint import pprint

import pytest
import torch

from rindti.layers import ChebConvNet, FilmConvNet, GatConvNet, GINConvNet, PNAConvNet, TransformerNet
from rindti.utils import MyArgParser

N_NODES = 10
N_EDGES = 50
INPUT_DIM = 16
HIDDEN_DIM = 64
OUTPUT_DIM = 32
EDGE_DIM = 6


@pytest.fixture
def fake_data(request):
    p = request.param
    if p == "label":
        edge_feats = torch.randint(low=0, high=EDGE_DIM - 1, size=(N_EDGES,))
    elif p == "onehot":
        edge_feats = torch.rand(size=(N_EDGES, EDGE_DIM))
    else:
        edge_feats = None
    return {
        "x": torch.rand(size=(N_NODES, INPUT_DIM)),
        "edge_index": torch.randint(low=0, high=N_NODES - 1, size=(2, N_EDGES)),
        "batch": torch.zeros((N_NODES), dtype=torch.long),
        "edge_feats": edge_feats,
        "type": p,
    }


@pytest.fixture
def default_config(fake_data):
    return {
        "K": 1,
        "deg": torch.randint(low=0, high=5, size=(25,)),
        "dropout": 0.2,
        "edge_dim": EDGE_DIM if fake_data["type"] != "none" else None,
        "hidden_dim": HIDDEN_DIM,
        "input_dim": INPUT_DIM,
        "num_heads": 4,
        "output_dim": OUTPUT_DIM,
    }


class BaseTestGraphConv:
    @pytest.mark.parametrize("fake_data", ["label", "onehot", "none"], indirect=True)
    def test_forward(self, default_config, fake_data):
        module = self.module(**default_config)
        output = module.forward(**fake_data)
        assert output.size(0) == N_NODES
        assert output.size(1) == OUTPUT_DIM

    def test_args(self):
        parser = MyArgParser()
        self.module.add_arguments(parser)


class BaseLabelEdgeConv(BaseTestGraphConv):
    @pytest.mark.parametrize("fake_data", ["label", "none"], indirect=True)
    def test_forward(self, default_config, fake_data):
        module = self.module(**default_config)
        output = module.forward(**fake_data)
        assert output.size(0) == N_NODES
        assert output.size(1) == OUTPUT_DIM


class TestGINConv(BaseTestGraphConv):
    module = GINConvNet


class TestChebConv(BaseTestGraphConv):

    module = ChebConvNet


class TestGATConv(BaseTestGraphConv):

    module = GatConvNet


class TestTransformerConv(BaseTestGraphConv):

    module = TransformerNet


class TestFilmConv(BaseLabelEdgeConv):

    module = FilmConvNet
