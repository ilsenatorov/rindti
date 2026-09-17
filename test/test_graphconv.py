import pytest
import torch

from rindti.layers.graphconv import (
    ChebConvNet,
    FilmConvNet,
    GatConvNet,
    GINConvNet,
    TransformerNet,
)

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

    # def test_args(self):
    #     parser = MyArgParser()
    #     self.module.add_arguments(parser)


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


@pytest.mark.parametrize(
    "module",
    [ChebConvNet, GatConvNet, GINConvNet, FilmConvNet, TransformerNet],
)
def test_stack_is_nonlinear(module):
    """A stack of convolutions must not collapse to a single affine map.

    None of these had an activation between layers except GINConvNet, whose
    non-linearity lives inside the GINConv MLP. ChebConvNet was measurably *exactly*
    affine, so three layers had the capacity of one, and a `node.module` ablation
    compared a real GNN against a linear model.
    """
    torch.manual_seed(0)
    dim, n_nodes = 16, 32
    edge_index = torch.randint(low=0, high=n_nodes - 1, size=(2, 120))
    net = module(input_dim=dim, output_dim=dim, hidden_dim=dim, num_layers=3).eval()
    x1, x2 = torch.randn(n_nodes, dim), torch.randn(n_nodes, dim)

    with torch.no_grad():
        # f(x1 + x2) - (f(x1) + f(x2) - f(0)) is zero for any affine f.
        at_zero = net(x=torch.zeros(n_nodes, dim), edge_index=edge_index, edge_feats=None)
        f1 = net(x=x1, edge_index=edge_index, edge_feats=None)
        f2 = net(x=x2, edge_index=edge_index, edge_feats=None)
        both = net(x=x1 + x2, edge_index=edge_index, edge_feats=None)
        defect = (both - (f1 + f2 - at_zero)).abs().mean()
        signal = f1.abs().mean()

    assert (defect / signal) > 1e-3, f"{module.__name__} behaves as an affine map"


@pytest.mark.parametrize(
    "module",
    [ChebConvNet, GatConvNet, GINConvNet, FilmConvNet, TransformerNet],
)
def test_node_dropout_is_wired_up(module):
    """`node.dropout` must actually do something.

    It used to be swallowed by **kwargs in every module except TransformerNet, where it
    configured *attention* dropout instead, so the config key was inert.
    """
    dim, n_nodes = 16, 32
    edge_index = torch.randint(low=0, high=n_nodes - 1, size=(2, 120))
    x = torch.randn(n_nodes, dim)
    net = module(input_dim=dim, output_dim=dim, hidden_dim=dim, num_layers=3, dropout=0.9).train()

    torch.manual_seed(1)
    first = net(x=x, edge_index=edge_index, edge_feats=None)
    torch.manual_seed(2)
    second = net(x=x, edge_index=edge_index, edge_feats=None)

    assert not torch.allclose(first, second), f"{module.__name__} ignores node.dropout"
