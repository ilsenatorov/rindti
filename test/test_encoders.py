import pytest
import torch

from rindti.layers.encoder import GraphEncoder, VectorEncoder


@pytest.fixture(params=["ginconv", "transformer", "filmconv"])
def config(request):
    """`transformer` is included because it is the only conv that consumes a
    continuous edge attribute - without it the `edge_feats` path is untested."""
    return {
        "hidden_dim": 16,
        "node": {"module": request.param, "hidden_dim": 16, "num_layers": 3},
        "pool": {"module": "mean", "hidden_dim": 16},
    }


@pytest.fixture(params=["label", "onehot"])
def node_features(request):
    if request.param == "label":
        return torch.randint(0, 10, (10,)), request.param
    elif request.param == "onehot":
        return torch.eye(10)[torch.randint(0, 10, (10,))], request.param


@pytest.fixture(params=["label", "onehot", "none"])
def edge_features(request):
    """Shapes and dims mirror what ``workflow/scripts/utils.py`` reports.

    In particular ``none`` means ``edge_dim`` 0, not 10 - getting that wrong hides
    the case the shipped default config actually produces.
    """
    if request.param == "label":
        return torch.randint(0, 10, (10,)), request.param, 10
    elif request.param == "onehot":
        # `edge_feats: distance` is a single continuous value per edge.
        return torch.rand(10, 1), request.param, 1
    elif request.param == "none":
        return None, request.param, 0


@pytest.fixture
def batch(node_features, edge_features) -> dict:
    """torch_geometric batch with edge_index and batch"""
    node_features, node_type = node_features
    edge_features, edge_type, edge_dim = edge_features
    config = {
        "feat_dim": 10,
        "feat_type": node_type,
        "edge_dim": edge_dim,
        "edge_type": edge_type,
        "max_nodes": 10,
    }
    data = {
        "x": node_features,
        "edge_index": torch.tensor(
            [
                [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                [1, 0, 2, 1, 3, 2, 4, 3, 5, 4],
            ]
        ),
        # `edge_feats`, not `edge_attr`: GraphEncoder.forward reads the former,
        # so under the old key the edge path was never exercised by any test.
        "edge_feats": edge_features,
        "batch": torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
    }
    return data, config


def test_graph_encoder(batch, config):
    batch, data_config = batch
    config["data"] = data_config
    encoder = GraphEncoder(**config)
    encoder(batch)


def test_vector_encoder():
    """The `esm` featurisation: one mean-pooled vector per protein, no edge_index.

    GraphEncoder cannot consume this - it dereferences `data["edge_index"]`
    unconditionally - which is why the ESM arm was never runnable.
    """
    batch_size, feat_dim, hidden_dim = 4, 1280, 16
    data = {
        "x": torch.randn(batch_size, feat_dim),
        "batch": torch.arange(batch_size),
    }
    config = {"feat_dim": feat_dim, "feat_type": "onehot", "max_nodes": 1}
    encoder = VectorEncoder(hidden_dim=hidden_dim, data=config)
    assert encoder(data).shape == (batch_size, hidden_dim)


def test_vector_encoder_rejects_label_features():
    """Label features are vocabulary indices, which a Linear cannot consume."""
    config = {"feat_dim": 20, "feat_type": "label", "max_nodes": 1}
    with pytest.raises(ValueError, match="continuous features"):
        VectorEncoder(hidden_dim=16, data=config)
