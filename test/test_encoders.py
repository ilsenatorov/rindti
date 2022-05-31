import pytest
import torch

from rindti.layers.encoder import GraphEncoder


@pytest.fixture
def config():
    return {
        "hidden_dim": 16,
        "node": {"module": "ginconv", "hidden_dim": 16},
        "pool": {"module": "gmt", "hidden_dim": 16},
    }


@pytest.fixture(params=["label", "onehot"])
def node_features(request):
    if request.param == "label":
        return torch.randint(0, 10, (10,)), request.param
    elif request.param == "onehot":
        return torch.eye(10)[torch.randint(0, 10, (10,))], request.param


@pytest.fixture(params=["label", "onehot", "none"])
def edge_features(request):
    if request.param == "label":
        return torch.randint(0, 10, (10,)), request.param
    elif request.param == "onehot":
        return torch.eye(10)[torch.randint(0, 10, (10,))], request.param
    elif request.param == "none":
        return None, request.param


@pytest.fixture
def batch(node_features, edge_features) -> dict:
    """torch_geometric batch with edge_index and batch"""
    node_features, node_type = node_features
    edge_features, edge_type = edge_features
    config = {
        "feat_dim": 10,
        "feat_type": node_type,
        "edge_dim": 10,
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
        "edge_attr": edge_features,
        "batch": torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
    }
    return data, config


def test_graph_encoder(batch, config):
    batch, data_config = batch
    config["data"] = data_config
    encoder = GraphEncoder(**config)
    encoder(batch)
