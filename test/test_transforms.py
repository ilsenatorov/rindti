import pytest
import torch
from torch_geometric.data import Data

from rindti.data.transforms import DataCorruptor, SizeFilter


@pytest.fixture
def graph() -> Data:
    data = {
        "x": torch.randint(1, 10, (10,)),
        "edge_index": torch.tensor(
            [
                [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                [1, 0, 2, 1, 3, 2, 4, 3, 5, 4],
            ]
        ),
    }
    return Data(**data)


def test_size_filter(graph: Data):
    """Test SizeFilter"""
    sf = SizeFilter(5, 15)
    assert sf(graph)


@pytest.mark.parametrize("which", ["mask", "corrupt"])
def test_corruptor(which: str, graph: Data):
    dc = DataCorruptor({"x": 0.5}, which)
    orig_feats = graph["x"].clone()
    corrdata = dc(graph)
    feats = corrdata["x"]
    if which == "mask":
        assert feats[feats == 0].size(0) == 5
    else:
        assert feats[feats == orig_feats].size(0) >= 5
