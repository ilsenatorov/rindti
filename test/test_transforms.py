import pytest
import torch
from torch_geometric.data import Data

from rindti.data.transforms import DataCorruptor, SizeFilter, TwoGraphData


def graph(n_nodes: int) -> Data:
    data = {
        "x": torch.eye(10)[torch.randint(0, 10, (n_nodes,))],
        "edge_index": torch.tensor(
            [
                [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                [1, 0, 2, 1, 3, 2, 4, 3, 5, 4],
            ]
        ),
    }
    return Data(**data)


@pytest.mark.parametrize("nnodes, result", [(10, True), (2, False), (20, False)])
def test_size_filter(nnodes, result):
    sf = SizeFilter(5, 15)
    assert sf(graph(nnodes)) == result
