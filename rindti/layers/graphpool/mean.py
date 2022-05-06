from argparse import ArgumentParser

import torch.nn.functional as F
from torch.functional import Tensor
from torch_geometric.nn import global_mean_pool
from torch_geometric.typing import Adj

from ..base_layer import BaseLayer


class MeanPool(BaseLayer):
    """Mean Pooling module
    Simply averages the node features.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor, edge_index: Adj, batch: Tensor, **kwargs) -> Tensor:
        """"""
        pool = global_mean_pool(x, batch)
        return F.normalize(pool, dim=1)
