from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from ..base_layer import BaseLayer


class MeanPool(BaseLayer):
    """Mean pooling."""

    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, data: Batch) -> Tensor:
        """"""
        x, batch = data.x, data.batch
        embeds = global_mean_pool(x, batch)
        data.aggr = embeds
        return data
