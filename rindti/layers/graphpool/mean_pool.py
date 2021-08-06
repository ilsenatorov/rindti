from argparse import ArgumentParser

from torch.functional import Tensor
from torch_geometric.nn import global_mean_pool
from torch_geometric.typing import Adj

from ..base_layer import BaseLayer


class MeanPool(BaseLayer):
    """Mean Pooling module"""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor, edge_index: Adj, batch: Tensor, **kwargs) -> Tensor:
        """Forward pass of the module

        Args:
            x (Tensor): Node features
            edge_index (Adj): Edge information
            batch (Tensor): Batch information

        Returns:
            Tensor: Graph representation vector
        """
        return global_mean_pool(x, batch)

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Generate arguments for this module

        Args:
            parser (ArgumentParser): Parent parser

        Returns:
            ArgumentParser: Updated parser
        """
        parser.add_argument("pool", default="gmt", type=str)
        return parser
