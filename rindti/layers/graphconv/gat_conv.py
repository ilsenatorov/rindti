from argparse import ArgumentParser
from torch.functional import Tensor
from torch_geometric.nn import GATv2Conv, GraphSizeNorm
from torch_geometric.typing import Adj

from ..base_layer import BaseLayer


class GatConvNet(BaseLayer):
    """Graph Attention Layer

    Args:
        input_dim (int): Size of the input vector
        output_dim (int): Size of the output vector
        hidden_dim (int, optional): Size of the hidden vector. Defaults to 32.
        heads (int, optional): Number of heads for multi-head attention. Defaults to 4.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32, heads: int = 4, **kwargs):
        super().__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads)
        self.conv2 = GATv2Conv(hidden_dim * heads, output_dim, heads=1)
        self.norm = GraphSizeNorm()

    def forward(self, x: Tensor, edge_index: Adj, batch: Tensor, **kwargs) -> Tensor:
        """Forward pass of the module

        Args:
            x (Tensor): Node features
            edge_index (Adj): Edge information
            batch (Tensor): Batch information

        Returns:
            Tensor: Updated node features
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.norm(x, batch)
        return x

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Generate arguments for this module

        Args:
            parser (ArgumentParser): Parent parser

        Returns:
            ArgumentParser: Updated parser
        """
        parser.add_argument("node_embed", default="gatconv", type=str)
        parser.add_argument("hidden_dim", default=32, type=int, help="Number of hidden dimensions")
        parser.add_argument("dropout", default=0.2, type=float, help="Dropout")
        return parser
