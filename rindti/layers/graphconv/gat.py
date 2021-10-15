from argparse import ArgumentParser

from torch.functional import Tensor
from torch.nn import ModuleList
from torch_geometric.nn import GATv2Conv
from torch_geometric.typing import Adj

from ..base_layer import BaseLayer


class GatConvNet(BaseLayer):
    """Graph Attention Layer

    Args:
        output_dim (int): Size of the output vector
        hidden_dim (int, optional): Size of the hidden vector. Defaults to 32.
        heads (int, optional): Number of heads for multi-head attention. Defaults to 4.
    """

    def __init__(
        self, input_dim, output_dim: int, hidden_dim: int = 32, heads: int = 4, num_layers: int = 4, **kwargs
    ):
        super().__init__()
        self.inp = GATv2Conv(input_dim, hidden_dim, heads)
        mid_layers = [GATv2Conv(-1, hidden_dim, heads) for _ in range(num_layers - 2)]
        self.mid_layers = ModuleList(mid_layers)
        self.out = GATv2Conv(-1, output_dim, heads=1)

    def forward(self, x: Tensor, edge_index: Adj, **kwargs) -> Tensor:
        """Forward pass of the module"""
        x = self.inp(x, edge_index)
        for module in self.mid_layers:
            x = module(x, edge_index)
        x = self.out(x, edge_index)
        return x

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Generate arguments for this module"""
        parser.add_argument("dropout", default=0.2, type=float, help="Dropout")
        parser.add_argument("hidden_dim", default=32, type=int, help="Number of hidden dimensions")
        parser.add_argument("node_embed", default="gatconv", type=str)
        parser.add_argument("num_layers", default=3, type=int, help="Number of convolutional layers")
        return parser
