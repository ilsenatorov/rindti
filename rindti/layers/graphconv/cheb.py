from argparse import ArgumentParser

from torch.functional import Tensor
from torch.nn import ModuleList
from torch_geometric.nn import ChebConv
from torch_geometric.typing import Adj

from ..base_layer import BaseLayer


class ChebConvNet(BaseLayer):
    """Chebyshev Convolution

    Args:
        input_dim (int): Size of the input vector
        output_dim (int): Size of the output vector
        hidden_dim (int, optional): Size of the hidden vector. Defaults to 32.
        K (int, optional): K parameter. Defaults to 1.
    """

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 32,
        K: int = 1,
        num_layers: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.inp = ChebConv(-1, hidden_dim, K)
        mid_layers = [ChebConv(-1, hidden_dim, K) for _ in range(num_layers - 2)]
        self.mid_layers = ModuleList(mid_layers)
        self.out = ChebConv(-1, output_dim, K)

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
        parser.add_argument("K", default=1, type=int, help="K argument of chebconv")
        parser.add_argument("hidden_dim", default=32, type=int, help="Number of hidden dimensions")
        parser.add_argument("node_embed", default="chebconv", type=str)
        parser.add_argument("num_layers", default=3, type=int, help="Number of convolutional layers")
        return parser
