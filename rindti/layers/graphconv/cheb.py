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
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 32,
        K: int = 1,
        num_layers: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.inp = ChebConv(input_dim, hidden_dim, K)
        mid_layers = [ChebConv(hidden_dim, hidden_dim, K) for _ in range(num_layers - 2)]
        self.mid_layers = ModuleList(mid_layers)
        self.out = ChebConv(hidden_dim, output_dim, K)

    def forward(self, x: Tensor, edge_index: Adj, **kwargs) -> Tensor:
        """Forward pass of the module"""
        x = self.inp(x, edge_index)
        for module in self.mid_layers:
            x = module(x, edge_index)
        x = self.out(x, edge_index)
        return x
