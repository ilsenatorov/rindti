from argparse import ArgumentParser

import torch
from torch.functional import Tensor
from torch.nn import BatchNorm1d, Linear, ModuleList, PReLU, ReLU, Sequential
from torch_geometric.nn import GINConv
from torch_geometric.typing import Adj

from ..base_layer import BaseLayer


class GINConvNet(BaseLayer):
    """Graph Isomorphism Network

    Args:
        input_dim (int): Size of the input vector
        output_dim (int): Size of the output vector
        hidden_dim (int, optional): Size of the hidden vector. Defaults to 32.
        num_layers (int, optional): Total number of layers. Defaults to 3.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, num_layers: int = 3, **kwargs):
        super().__init__()
        self.inp = GINConv(
            Sequential(
                Linear(input_dim, hidden_dim),
                PReLU(),
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim),
            )
        )
        mid_layers = [
            GINConv(
                Sequential(
                    Linear(hidden_dim, hidden_dim),
                    PReLU(),
                    Linear(hidden_dim, hidden_dim),
                    BatchNorm1d(hidden_dim),
                )
            )
            for _ in range(num_layers - 2)
        ]
        self.mid_layers = ModuleList(mid_layers)
        self.out = GINConv(
            Sequential(
                Linear(hidden_dim, hidden_dim),
                PReLU(),
                Linear(hidden_dim, output_dim),
                BatchNorm1d(output_dim),
            )
        )

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
        parser.add_argument("node_embed", default="ginconv", type=str)
        parser.add_argument("hidden_dim", default=32, type=int, help="Number of hidden dimensions")
        parser.add_argument("dropout", default=0.2, type=float, help="Dropout")
        parser.add_argument("num_layers", default=3, type=int, help="Number of convolutional layers")
        return parser
