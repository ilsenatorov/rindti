from argparse import ArgumentParser

from torch import nn
from torch.functional import Tensor
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
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.PReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            )
        )
        mid_layers = [
            GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.PReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                )
            )
            for _ in range(num_layers - 2)
        ]
        self.mid_layers = nn.ModuleList(mid_layers)
        self.out = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.PReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.BatchNorm1d(output_dim),
            )
        )

    def forward(self, x: Tensor, edge_index: Adj, **kwargs) -> Tensor:
        """Forward pass of the module"""
        x = self.inp(x, edge_index)
        for module in self.mid_layers:
            x = module(x, edge_index)
        x = self.out(x, edge_index)
        return x
