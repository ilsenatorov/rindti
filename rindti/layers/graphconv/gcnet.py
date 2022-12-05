from torch import nn, Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.typing import Adj

from rindti.layers.base_layer import BaseLayer


class GCNet(BaseLayer):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, num_layers: int = 3, **kwargs):
        super().__init__()
        self.inp_layer = GCNConv(in_channels=input_dim, out_channels=hidden_dim)
        self.mid_layers = nn.ModuleList([
            GCNConv(in_channels=hidden_dim, out_channels=hidden_dim) for _ in range(num_layers - 2)
        ])
        self.out_layer = GCNConv(in_channels=hidden_dim, out_channels=output_dim)

    def forward(self, x: Tensor, edge_index: Adj, **kwargs) -> Tensor:
        """Forward the data through the GNN module"""
        x = self.inp_layer(x, edge_index)
        for module in self.mid_layers:
            x = module(x, edge_index)
        x = self.out_layer(x, edge_index)
        return x
