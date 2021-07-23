import torch
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, GraphSizeNorm
from torch.nn import ModuleList

from ..base_layer import BaseLayer


class GINConvNet(BaseLayer):

    @staticmethod
    def add_arguments(group):
        group.add_argument('node_embed', default='ginconv', type=str)
        group.add_argument('hidden_dim', default=32, type=int, help='Number of hidden dimensions')
        group.add_argument('dropout', default=0.2, type=float, help='Dropout')
        group.add_argument('num_layers', default=3, type=int, help='Number of convolutional layers')
        return group

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim=64,
                 num_layers=3,
                 **kwargs):
        super().__init__()
        self.inp = GINConv(Sequential(
            Linear(input_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim)
        ))
        mid_layers = [GINConv(Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                BatchNorm1d(hidden_dim)
            )) for _ in range(num_layers-2)]
        self.mid_layers = ModuleList(mid_layers)
        self.out = GINConv(Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, output_dim),
            BatchNorm1d(output_dim)
        ))

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor,
                **kwargs):
        x = self.inp(x, edge_index)
        for module in self.mid_layers:
            x = module(x, edge_index)
        x = self.out(x, edge_index)
        return x
