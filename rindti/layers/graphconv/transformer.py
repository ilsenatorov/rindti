from argparse import ArgumentParser

from torch.functional import Tensor
from torch.nn import ModuleList
from torch_geometric.nn import TransformerConv
from torch_geometric.typing import Adj

from ..base_layer import BaseLayer


class TransformerNet(BaseLayer):
    def __init__(
        self,
        input_dim,
        output_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        edge_dim: int = None,
        heads: int = 1,
        num_layers: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.edge_dim = edge_dim
        self.inp = TransformerConv(input_dim, hidden_dim, heads=heads, dropout=dropout, edge_dim=edge_dim)
        mid_layers = [
            TransformerConv(-1, hidden_dim, heads=heads, dropout=dropout, edge_dim=edge_dim)
            for _ in range(num_layers - 2)
        ]
        self.mid_layers = ModuleList(mid_layers)
        self.out = TransformerConv(-1, output_dim, heads=1, dropout=dropout, edge_dim=edge_dim)

    def forward(self, x: Tensor, edge_index: Adj, edge_feats: Tensor = None, **kwargs) -> Tensor:
        """Forward pass of the module"""
        if self.edge_dim is None:
            edge_feats = None
        x = self.inp(x, edge_index, edge_feats)
        for module in self.mid_layers:
            x = module(x, edge_index, edge_feats)
        x = self.out(x, edge_index, edge_feats)
        return x

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Generate arguments for this module"""
        parser.add_argument("dropout", default=0.2, type=float, help="Dropout")
        parser.add_argument("hidden_dim", default=32, type=int, help="Number of hidden dimensions")
        parser.add_argument("node_embed", default="gatconv", type=str)
        parser.add_argument("num_layers", default=3, type=int, help="Number of convolutional layers")
        parser.add_argument("dropout", default=0.2, type=float)
        return parser
