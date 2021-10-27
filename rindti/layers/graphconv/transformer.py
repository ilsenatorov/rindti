from argparse import ArgumentParser

from torch import Tensor, nn
from torch_geometric.nn import TransformerConv
from torch_geometric.typing import Adj

from ..base_layer import BaseLayer


class TransformerNet(BaseLayer):
    """https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.TransformerConv"""

    def __init__(
        self,
        input_dim,
        output_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        edge_dim: int = None,
        edge_type: str = "none",
        heads: int = 1,
        num_layers: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.edge_type = edge_type
        if edge_type == "none":
            edge_dim = None
        if edge_type == "label":
            self.edge_embed = nn.Embedding(edge_dim + 1, edge_dim)
        self.inp = TransformerConv(
            input_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=False,
        )
        self.mid_layers = nn.ModuleList(
            [
                TransformerConv(
                    hidden_dim,
                    hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    concat=False,
                )
                for _ in range(num_layers - 2)
            ]
        )
        self.out = TransformerConv(hidden_dim, output_dim, heads=1, dropout=dropout, edge_dim=edge_dim, concat=False)

    def forward(self, x: Tensor, edge_index: Adj, edge_feats: Tensor = None, **kwargs) -> Tensor:
        """Forward pass of the module"""
        if self.edge_type == "none":
            edge_feats = None
        elif self.edge_type == "label":
            edge_feats = self.edge_embed(edge_feats)
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
        parser.add_argument("heads", default=1, type=int)
        return parser
