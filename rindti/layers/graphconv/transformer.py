from torch import Tensor, nn
from torch_geometric.nn import TransformerConv
from torch_geometric.typing import Adj

from ..base_layer import BaseLayer


class TransformerNet(BaseLayer):
    """Transformer Network.

    Refer to :class:`torch_geometric.nn.conv.TransformerConv` for more details.

    Args:
        input_dim (int): Size of the input vector
        output_dim (int): Size of the output vector
        hidden_dim (int, optional): Size of the hidden vector. Defaults to 32.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        edge_dim (int, optional): Size of the edge vector. Defaults to None.
        edge_type (int, optional): Number of edge types. Defaults to "none.
        heads (int, optional): Number of heads. Defaults to 1.
        num_layers (int, optional): Number of layers. Defaults to 3.

    """

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
        """Forward the data through the GNN module"""
        if self.edge_type == "none":
            edge_feats = None
        elif self.edge_type == "label":
            edge_feats = self.edge_embed(edge_feats)
        x = self.inp(x, edge_index, edge_feats)
        for module in self.mid_layers:
            x = module(x, edge_index, edge_feats)
        x = self.out(x, edge_index, edge_feats)
        return x
