from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.nn import TransformerConv

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
        heads: int = 1,
        num_layers: int = 3,
        **kwargs,
    ):
        super().__init__()
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

    def forward(self, data: Batch) -> Tensor:
        """"""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.inp(x, edge_index, edge_attr)
        for module in self.mid_layers:
            x = module(x, edge_index, edge_attr)
        x = self.out(x, edge_index, edge_attr)
        data.x = x
        return data
