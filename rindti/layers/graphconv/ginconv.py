from torch import Tensor, nn
from torch_geometric.nn import GINConv
from torch_geometric.typing import Adj

from ..base_layer import BaseLayer


class GINConvNet(BaseLayer):
    """Graph Isomorphism Network.

    Refer to :class:`torch_geometric.nn.conv.GINConv` for more details.

    Args:
        input_dim (int): Size of the input vector
        output_dim (int): Size of the output vector
        hidden_dim (int, optional): Size of the hidden vector. Defaults to 32.
        num_layers (int, optional): Total number of layers. Defaults to 3.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0,
        **kwargs,
    ):
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
        # Unlike the other modules GINConv already carries its own PReLU, so only the
        # dropout is added here - `node.dropout` was accepted and discarded before.
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(1 + len(mid_layers))])
        self.out = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.PReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.BatchNorm1d(output_dim),
            )
        )

    def forward(self, x: Tensor, edge_index: Adj, **kwargs) -> Tensor:
        """"""
        x = self.dropouts[0](self.inp(x, edge_index))
        for module, drop in zip(self.mid_layers, self.dropouts[1:], strict=True):
            x = drop(module(x, edge_index))
        x = self.out(x, edge_index)
        return x
