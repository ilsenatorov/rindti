from torch.functional import Tensor
from torch.nn import ModuleList
from torch_geometric.nn import GATConv
from torch_geometric.typing import Adj

from ..base_layer import BaseLayer


class GatConvNet(BaseLayer):
    """Graph Attention Layer.

    Refer to :class:`torch_geometric.nn.conv.GATConv` for more details.

    Args:
        input_dim (int): Size of the input vector
        output_dim (int): Size of the output vector
        hidden_dim (int, optional): Size of the hidden vector. Defaults to 32.
        heads (int, optional): Number of heads for multi-head attention. Defaults to 4.
        num_layers (int, optional): Number of layers. Defaults to 4.
    """

    def __init__(
        self,
        input_dim,
        output_dim: int,
        hidden_dim: int = 32,
        heads: int = 4,
        num_layers: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.inp = GATConv(input_dim, hidden_dim, heads, concat=False)
        self.mid_layers = ModuleList(
            [
                GATConv(hidden_dim, hidden_dim, heads, concat=False)
                for _ in range(num_layers - 2)
            ]
        )

        self.out = GATConv(hidden_dim, output_dim, concat=False)

    def forward(self, x: Tensor, edge_index: Adj, **kwargs) -> Tensor:
        """"""
        x = self.inp(x, edge_index)
        for module in self.mid_layers:
            x = module(x, edge_index)
        x = self.out(x, edge_index)
        return x
