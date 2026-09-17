from torch import Tensor
from torch.nn import ModuleList
from torch_geometric.nn import GATConv
from torch_geometric.typing import Adj

from ..base_layer import BaseLayer, interlayer_activations


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
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.inp = GATConv(input_dim, hidden_dim, heads, concat=False)
        self.mid_layers = ModuleList(
            [GATConv(hidden_dim, hidden_dim, heads, concat=False) for _ in range(num_layers - 2)]
        )

        self.out = GATConv(hidden_dim, output_dim, concat=False)
        self.acts = interlayer_activations(1 + len(self.mid_layers), dropout)

    def forward(self, x: Tensor, edge_index: Adj, **kwargs) -> Tensor:
        """"""
        x = self.acts[0](self.inp(x, edge_index))
        for module, act in zip(self.mid_layers, self.acts[1:], strict=True):
            x = act(module(x, edge_index))
        x = self.out(x, edge_index)
        return x
