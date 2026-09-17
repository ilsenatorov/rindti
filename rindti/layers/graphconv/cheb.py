from torch import Tensor
from torch.nn import ModuleList
from torch_geometric.nn import ChebConv
from torch_geometric.typing import Adj

from ..base_layer import BaseLayer, interlayer_activations


class ChebConvNet(BaseLayer):
    r"""Chebyshev Convolution.

    Refer to :class:`torch_geometric.nn.conv.ChebConv` for more details.

    Args:
        input_dim (int): Size of the input vector
        output_dim (int): Size of the output vector
        hidden_dim (int, optional): Size of the hidden vector. Defaults to 32.
        K (int, optional): K parameter. Defaults to 1.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 32,
        K: int = 1,
        num_layers: int = 4,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.inp = ChebConv(input_dim, hidden_dim, K)
        mid_layers = [ChebConv(hidden_dim, hidden_dim, K) for _ in range(num_layers - 2)]
        self.mid_layers = ModuleList(mid_layers)
        self.out = ChebConv(hidden_dim, output_dim, K)
        self.acts = interlayer_activations(1 + len(mid_layers), dropout)

    def forward(self, x: Tensor, edge_index: Adj, **kwargs) -> Tensor:
        """"""
        x = self.acts[0](self.inp(x, edge_index))
        for module, act in zip(self.mid_layers, self.acts[1:], strict=True):
            x = act(module(x, edge_index))
        x = self.out(x, edge_index)
        return x
