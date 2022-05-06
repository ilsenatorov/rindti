from argparse import ArgumentParser

from torch.functional import Tensor
from torch.nn import ModuleList
from torch_geometric.nn import FiLMConv
from torch_geometric.typing import Adj

from ..base_layer import BaseLayer


class FilmConvNet(BaseLayer):
    r"""FiLM Convolution

    Refer to :class:`torch_geometric.nn.conv.FiLMConv` for more details.


    Args:
        input_dim (int): Size of the input vector
        output_dim (int): Size of the output vector
        hidden_dim (int, optional): Size of the hidden layer(s). Defaults to 32.
        edge_dim (int, optional): Size of the edge input vector. Defaults to None.
        num_layers (int, optional): Number of layers. Defaults to 10.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 32,
        edge_dim: int = None,
        num_layers: int = 10,
        **kwargs,
    ):
        super().__init__()
        if edge_dim is None:
            edge_dim = 1
        self.edge_dim = edge_dim
        self.inp = FiLMConv(input_dim, hidden_dim, num_relations=edge_dim)
        mid_layers = [FiLMConv(hidden_dim, hidden_dim, num_relations=edge_dim) for _ in range(num_layers - 2)]
        self.mid_layers = ModuleList(mid_layers)

        self.out = FiLMConv(hidden_dim, output_dim, num_relations=edge_dim)

    def forward(self, x: Tensor, edge_index: Adj, edge_feats: Tensor = None, **kwargs) -> Tensor:
        """"""
        if self.edge_dim <= 1:
            edge_feats = None
        x = self.inp(x, edge_index, edge_feats)
        for module in self.mid_layers:
            x = module(x, edge_index, edge_feats)
        x = self.out(x, edge_index, edge_feats)
        return x
