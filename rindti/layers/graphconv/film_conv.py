from argparse import ArgumentParser

from torch.functional import Tensor
from torch_geometric.nn import FiLMConv
from torch_geometric.typing import Adj

from ..base_layer import BaseLayer


class FilmConvNet(BaseLayer):
    """FiLMConv
    https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.FiLMConv
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32, num_edge_dim=None, **kwargs):
        super().__init__()
        self.conv1 = FiLMConv(input_dim, hidden_dim, num_relations=num_edge_dim)
        self.conv2 = FiLMConv(hidden_dim, output_dim, num_relations=num_edge_dim)

    def forward(self, x: Tensor, edge_index: Adj, edge_feats: Tensor, **kwargs) -> Tensor:
        """Forward pass of the module"""
        x = self.conv1(x, edge_index, edge_feats)
        x = self.conv2(x, edge_index, edge_feats)
        return x

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Generate arguments for this module"""

        parser.add_argument("node_embed", default="filmconv", type=str)
        parser.add_argument("hidden_dim", default=32, type=int, help="Number of hidden dimensions")
        return parser
