from torch.functional import Tensor
from torch_geometric.nn import ChebConv

from ..base_layer import BaseLayer
from argparse import ArgumentParser
from torch_geometric.typing import Adj


class ChebConvNet(BaseLayer):
    """Chebyshev Convolution

    Args:
        input_dim (int): Size of the input vector
        output_dim (int): Size of the output vector
        hidden_dim (int, optional): Size of the hidden vector. Defaults to 32.
        K (int, optional): K parameter. Defaults to 1.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32, K: int = 1, **kwargs):
        super().__init__()
        self.conv1 = ChebConv(input_dim, hidden_dim, K)
        self.conv2 = ChebConv(hidden_dim, output_dim, K)

    def forward(self, x: Tensor, edge_index: Adj, batch: Tensor, **kwargs) -> Tensor:
        """Forward pass of the module

        Args:
            x (Tensor): Node features
            edge_index (Adj): Edge information
            batch (Tensor): Batch information

        Returns:
            Tensor: Updated node features
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Generate arguments for this module

        Args:
            parser (ArgumentParser): Parent parser

        Returns:
            ArgumentParser: Updated parser
        """
        parser.add_argument("node_embed", default="chebconv", type=str)
        parser.add_argument("hidden_dim", default=32, type=int, help="Number of hidden dimensions")
        parser.add_argument("dropout", default=0.2, type=float, help="Dropout")
        parser.add_argument("K", default=1, type=int, help="K argument of chebconv")
        return parser
