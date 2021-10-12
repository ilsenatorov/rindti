from argparse import ArgumentParser

from torch import nn
from torch.functional import Tensor

from ..base_layer import BaseLayer


class MLP(BaseLayer):
    """Lazy Multi-layer perceptron

    Args:
        output_dim (int): Size of the output vector
        hidden_dim (int, optional): Size of the hidden vector. Defaults to 32.
        num_layers (int, optional): Total Number of layers. Defaults to 2.
        dropout (float, optional): Dropout ratio. Defaults to 0.2.
    """

    def __init__(
        self,
        out_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__()

        self.mlp = nn.Sequential(nn.LazyLinear(hidden_dim), nn.ReLU(), nn.Dropout(dropout))

        for i in range(num_layers - 2):
            self.mlp.add_module("hidden_linear{}".format(i), nn.LazyLinear(hidden_dim))
            self.mlp.add_module("hidden_relu{}".format(i), nn.ReLU())
            self.mlp.add_module("hidden_dropout{}".format(i), nn.Dropout(dropout))
        self.mlp.add_module("final_linear", nn.LazyLinear(out_dim))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the module"""
        return self.mlp(x)

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Generate arguments for this module"""
        parser.add_argument("hidden_dim", default=32, type=int, help="Number of hidden dimensions")
        parser.add_argument("dropout", default=0.2, type=float, help="Dropout")
        parser.add_argument("num_layers", default=3, type=int, help="Number of convolutional layers")
        return parser
