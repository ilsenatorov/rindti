from os import stat
from ..base_layer import BaseLayer
from torch import nn


class MLP(BaseLayer):
    """
    Simple multi-layer perceptron.
    :param input_dim: int, number of input neurons
    :param hidden_dim: int, number of neurons in hidden layers
    :param num_layers: int, number of hidden layers
    :param dropout: float, dropout probability
    :param out_dim: int, number of output neurons (2 for classification, 1 for regression)
    """

    def __init__(self,
                 input_dim=64,
                 hidden_dim=128,
                 out_dim=2,
                 num_layers=0,
                 dropout=0.2,
                 **kwargs):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        for i in range(num_layers):
            self.mlp.add_module('hidden_linear{}'.format(i), nn.Linear(hidden_dim, hidden_dim))
            self.mlp.add_module('hidden_relu{}'.format(i), nn.ReLU())
            self.mlp.add_module('hidden_dropout{}'.format(i), nn.Dropout(dropout))
        self.mlp.add_module('final_linear', nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.mlp(x)
