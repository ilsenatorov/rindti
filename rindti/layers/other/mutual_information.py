from typing import Tuple

import torch
import torch.nn.functional as F
from torch.functional import Tensor

from ..base_layer import BaseLayer
from .mlp import MLP


def shifted_softplus(input):
    """Shifted softplus function."""
    return F.softplus(input) - F.softplus(torch.zeros(1, device=input.device))


class MutualInformation(BaseLayer):
    r"""Estimate MI between two entries. Uses MLP.

    `[paper] <https://arxiv.org/pdf/1808.06670.pdf>_`

    Args:
        input_dim (int): Size of the input vector
        hidden_dim (int): Size of the hidden vector
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.x_mlp = MLP(input_dim, hidden_dim, hidden_dim)
        self.y_mlp = MLP(input_dim, hidden_dim, hidden_dim)

    def forward(self, x: Tensor, y: Tensor, pair_index=None) -> Tuple[Tensor, Tensor]:
        """"""
        x = self.x_mlp(x)
        y = self.y_mlp(y)
        score = x @ y.t()
        score = score.flatten()

        if pair_index is None:
            assert len(x) == len(y)
            pair_index = torch.arange(len(x), device=self.device).unsqueeze(-1).expand(-1, 2)

        index = pair_index[:, 0] * len(y) + pair_index[:, 1]
        positive = torch.zeros_like(score, dtype=torch.bool)
        positive[index] = 1
        negative = ~positive

        return -shifted_softplus(-score[positive]).mean() - shifted_softplus(score[negative]).mean()
