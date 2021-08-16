from typing import Tuple

import torch
import torch.nn.functional as F
from torch.functional import Tensor

from ..base_layer import BaseLayer
from .mlp import MLP


class MutualInformation(BaseLayer):
    def __init__(self, input_dim: int, hidden_dim: int):
        """Estimate MI between two entries. Uses MLP"""
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
            pair_index = torch.arange(len(x), device=x.device).unsqueeze(-1).expand(-1, 2)

        index = pair_index[:, 0] * len(y) + pair_index[:, 1]
        positive = torch.zeros_like(score, dtype=torch.bool)
        positive[index] = 1
        negative = ~positive

        return -F.shifted_softplus(-score[positive]).mean() - F.shifted_softplus(score[negative]).mean()
