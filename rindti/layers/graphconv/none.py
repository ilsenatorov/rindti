import torch
from torch.functional import Tensor

from ..base_layer import BaseLayer


class NoneNet(BaseLayer):
    """No changes Net"""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Just return the tensor that it receives

        Args:
            x (Tensor): Node features
        Returns:
            (Tensor): Node features
        """
        return x
