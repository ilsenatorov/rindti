import torch

from ..base_layer import BaseLayer


class NoneNet(BaseLayer):

    def __init__(self,
                 **kwargs):
        super().__init__()

    def forward(self,
                x: torch.Tensor,
                **kwargs):
        return x
