from torch import nn


class BaseLayer(nn.Module):
    """Base class for all layers."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        """"""
        raise NotImplementedError()
