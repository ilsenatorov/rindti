from pytorch_lightning import LightningModule


class BaseLayer(LightningModule):
    """Base class for all layers"""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        """Forward pass of the module"""
        raise NotImplementedError()
