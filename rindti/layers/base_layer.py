from pytorch_lightning import LightningModule


class BaseLayer(LightningModule):
    """Base class for all layers. This class extends :class:`pytorch_lightning.LightningModule`, refer it for more details."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        """"""
        raise NotImplementedError()
