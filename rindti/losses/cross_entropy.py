import torch
from pytorch_lightning import LightningModule
from torch import Tensor


class CrossEntropyLoss(LightningModule):
    r"""Simple cross-entropy loss with the added MLP to match dimensions."""

    def __init__(self, label_list: list, **kwargs):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        self.label_encoder.fit(label_list)

    def forward(self, x: Tensor, y: list) -> Tensor:
        """"""
        loss = self.loss(x, y)
        return loss
