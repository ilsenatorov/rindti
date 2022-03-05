from typing import List

import torch
from torch import Tensor

from ...data import TwoGraphData
from ...losses import CrossEntropyLoss
from ..base_model import BaseModel
from ..encoder import Encoder


class ProtClassModel(BaseModel):
    """Model for basic protein classification. Data in self.forward has to contain data.y, which will be the label we aim to predict"""

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(return_nodes=False, **kwargs)
        self.loss = CrossEntropyLoss(**kwargs)

    def forward(self, data: dict) -> Tensor:
        """"""
        return self.encoder(data)

    def shared_step(self, data: TwoGraphData) -> dict:
        """"""
        embeds = self.forward(data)
        metrics = self.loss(embeds, data.y)
        return {k: v.detach() if k != "loss" else v for k, v in metrics.items()}
