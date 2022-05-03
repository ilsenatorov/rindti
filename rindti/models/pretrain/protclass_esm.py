from typing import List

import torch
from torch import Tensor

from ...data import TwoGraphData
from ...layers import MLP
from ...losses import CrossEntropyLoss
from ..base_model import BaseModel


class ProtClassESMModel(BaseModel):
    """Model for basic protein classification with ESM. Data in self.forward has to contain data.y, which will be the label we aim to predict"""

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = MLP(1280, kwargs["hidden_dim"], kwargs["hidden_dim"], kwargs["num_layers"], kwargs["dropout"])
        self.loss = CrossEntropyLoss(**kwargs)

    def forward(self, data: dict) -> Tensor:
        """"""
        return self.encoder(data.x.view(-1, 1280))

    def shared_step(self, data: TwoGraphData) -> dict:
        """"""
        embeds = self.forward(data)
        metrics = self.loss(embeds, data.y)
        return {k: v.detach() if k != "loss" else v for k, v in metrics.items()}
