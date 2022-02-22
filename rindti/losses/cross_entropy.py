import torch
from pytorch_lightning import LightningModule
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torchmetrics.functional import accuracy

from ..layers import MLP


class CrossEntropyLoss(LightningModule):
    r"""Simple cross-entropy loss with the added MLP to match dimensions"""

    def __init__(self, **kwargs):
        super().__init__()
        self.mlp = MLP(kwargs["hidden_dim"], len(kwargs["fam_list"]))
        self.loss = torch.nn.CrossEntropyLoss()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(kwargs["fam_list"])

    def forward(self, x: Tensor, y: list) -> Tensor:
        """"""
        x = self.mlp(x)
        y = torch.tensor(self.label_encoder.transform(y), device=self.device, dtype=torch.long)
        loss = self.loss(x, y)
        return dict(
            graph_loss=loss,
            graph_acc=accuracy(x, y),
        )
