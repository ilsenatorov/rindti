import torch
from pytorch_lightning import LightningModule
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch._C import dtype

from ..layers import MLP


class PfamCrossEntropyLoss(LightningModule):
    """Simple cross=entropy loss with the added MLP to match dimensions"""

    def __init__(self, **kwargs):
        super().__init__()
        self.mlp = MLP(kwargs["hidden_dim"], len(kwargs["fam_list"]))
        self.loss = torch.nn.CrossEntropyLoss()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(kwargs["fam_list"])

    def forward(self, x: Tensor, y: list) -> Tensor:
        """Forward pass of the module"""
        x = self.mlp(x)
        y = torch.tensor(self.label_encoder.transform(y), device=self.device, dtype=torch.long)
        x = self.loss(x, y)
        return x
