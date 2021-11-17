import torch
from pytorch_lightning import LightningModule
from sklearn.preprocessing import LabelEncoder
from torch import Tensor

from ..layers import MLP


class PfamCrossEntropy(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.mlp = MLP(kwargs["hidden_dim"], len(kwargs["fam_list"]))
        self.loss = torch.nn.CrossEntropyLoss()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(kwargs["fam_list"])

    def forward(self, x: Tensor, fams: list) -> Tensor:
        x = self.mlp(x)
        x = self.loss(x, self.label_encoder.transform(fams))
        return x
