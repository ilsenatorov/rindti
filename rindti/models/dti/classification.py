import argparse

import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.functional import Tensor

from ...data import TwoGraphData
from ...layers.other import MLP
from ...utils import remove_arg_prefix
from ..base_model import BaseModel


class ClassificationModel(BaseModel):
    """Model for DTI prediction as a classification problem."""

    def __init__(
        self,
        drug_encoder: LightningModule,
        prot_encoder: LightningModule,
        merge_features: str = "concat",
        num_layers: int = 3,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.drug_encoder = drug_encoder
        self.prot_encoder = prot_encoder
        self._determine_feat_method(
            merge_features,
            drug_encoder.hidden_dim,
            prot_encoder.hidden_dim,
        )
        self.mlp = MLP(self.feat_dim, 1, self.feat_dim, num_layers, dropout)

    def forward(self, prot: dict, drug: dict) -> Tensor:
        """"""
        prot_embed = self.prot_encoder(prot)
        drug_embed = self.drug_encoder(drug)
        joint_embedding = self.merge_features(drug_embed, prot_embed)
        return self.mlp(joint_embedding)

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test.

        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        prot = remove_arg_prefix("prot_", data)
        drug = remove_arg_prefix("drug_", data)
        pred = self.forward(prot, drug)
        labels = data.label.unsqueeze(1)
        bce_loss = F.binary_cross_entropy_with_logits(pred, labels.float())
        return dict(loss=bce_loss)
