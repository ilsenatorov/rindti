import torch
import torch.nn.functional as F
from torch.functional import Tensor
from torchmetrics.functional import explained_variance, mean_squared_error, pearson_corrcoef

from ..utils import remove_arg_prefix
from ..utils.data import TwoGraphData
from .classification import ClassificationModel


class RegressionModel(ClassificationModel):
    """Model for DTI prediction as a regression problem"""

    def forward(self, prot: dict, drug: dict) -> Tensor:
        """Forward pass of the model"""
        prot["x"] = self.prot_feat_embed(prot["x"])
        drug["x"] = self.drug_feat_embed(drug["x"])
        prot["x"] = self.prot_node_embed(**prot)
        drug["x"] = self.drug_node_embed(**drug)
        prot_embed = self.prot_pool(**prot)
        drug_embed = self.drug_pool(**drug)
        joint_embedding = self.merge_features(drug_embed, prot_embed)
        logit = self.mlp(joint_embedding)
        return torch.sigmoid(logit)

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test
        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        prot = remove_arg_prefix("prot_", data)
        drug = remove_arg_prefix("drug_", data)
        output = self.forward(prot, drug)
        labels = data.label.unsqueeze(1).float()
        loss = F.mse_loss(output, labels)
        corr = pearson_corrcoef(output, labels)
        mse = mean_squared_error(output, labels)
        expvar = explained_variance(output, labels)
        return {
            "loss": loss,
            "corr": corr,
            "mse": mse,
            "expvar": expvar,
        }
