from copy import deepcopy
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from torch.functional import Tensor
from torchmetrics.functional import accuracy, auroc, matthews_corrcoef

from rindti.utils.data import TwoGraphData
from rindti.utils.utils import MyArgParser

from ..utils import remove_arg_prefix
from ..utils.data import corrupt_features
from .classification_model import ClassificationModel


class NoisyNodesModel(ClassificationModel):
    """Model for DTI prediction as a classification problem"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        prot_param = remove_arg_prefix("prot_", kwargs)
        drug_param = remove_arg_prefix("drug_", kwargs)
        self.prot_node_pred = self._get_node_embed(prot_param, out_dim=prot_param["feat_dim"])
        self.drug_node_pred = self._get_node_embed(drug_param, out_dim=drug_param["feat_dim"])

    def corrupt_data(
        self,
        orig_data: TwoGraphData,
        prot_frac: float = 0.05,
        drug_frac: float = 0.05,
    ) -> TwoGraphData:
        """Corrupt a TwoGraphData entry

        Args:
            orig_data (TwoGraphData): Original data
            prot_frac (float, optional): Fraction of nodes to corrupt for proteins. Defaults to 0.05.
            drug_frac (float, optional): Fraction of nodes to corrupt for drugs. Defaults to 0.05.

        Returns:
            TwoGraphData: Corrupted data
        """
        # sourcery skip: extract-duplicate-method
        data = deepcopy(orig_data)
        if prot_frac > 0:
            prot_feat, prot_idx = corrupt_features(data["prot_x"], prot_frac, self.device)
            data["prot_x"] = prot_feat
            data["prot_cor_idx"] = prot_idx
        if drug_frac > 0:
            drug_feat, drug_idx = corrupt_features(data["drug_x"], drug_frac, self.device)
            data["drug_x"] = drug_feat
            data["drug_cor_idx"] = drug_idx
        return data

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
        prot_pred = self.prot_node_pred(**prot)
        drug_pred = self.drug_node_pred(**drug)
        return torch.sigmoid(logit), prot_pred, drug_pred

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test

        Args:
            data (TwoGraphData): Input data

        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        cor_data = self.corrupt_data(data, self.hparams.prot_frac, self.hparams.drug_frac)
        prot = remove_arg_prefix("prot_", data)
        drug = remove_arg_prefix("drug_", data)
        output, prot_pred, drug_pred = self.forward(prot, drug)
        labels = data.label.unsqueeze(1)
        if self.hparams.weighted:
            weight = 1 / torch.sqrt(data.prot_count * data.drug_count)
            loss = F.binary_cross_entropy(output, labels.float(), weight=weight.unsqueeze(1))
        else:
            loss = F.binary_cross_entropy(output, labels.float())
        prot_idx = cor_data.prot_cor_idx
        drug_idx = cor_data.drug_cor_idx
        prot_loss = F.cross_entropy(prot_pred[prot_idx], data["prot_x"][prot_idx])
        drug_loss = F.cross_entropy(drug_pred[drug_idx], data["drug_x"][drug_idx])
        acc = accuracy(output, labels)
        try:
            _auroc = auroc(output, labels, pos_label=1)
        except Exception:
            _auroc = torch.tensor(np.nan, device=self.device)
        _mc = matthews_corrcoef(output, labels.squeeze(1), num_classes=2)
        return {
            "loss": loss + self.hparams.prot_alpha * prot_loss + self.hparams.drug_alpha * drug_loss,
            "prot_loss": prot_loss.detach(),
            "drug_loss": drug_loss.detach(),
            "pred_loss": loss.detach(),
            "acc": acc,
            "auroc": _auroc,
            "matthews": _mc,
        }

    @staticmethod
    def add_arguments(parser: MyArgParser) -> MyArgParser:
        """Generate arguments for this module

        Args:
            parser (ArgumentParser): Parent parser

        Returns:
            ArgumentParser: Updated parser
        """
        parser = ClassificationModel.add_arguments(parser)
        drug = parser.get_arg_group("Drug")
        prot = parser.get_arg_group("Prot")
        prot.add_argument("alpha", default=0.1, type=float, help="Prot node loss factor")
        drug.add_argument("alpha", default=0.1, type=float, help="Drug node loss factor")
        prot.add_argument("frac", default=0.05, type=float, help="Proportion of prot nodes to corrupt")
        drug.add_argument("frac", default=0.05, type=float, help="Proportion of drug nodes to corrupt")
        return parser
