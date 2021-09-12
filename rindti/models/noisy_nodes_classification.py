from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.functional import Tensor

from rindti.utils.data import TwoGraphData
from rindti.utils.utils import MyArgParser

from ..utils import remove_arg_prefix
from ..utils.data import corrupt_features, mask_features
from .classification import ClassificationModel


class NoisyNodesClassModel(ClassificationModel):
    """Model for DTI prediction as a classification problem"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        prot_param = remove_arg_prefix("prot_", kwargs)
        drug_param = remove_arg_prefix("drug_", kwargs)
        self.prot_node_pred = self._get_node_embed(prot_param, out_dim=prot_param["feat_dim"])
        self.drug_node_pred = self._get_node_embed(drug_param, out_dim=drug_param["feat_dim"])

    def _corrupt_data(
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
            prot_feat, prot_idx = corrupt_features(data["prot_x"], prot_frac)
            data["prot_x"] = prot_feat
            data["prot_cor_idx"] = prot_idx
        if drug_frac > 0:
            drug_feat, drug_idx = corrupt_features(data["drug_x"], drug_frac)
            data["drug_x"] = drug_feat
            data["drug_cor_idx"] = drug_idx
        return data

    def _mask_data(
        self,
        orig_data: TwoGraphData,
        prot_frac: float = 0.05,
        drug_frac: float = 0.05,
    ) -> TwoGraphData:
        """Corrupt a TwoGraphData entry

        Args:
            orig_data (TwoGraphData): Original data
            prot_frac (float, optional): Fraction of nodes to mask for proteins. Defaults to 0.05.
            drug_frac (float, optional): Fraction of nodes to mask for drugs. Defaults to 0.05.

        Returns:
            TwoGraphData: Masked data
        """
        # sourcery skip: extract-duplicate-method
        data = deepcopy(orig_data)
        if prot_frac > 0:
            prot_feat, prot_idx = mask_features(data["prot_x"], prot_frac)
            data["prot_x"] = prot_feat
            data["prot_cor_idx"] = prot_idx
        if drug_frac > 0:
            drug_feat, drug_idx = mask_features(data["drug_x"], drug_frac)
            data["drug_x"] = drug_feat
            data["drug_cor_idx"] = drug_idx
        return data

    def corrupt_data(
        self,
        orig_data: TwoGraphData,
        prot_frac: float = 0.05,
        drug_frac: float = 0.05,
    ) -> TwoGraphData:
        if self.hparams.corruption == "mask":
            return self._mask_data(orig_data, prot_frac, drug_frac)
        elif self.hparams.corruption == "corrupt":
            return self._corrupt_data(orig_data, prot_frac, drug_frac)
        else:
            raise ValueError("Unknown corruption parameter")

    def forward(self, prot: dict, drug: dict) -> Tensor:
        """Forward pass of the model"""
        prot_embed, prot_pred = self.prot_encoder(prot, return_nodes=True)
        drug_embed, drug_pred = self.drug_encoder(drug, return_nodes=True)
        joint_embedding = self.merge_features(drug_embed, prot_embed)
        pred = self.mlp(joint_embedding)
        prot["x"] = prot_pred
        prot_pred = self.prot_node_pred(**prot)
        drug["x"] = drug_pred
        drug_pred = self.drug_node_pred(**drug)
        return pred, torch.softmax(prot_pred, dim=1), torch.softmax(drug_pred, dim=1)

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test

        Args:
            data (TwoGraphData): Input data

        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        cor_data = self.corrupt_data(data, self.hparams.prot_frac, self.hparams.drug_frac)
        prot = remove_arg_prefix("prot_", cor_data)
        drug = remove_arg_prefix("drug_", cor_data)
        output, prot_pred, drug_pred = self.forward(prot, drug)
        output = torch.sigmoid(output)
        labels = data.label.unsqueeze(1)
        loss = F.binary_cross_entropy(output, labels.float())
        prot_idx = cor_data.prot_cor_idx
        drug_idx = cor_data.drug_cor_idx
        prot_loss = F.cross_entropy(prot_pred[prot_idx], data["prot_x"][prot_idx])
        drug_loss = F.cross_entropy(drug_pred[drug_idx], data["drug_x"][drug_idx])
        metrics = self._get_classification_metrics(output, labels)
        metrics.update(
            dict(
                loss=loss + self.hparams.prot_alpha * prot_loss + self.hparams.drug_alpha * drug_loss,
                prot_loss=prot_loss.detach(),
                drug_loss=drug_loss.detach(),
                pred_loss=loss.detach(),
            )
        )
        return metrics

    @staticmethod
    def add_arguments(parser: MyArgParser) -> MyArgParser:
        """Generate arguments for this module

        Args:
            parser (ArgumentParser): Parent parser

        Returns:
            ArgumentParser: Updated parser
        """
        parser = ClassificationModel.add_arguments(parser)
        parser.add_argument("--corruption", type=str, default="corrupt", help="'corrupt' or 'mask'")
        drug = parser.get_arg_group("Drug")
        prot = parser.get_arg_group("Prot")
        prot.add_argument("alpha", default=0.1, type=float, help="Prot node loss factor")
        drug.add_argument("alpha", default=0.1, type=float, help="Drug node loss factor")
        prot.add_argument("frac", default=0.05, type=float, help="Proportion of prot nodes to corrupt")
        drug.add_argument("frac", default=0.05, type=float, help="Proportion of drug nodes to corrupt")
        return parser
