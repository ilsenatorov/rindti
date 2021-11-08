from typing import Tuple

import torch
import torch.nn.functional as F
from torch import FloatTensor, LongTensor, Tensor

from ..data import DataCorruptor, TwoGraphData
from ..utils import MyArgParser, get_node_loss, remove_arg_prefix
from .classification import ClassificationModel


class NoisyNodesClassModel(ClassificationModel):
    """Model for DTI prediction as a class problem"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prot_encoder.return_nodes = True
        self.drug_encoder.return_nodes = True
        prot_param = remove_arg_prefix("prot_", kwargs)
        drug_param = remove_arg_prefix("drug_", kwargs)
        self.prot_node_pred = self._get_node_embed(prot_param, out_dim=prot_param["feat_dim"])
        self.drug_node_pred = self._get_node_embed(drug_param, out_dim=drug_param["feat_dim"])
        self.corruptor = DataCorruptor(
            dict(prot_x=prot_param["frac"], drug_x=drug_param["frac"]), type=kwargs["corruption"]
        )

    def forward(self, prot: dict, drug: dict) -> Tensor:
        """Forward pass of the model"""
        prot_embed, prot_pred = self.prot_encoder(prot)
        drug_embed, drug_pred = self.drug_encoder(drug)
        joint_embedding = self.merge_features(drug_embed, prot_embed)
        pred = self.mlp(joint_embedding)
        prot["x"] = prot_pred
        prot_pred = self.prot_node_pred(**prot)
        drug["x"] = drug_pred
        drug_pred = self.drug_node_pred(**drug)
        return pred, prot_pred, drug_pred

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test

        Args:
            data (TwoGraphData): Input data

        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        data = self.corruptor(data)
        prot = remove_arg_prefix("prot_", data)
        drug = remove_arg_prefix("drug_", data)
        output, prot_pred, drug_pred = self.forward(prot, drug)
        print(prot_pred)
        print(drug_pred)
        labels = data.label.unsqueeze(1)
        output = torch.sigmoid(output)
        loss = F.binary_cross_entropy(output, labels.float())
        prot_loss = get_node_loss(prot_pred[data["prot_x_idx"]], data["prot_x_orig"])
        drug_loss = get_node_loss(drug_pred[data["drug_x_idx"]], data["drug_x_orig"])
        metrics = self._get_class_metrics(output, labels)
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
