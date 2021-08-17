from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from torch.functional import Tensor
from torchmetrics.functional import accuracy, auroc, matthews_corrcoef

from ..layers import (
    MLP,
    ChebConvNet,
    DiffPoolNet,
    FilmConvNet,
    GatConvNet,
    GINConvNet,
    GMTNet,
    MeanPool,
    NoneNet,
    PNAConvNet,
)
from ..utils import remove_arg_prefix
from ..utils.data import TwoGraphData
from .base_model import BaseModel

node_embedders = {
    "ginconv": GINConvNet,
    "chebconv": ChebConvNet,
    "gatconv": GatConvNet,
    "filmconv": FilmConvNet,
    "pnaconv": PNAConvNet,
    "none": NoneNet,
}
poolers = {"gmt": GMTNet, "diffpool": DiffPoolNet, "mean": MeanPool}


class ClassificationModel(BaseModel):
    """Model for DTI prediction as a classification problem"""

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self._determine_feat_method(**kwargs)
        drug_param = remove_arg_prefix("drug_", kwargs)
        prot_param = remove_arg_prefix("prot_", kwargs)
        mlp_param = remove_arg_prefix("mlp_", kwargs)
        self.prot_feat_embed = self._get_feat_embed(prot_param)
        self.drug_feat_embed = self._get_feat_embed(drug_param)
        self.prot_node_embed = self._get_node_embed(prot_param)
        self.drug_node_embed = self._get_node_embed(drug_param)
        self.prot_pool = self._get_pooler(prot_param)
        self.drug_pool = self._get_pooler(drug_param)
        self.mlp = self._get_mlp(mlp_param)

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
        labels = data.label.unsqueeze(1)
        if self.hparams.weighted:
            weight = 1 / torch.sqrt(prot["count"] * drug["count"])
            loss = F.binary_cross_entropy(output, labels.float(), weight=weight.unsqueeze(1))
        else:
            loss = F.binary_cross_entropy(output, labels.float())
        acc = accuracy(output, labels)
        try:
            _auroc = auroc(output, labels, pos_label=1)
        except Exception:
            _auroc = torch.tensor(np.nan, device=self.device)
        _mc = matthews_corrcoef(output, labels.squeeze(1), num_classes=2)
        return {
            "loss": loss,
            "acc": acc,
            "auroc": _auroc,
            "matthews": _mc,
        }

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> ArgumentParser:
        """Generate arguments for this module"""
        # Hack to find which embedding are used and add their arguments
        tmp_parser = ArgumentParser(add_help=False)
        tmp_parser.add_argument("--drug_node_embed", type=str, default="ginconv")
        tmp_parser.add_argument("--prot_node_embed", type=str, default="ginconv")
        tmp_parser.add_argument("--prot_pool", type=str, default="gmt")
        tmp_parser.add_argument("--drug_pool", type=str, default="gmt")
        args = tmp_parser.parse_known_args()[0]

        prot_node_embed = node_embedders[args.prot_node_embed]
        drug_node_embed = node_embedders[args.drug_node_embed]
        prot_pool = poolers[args.prot_pool]
        drug_pool = poolers[args.drug_pool]
        prot = parser.add_argument_group("Prot", prefix="--prot_")
        drug = parser.add_argument_group("Drug", prefix="--drug_")
        prot.add_argument("feat_embed_dim", default=32, type=int)
        drug.add_argument("feat_embed_dim", default=32, type=int)
        ## Add module-specific embeddings
        prot_node_embed.add_arguments(prot)
        drug_node_embed.add_arguments(drug)
        prot_pool.add_arguments(prot)
        drug_pool.add_arguments(drug)
        return parser
