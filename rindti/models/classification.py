from argparse import ArgumentParser
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.functional import Tensor

from ..layers.base_layer import BaseLayer
from ..utils import remove_arg_prefix
from ..utils.data import TwoGraphData
from .base_model import BaseModel, node_embedders, poolers
from .graphlog import GraphLogModel
from .infograph import InfoGraphModel
from .pfam import PfamModel


class ClassificationModel(BaseModel):
    """Model for DTI prediction as a classification problem"""

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self._determine_feat_method(**kwargs)
        drug_param = remove_arg_prefix("drug_", kwargs)
        prot_param = remove_arg_prefix("prot_", kwargs)
        mlp_param = remove_arg_prefix("mlp_", kwargs)
        if prot_param["pretrain"]:
            self.prot_feat_embed, self.prot_node_embed, self.prot_pool = self._load_pretrained(prot_param["pretrain"])
        else:
            self.prot_feat_embed = self._get_feat_embed(prot_param)
            self.prot_node_embed = self._get_node_embed(prot_param)
            self.prot_pool = self._get_pooler(prot_param)
            self.prot_feat_embed.requires_grad = False
            self.prot_node_embed.requires_grad = False
            self.prot_pool.requires_grad = False
        if drug_param["pretrain"]:
            self.drug_feat_embed, self.drug_node_embed, self.drug_pool = self._load_pretrained(drug_param["pretrain"])
        else:
            self.drug_feat_embed = self._get_feat_embed(drug_param)
            self.drug_node_embed = self._get_node_embed(drug_param)
            self.drug_pool = self._get_pooler(drug_param)
            self.drug_feat_embed.requires_grad = False
            self.drug_node_embed.requires_grad = False
            self.drug_pool.requires_grad = False
        self.mlp = self._get_mlp(mlp_param)

    def _load_pretrained(self, checkpoint_path: str) -> Iterable[BaseLayer]:
        """Load pretrained model

        Args:
            checkpoint_path (str): Path to checkpoint file.
            Has to contain 'infograph' or 'graphlog', which will point to the type of model.

        Returns:
            Iterable[BaseLayer]: feat_embed, node_embed, pool of the pretrained model
        """
        if "infograph" in checkpoint_path:
            model = InfoGraphModel.load_from_checkpoint(checkpoint_path)
        elif "graphlog" in checkpoint_path:
            model = GraphLogModel.load_from_checkpoint(checkpoint_path)
        elif "pfam" in checkpoint_path:
            model = PfamModel.load_from_checkpoint(checkpoint_path)
        else:
            raise ValueError(
                """Unknown pretraining model type!
                Please ensure 'pfam', 'graphlog' or 'infograph' are present in the model path"""
            )
        return model.feat_embed, model.node_embed, model.pool

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
        metrics = self._get_classification_metrics(output, labels)
        metrics.update(dict(loss=loss))
        return metrics

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
        prot.add_argument("pretrain", default=None, type=str)
        drug.add_argument("pretrain", default=None, type=str)
        ## Add module-specific embeddings
        prot_node_embed.add_arguments(prot)
        drug_node_embed.add_arguments(drug)
        prot_pool.add_arguments(prot)
        drug_pool.add_arguments(drug)
        return parser
