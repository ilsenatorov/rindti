from argparse import ArgumentParser
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.functional import Tensor

from rindti.models.bgrl import BGRLModel
from rindti.models.encoder import Encoder

from ..data import TwoGraphData
from ..layers.base_layer import BaseLayer
from ..utils import remove_arg_prefix
from .base_model import BaseModel, node_embedders, poolers
from .graphlog import GraphLogModel
from .infograph import InfoGraphModel
from .pfam import PfamModel


class ClassificationModel(BaseModel):
    """Model for DTI prediction as a class problem"""

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)
        self._determine_feat_method(**kwargs)
        drug_param = remove_arg_prefix("drug_", kwargs)
        prot_param = remove_arg_prefix("prot_", kwargs)
        mlp_param = remove_arg_prefix("mlp_", kwargs)
        if prot_param["pretrain"]:
            self.prot_encoder = self._load_pretrained(prot_param["pretrain"])
            self.hparams.prot_lr *= 0.001
        else:
            self.prot_encoder = Encoder(**prot_param)
        if drug_param["pretrain"]:
            self.drug_encoder = self._load_pretrained(drug_param["pretrain"])
            self.hparams.drug_lr *= 0.001
        else:
            self.drug_encoder = Encoder(**drug_param)
        self.mlp = self._get_mlp(mlp_param)

    def _load_pretrained(self, checkpoint_path: str) -> Iterable[BaseLayer]:
        """Load pretrained model

        Args:
            checkpoint_path (str): Path to checkpoint file.
            Has to contain 'infograph', 'graphlog', 'pfam' or 'bgrl', which will point to the type of model.

        Returns:
            Iterable[BaseLayer]: feat_embed, node_embed, pool of the pretrained model
        """
        if "infograph" in checkpoint_path:
            return InfoGraphModel.load_from_checkpoint(checkpoint_path).encoder
        elif "graphlog" in checkpoint_path:
            return GraphLogModel.load_from_checkpoint(checkpoint_path).encoder
        elif "pfam" in checkpoint_path:
            return PfamModel.load_from_checkpoint(checkpoint_path).encoder
        elif "bgrl" in checkpoint_path:
            return BGRLModel.load_from_checkpoint(checkpoint_path).student_encoder
        else:
            raise ValueError(
                """Unknown pretraining model type!
                Please ensure 'pfam', 'graphlog', 'bgrl' or 'infograph' are present in the model path"""
            )

    def forward(self, prot: dict, drug: dict) -> Tensor:
        """Forward pass of the model"""
        prot_embed = self.prot_encoder(prot)
        drug_embed = self.drug_encoder(drug)
        joint_embedding = self.merge_features(drug_embed, prot_embed)
        return self.mlp(joint_embedding)

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test
        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        prot = remove_arg_prefix("prot_", data)
        drug = remove_arg_prefix("drug_", data)
        output = torch.sigmoid(self.forward(prot, drug))
        labels = data.label.unsqueeze(1)
        loss = F.binary_cross_entropy(output, labels.float())
        metrics = self._get_class_metrics(output, labels)
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
        prot.add_argument("lr", type=float, default=0.0005)
        drug.add_argument("lr", type=float, default=0.0005)
        ## Add module-specific embeddings
        prot_node_embed.add_arguments(prot)
        drug_node_embed.add_arguments(drug)
        prot_pool.add_arguments(prot)
        drug_pool.add_arguments(drug)
        return parser
