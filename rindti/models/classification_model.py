from argparse import ArgumentParser

import numpy as np
import torch
from torch.functional import Tensor
import torch.nn.functional as F
from torch.nn import Embedding
from torch_geometric.typing import Adj
from torchmetrics.functional import accuracy, auroc, matthews_corrcoef
from ..utils.data import TwoGraphData

from ..layers import (
    MLP,
    ChebConvNet,
    DiffPoolNet,
    GatConvNet,
    GINConvNet,
    GMTNet,
    MeanPool,
    NoneNet,
)
from ..utils import remove_arg_prefix
from .base_model import BaseModel

node_embedders = {
    "ginconv": GINConvNet,
    "chebconv": ChebConvNet,
    "gatconv": GatConvNet,
    "none": NoneNet,
}
poolers = {"gmt": GMTNet, "diffpool": DiffPoolNet, "mean": MeanPool}


class ClassificationModel(BaseModel):
    """Model for DTI prediction as a classification problem"""

    def __init__(self, **kwargs):
        super().__init__()
        print(kwargs)
        self.save_hyperparameters()
        self._determine_feat_method(kwargs["feat_method"], kwargs["drug_hidden_dim"], kwargs["prot_hidden_dim"])
        drug_param = remove_arg_prefix("drug_", kwargs)
        prot_param = remove_arg_prefix("prot_", kwargs)
        # TODO fix hardcoded values
        self.prot_feat_embed = Embedding(20, kwargs["prot_node_embed_dim"])
        self.drug_feat_embed = Embedding(30, kwargs["drug_node_embed_dim"])
        self.prot_node_embed = node_embedders[prot_param["node_embed"]](
            prot_param["node_embed_dim"], prot_param["hidden_dim"], **prot_param
        )
        self.drug_node_embed = node_embedders[drug_param["node_embed"]](
            drug_param["node_embed_dim"], drug_param["hidden_dim"], **drug_param
        )
        self.prot_pool = poolers[prot_param["pool"]](prot_param["hidden_dim"], prot_param["hidden_dim"], **prot_param)
        self.drug_pool = poolers[drug_param["pool"]](drug_param["hidden_dim"], drug_param["hidden_dim"], **drug_param)
        mlp_param = remove_arg_prefix("mlp_", kwargs)
        self.mlp = MLP(**mlp_param, input_dim=self.embed_dim, out_dim=1)

    def forward(
        self,
        prot_x: Tensor,
        drug_x: Tensor,
        prot_edge_index: Adj,
        drug_edge_index: Adj,
        prot_batch: Tensor,
        drug_batch: Tensor,
        *args,
    ) -> Tensor:
        """Forward pass of the model

        Args:
            prot_x (Tensor): Protein node features
            drug_x (Tensor): Drug node features
            prot_edge_index (Adj): Protein edge info
            drug_edge_index (Adj): Drug edge info
            prot_batch (Tensor): Protein batch
            drug_batch (Tensor): Drug batch

        Returns:
            (Tensor): Final prediction
        """
        prot_x = self.prot_feat_embed(prot_x)
        drug_x = self.drug_feat_embed(drug_x)
        prot_x = self.prot_node_embed(prot_x, prot_edge_index, prot_batch)
        drug_x = self.drug_node_embed(drug_x, drug_edge_index, drug_batch)
        prot_embed = self.prot_pool(prot_x, prot_edge_index, prot_batch)
        drug_embed = self.drug_pool(drug_x, drug_edge_index, drug_batch)
        joint_embedding = self.merge_features(drug_embed, prot_embed)
        logit = self.mlp(joint_embedding)
        return torch.sigmoid(logit)

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test

        Args:
            data (TwoGraphData): Input data

        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        output = self.forward(
            data.prot_x,
            data.drug_x,
            data.prot_edge_index,
            data.drug_edge_index,
            data.prot_x_batch,
            data.drug_x_batch,
        )
        labels = data.label.unsqueeze(1)
        if self.hparams.weighted:
            weight = 1 / torch.sqrt(data.prot_count * data.drug_count)
            loss = F.binary_cross_entropy(output, labels.float(), weight=weight.unsqueeze(1))
        else:
            loss = F.binary_cross_entropy(output, labels.float())
        t = (output > 0.5).float()
        acc = accuracy(t, labels)
        try:
            _auroc = auroc(t, labels)
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
        """Generate arguments for this module

        Args:
            parser (ArgumentParser): Parent parser

        Returns:
            ArgumentParser: Updated parser
        """
        # Hack to find which embedding are used and add their arguments
        tmp_parser = ArgumentParser(add_help=False)
        tmp_parser.add_argument("--drug_node_embed", type=str, default="chebconv")
        tmp_parser.add_argument("--prot_node_embed", type=str, default="chebconv")
        tmp_parser.add_argument("--prot_pool", type=str, default="gmt")
        tmp_parser.add_argument("--drug_pool", type=str, default="gmt")

        args = tmp_parser.parse_known_args()[0]
        prot_node_embed = node_embedders[args.prot_node_embed]
        drug_node_embed = node_embedders[args.drug_node_embed]
        prot_pool = poolers[args.prot_pool]
        drug_pool = poolers[args.drug_pool]
        prot = parser.add_argument_group("Prot", prefix="--prot_")
        drug = parser.add_argument_group("Drug", prefix="--drug_")
        prot.add_argument("node_embed", default="chebconv")
        prot.add_argument("node_embed_dim", default=16, type=int, help="Size of aminoacid embedding")
        drug.add_argument("node_embed", default="chebconv")
        drug.add_argument(
            "node_embed_dim",
            default=16,
            type=int,
            help="Size of atom element embedding",
        )

        prot_node_embed.add_arguments(prot)
        drug_node_embed.add_arguments(drug)
        prot_pool.add_arguments(prot)
        drug_pool.add_arguments(drug)
        return parser
