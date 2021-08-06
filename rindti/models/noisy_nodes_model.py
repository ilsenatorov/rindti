from argparse import ArgumentParser
from copy import deepcopy
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from torch.functional import Tensor
from torch_geometric.typing import Adj
from torchmetrics.functional import accuracy, auroc, matthews_corrcoef

from rindti.utils.data import TwoGraphData

from ..layers import ChebConvNet, DiffPoolNet, GatConvNet, GINConvNet, GMTNet, MeanPool, NoneNet
from .classification_model import ClassificationModel

node_embedders = {
    "ginconv": GINConvNet,
    "chebconv": ChebConvNet,
    "gatconv": GatConvNet,
    "none": NoneNet,
}
poolers = {"gmt": GMTNet, "diffpool": DiffPoolNet, "mean": MeanPool}


class NoisyNodesModel(ClassificationModel):
    """Model for DTI prediction as a classification problem"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prot_pred = node_embedders[kwargs["prot_node_embed"]](
            kwargs["prot_hidden_dim"], 20, num_layers=3, hidden_dim=32
        )
        self.drug_pred = node_embedders[kwargs["drug_node_embed"]](
            kwargs["drug_hidden_dim"], 30, num_layers=3, hidden_dim=32
        )

    def corrupt_features(self, features: torch.Tensor, frac: float) -> torch.Tensor:
        """Corrupt the features

        Args:
            features (torch.Tensor): Node features
            frac (float): Fraction of nodes to corrupt

        Returns:
            torch.Tensor: New corrupt features
        """
        num_feat = features.size(0)
        num_node_types = int(features.max())
        num_corrupt_nodes = ceil(num_feat * frac)
        corrupt_idx = np.random.choice(range(num_feat), num_corrupt_nodes, replace=False)
        corrupt_features = torch.tensor(
            np.random.choice(range(num_node_types), num_corrupt_nodes, replace=True),
            dtype=torch.long,
            device=self.device,
        )
        features[corrupt_idx] = corrupt_features
        return features, corrupt_idx

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
            prot_feat, prot_idx = self.corrupt_features(data["prot_x"], prot_frac)
            data["prot_x"] = prot_feat
            data["prot_cor_idx"] = prot_idx
        if drug_frac > 0:
            drug_feat, drug_idx = self.corrupt_features(data["drug_x"], drug_frac)
            data["drug_x"] = drug_feat
            data["drug_cor_idx"] = drug_idx
        return data

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
        prot_pred = self.prot_pred(prot_x, prot_edge_index, prot_batch)
        drug_pred = self.drug_pred(drug_x, drug_edge_index, drug_batch)
        return torch.sigmoid(logit), prot_pred, drug_pred

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test

        Args:
            data (TwoGraphData): Input data

        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        cor_data = self.corrupt_data(data, self.hparams.prot_frac, self.hparams.drug_frac)
        output, prot_pred, drug_pred = self.forward(
            cor_data["prot_x"],
            cor_data["drug_x"],
            cor_data["prot_edge_index"],
            cor_data["drug_edge_index"],
            cor_data["prot_x_batch"],
            cor_data["drug_x_batch"],
        )
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
        t = (output > 0.5).float()
        acc = accuracy(t, labels)
        try:
            _auroc = auroc(t, labels)
        except Exception:
            _auroc = torch.tensor(np.nan, device=self.device)
        _mc = matthews_corrcoef(output, labels.squeeze(1), num_classes=2)
        return {
            "loss": loss + self.hparams.prot_alpha * prot_loss + self.hparams.drug_alpha * drug_loss,
            "prot_loss": prot_loss,
            "drug_loss": drug_loss,
            "pred_loss": loss,
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
        drug.add_argument("alpha", default=0.1, type=float, help="Drug node loss factor")
        drug.add_argument("frac", default=0.05, type=float, help="Proportion of drug nodes to corrupt")
        drug.add_argument("node_embed", default="chebconv")
        drug.add_argument(
            "node_embed_dim",
            default=16,
            type=int,
            help="Size of atom element embedding",
        )
        prot.add_argument("alpha", default=0.1, type=float, help="Prot node loss factor")
        prot.add_argument("frac", default=0.05, type=float, help="Proportion of prot nodes to corrupt")
        prot.add_argument("node_embed", default="chebconv")
        prot.add_argument("node_embed_dim", default=16, type=int, help="Size of aminoacid embedding")

        prot_node_embed.add_arguments(prot)
        drug_node_embed.add_arguments(drug)
        prot_pool.add_arguments(prot)
        drug_pool.add_arguments(drug)
        return parser
