import torch
import torch.nn.functional as F
from torch.functional import Tensor
from torch_geometric.typing import Adj
from torchmetrics.functional import explained_variance, mean_squared_error, pearson_corrcoef

from ..layers import MLP, ChebConvNet, DiffPoolNet, GatConvNet, GINConvNet, GMTNet, MeanPool, NoneNet
from ..utils.data import TwoGraphData
from .classification_model import ClassificationModel

node_embedders = {
    "ginconv": GINConvNet,
    "chebconv": ChebConvNet,
    "gatconv": GatConvNet,
    "none": NoneNet,
}
poolers = {"gmt": GMTNet, "diffpool": DiffPoolNet, "mean": MeanPool}


class RegressionModel(ClassificationModel):
    """Model for DTI prediction as a regression problem"""

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
        return self.mlp(joint_embedding)

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
        loss = F.mse_loss(output, labels.float())
        corr = pearson_corrcoef(output, labels)
        mse = mean_squared_error(output, labels)
        expvar = explained_variance(output, labels)
        return {
            "loss": loss,
            "corr": corr,
            "mse": mse,
            "expvar": expvar,
        }
