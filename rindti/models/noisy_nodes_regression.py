import torch.nn.functional as F

from ..data import TwoGraphData
from ..utils import get_node_loss, remove_arg_prefix
from .noisy_nodes_classification import NoisyNodesClassModel


class NoisyNodesRegModel(NoisyNodesClassModel):
    """Regression model with noisy nodes"""

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test

        Args:
            data (TwoGraphData): Input data

        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        cor_data = self.corruptor(data)
        prot = remove_arg_prefix("prot_", cor_data)
        drug = remove_arg_prefix("drug_", cor_data)
        output, prot_pred, drug_pred = self.forward(prot, drug)
        labels = data.label.unsqueeze(1).float()
        loss = F.mse_loss(output, labels)
        prot_idx = cor_data.prot_idx
        drug_idx = cor_data.drug_idx
        prot_loss = get_node_loss(data["prot_x"][prot_idx], prot_pred[prot_idx])
        drug_loss = get_node_loss(data["drug_x"][drug_idx], drug_pred[drug_idx])
        metrics = self._get_reg_metrics(output, labels)
        metrics.update(
            dict(
                loss=loss + self.hparams.prot_alpha * prot_loss + self.hparams.drug_alpha * drug_loss,
                prot_loss=prot_loss.detach(),
                drug_loss=drug_loss.detach(),
                pred_loss=loss.detach(),
            )
        )
        return metrics
