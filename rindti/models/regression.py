import torch
import torch.nn.functional as F
from torch.functional import Tensor
from torchmetrics.functional import explained_variance, mean_absolute_error, pearson_corrcoef

from ..utils import remove_arg_prefix
from ..utils.data import TwoGraphData
from .classification import ClassificationModel


class RegressionModel(ClassificationModel):
    """Model for DTI prediction as a reg problem"""

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test
        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        prot = remove_arg_prefix("prot_", data)
        drug = remove_arg_prefix("drug_", data)
        output = self.forward(prot, drug)
        labels = data.label.unsqueeze(1).float()
        loss = F.mse_loss(output, labels)
        metrics = self._get_reg_metrics(output, labels)
        metrics.update(dict(loss=loss))
        return metrics
