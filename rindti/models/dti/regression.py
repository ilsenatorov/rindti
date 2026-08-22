import torch
import torch.nn.functional as F

from ...data import TwoGraphData
from ...utils import remove_arg_prefix
from .classification import ClassificationModel


class RegressionModel(ClassificationModel):
    """Model for DTI prediction as a reg problem. Based on the same MLP architecture of the classification class but using mean square error (mse) loss function."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_reg_metrics()

    def shared_step(self, data: TwoGraphData) -> dict:
        """The prediction labels are normalized between 0 and 1 using the sigmoid function as this prevents domination of large values while computing the mean square error loss function."""
        prot = remove_arg_prefix("prot_", data)
        drug = remove_arg_prefix("drug_", data)
        fwd_dict = self.forward(prot, drug)
        labels = data.label.unsqueeze(1)
        mse_loss = F.mse_loss(torch.sigmoid(fwd_dict["pred"]), labels.float())
        return dict(loss=mse_loss, preds=fwd_dict["pred"].detach(), labels=labels.detach())
