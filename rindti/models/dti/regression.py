import torch.nn.functional as F

from ...data import TwoGraphData
from ...utils import remove_arg_prefix
from .classification import ClassificationModel


class RegressionModel(ClassificationModel):
    """Model for DTI prediction as a reg problem."""

    def _setup_metrics(self):
        self._set_reg_metrics()

    def shared_step(self, data: TwoGraphData) -> dict:
        """Regress the affinity directly.

        The loss used to be MSE against ``sigmoid(pred)`` while the raw logits were
        handed to the metrics, so the logged MAE/MSE were on a different scale than
        the optimized loss. The sigmoid was wrong regardless: ``parse_dataset``
        emits raw or log10 affinities, which are not bounded to [0, 1].
        """
        prot = remove_arg_prefix("prot_", data)
        drug = remove_arg_prefix("drug_", data)
        fwd_dict = self.forward(prot, drug)
        labels = data.label.unsqueeze(1).float()
        preds = fwd_dict["pred"]
        mse_loss = F.mse_loss(preds, labels)
        return dict(loss=mse_loss, preds=preds.detach(), labels=labels.detach())
