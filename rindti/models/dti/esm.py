import torch.nn.functional as F

from ...data import TwoGraphData
from ...layers import MLP
from ...utils import remove_arg_prefix
from ..encoder import Encoder
from .classification import ClassificationModel


class ESMModel(ClassificationModel):
    """
    ESM Model Class for DTI prediction
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self._determine_feat_method(**kwargs)
        drug_param = remove_arg_prefix("drug_", kwargs)
        prot_param = remove_arg_prefix("prot_", kwargs)
        mlp_param = remove_arg_prefix("mlp_", kwargs)
        self.prot_encoder = MLP(
            1280, prot_param["hidden_dim"], prot_param["hidden_dim"], prot_param["num_layers"], prot_param["dropout"]
        )
        self.drug_encoder = Encoder(**drug_param)
        self.mlp = self._get_mlp(mlp_param)

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test
        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        prot = remove_arg_prefix("prot_", data)
        drug = remove_arg_prefix("drug_", data)
        fwd_dict = self.forward(prot, drug)
        labels = data.label.unsqueeze(1)
        bce_loss = F.binary_cross_entropy(fwd_dict["pred"], labels.float())
        metrics = self._get_class_metrics(fwd_dict["pred"], labels)
        snnl = 1 / self.snnl(fwd_dict["joint_embed"], data.label)["graph_loss"]
        metrics.update(
            dict(
                loss=bce_loss + self.hparams.alpha * snnl,
                snn_loss=snnl.detach(),
                bce_loss=bce_loss.detach(),
            )
        )
        return metrics
