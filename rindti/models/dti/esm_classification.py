import torch
import torch.nn.functional as F

from ...data import TwoGraphData
from ...layers import MLP
from ...utils import remove_arg_prefix
from ..base_model import BaseModel
from ..encoder import Encoder


class ESMClassModel(BaseModel):
    """ESM Model Class for DTI prediction."""

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

    def forward(self, prot: dict, drug: dict):
        """Forward pass of the model."""
        prot_embed = self.prot_encoder(prot["x"].view(-1, 1280))
        drug_embed = self.drug_encoder(drug)
        joint_embedding = self.merge_features(drug_embed, prot_embed)
        return dict(
            pred=torch.sigmoid(self.mlp(joint_embedding)),
            prot_embed=prot_embed,
            drug_embed=drug_embed,
            joint_embed=joint_embedding,
        )

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test.

        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        prot = remove_arg_prefix("prot_", data)
        drug = remove_arg_prefix("drug_", data)
        fwd_dict = self.forward(prot, drug)
        labels = data.label.unsqueeze(1)
        bce_loss = F.binary_cross_entropy(fwd_dict["pred"], labels.float())
        metrics = self._get_class_metrics(fwd_dict["pred"], labels)
        metrics.update(
            dict(
                loss=bce_loss,
            )
        )
        return metrics
