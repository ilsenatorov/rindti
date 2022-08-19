import torch.nn.functional as F
from torch.functional import Tensor

from ...data import TwoGraphData
from ...layers.encoder import GraphEncoder, PretrainedEncoder, SweetNetEncoder
from ...layers.other import MLP
from ...utils import remove_arg_prefix
from ..base_model import BaseModel

encoders = {"graph": GraphEncoder, "sweetnet": SweetNetEncoder, "pretrained": PretrainedEncoder}


class ClassificationModel(BaseModel):
    """Model for DTI prediction as a classification problem."""

    def __init__(
        self,
        drug_encoder: str,
        drug_encoder_config: dict,
        prot_encoder: str,
        prot_encoder_config: dict,
        mlp_config: dict,
        merge_features: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.drug_encoder = encoders[drug_encoder](**drug_encoder_config)
        self.prot_encoder = encoders[prot_encoder](**prot_encoder_config)
        self._determine_feat_method(
            merge_features,
            drug_encoder_config["hidden_dim"],
            prot_encoder_config["hidden_dim"],
        )
        self.mlp = MLP(self.embed_dim, 1, **mlp_config)

    def forward(self, prot: dict, drug: dict) -> Tensor:
        """"""
        prot_embed = self.prot_encoder(prot)
        drug_embed = self.drug_encoder(drug)
        joint_embedding = self.merge_features(drug_embed, prot_embed)
        return self.mlp(joint_embedding)

    def shared_step(self, data: TwoGraphData) -> dict:
        """Step that is the same for train, validation and test.

        Returns:
            dict: dict with different metrics - losses, accuracies etc. Has to contain 'loss'.
        """
        prot = remove_arg_prefix("prot_", data)
        drug = remove_arg_prefix("drug_", data)
        pred = self.forward(prot, drug)
        labels = data.label.unsqueeze(1)
        bce_loss = F.binary_cross_entropy_with_logits(pred, labels.float())
        return dict(loss=bce_loss)
