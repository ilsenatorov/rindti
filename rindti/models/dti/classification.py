import torch
import torch.nn.functional as F
from torch.functional import Tensor

from ..SweetNetEncoder import SweetNetEncoder
from ...data import TwoGraphData
from ...utils import remove_arg_prefix
from ..base_model import BaseModel
from ..encoder import Encoder


class ClassificationModel(BaseModel):
    """Model for DTI prediction as a class problem."""

    def __init__(self, **kwargs):
        super().__init__()
        self._determine_feat_method(kwargs["model"]["feat_method"], kwargs["model"]["prot"]["hidden_dim"], kwargs["model"]["drug"]["hidden_dim"])
        self.prot_encoder = Encoder(**kwargs["model"]["prot"])
        if kwargs["model"]["drug"]["node"]["module"] == "SweetNet":
            self.drug_encoder = SweetNetEncoder(**kwargs["model"]["drug"])
        else:
            self.drug_encoder = Encoder(**kwargs["model"]["drug"])
        self.mlp = self._get_mlp(**kwargs["model"]["mlp"])
        self._set_class_metrics()

    def forward(self, prot: dict, drug: dict) -> Tensor:
        """"""
        prot_embed = self.prot_encoder(prot)
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
        return dict(loss=bce_loss, preds=fwd_dict["pred"].detach(), labels=labels.detach())
