import torch.nn.functional as F
from torch import Tensor, nn

from ...data import TwoGraphData
from ...layers.encoder import GraphEncoder, VectorEncoder
from ...layers.other import MLP
from ...utils import remove_arg_prefix
from ..base_model import BaseModel

encoders = {
    "graph": GraphEncoder,
    # For entities with no graph structure - notably `prots.features.method: esm`,
    # which is a single mean-pooled vector per protein.
    "vector": VectorEncoder,
    # "sweetnet": SweetNetEncoder,
    # "pretrained": PretrainedEncoder,
}


class ClassificationModel(BaseModel):
    """Model for DTI prediction as a classification problem."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._determine_feat_method(
            kwargs["model"]["feat_method"],
            drug_hidden_dim=kwargs["model"]["drug"]["hidden_dim"],
            prot_hidden_dim=kwargs["model"]["prot"]["hidden_dim"],
        )
        self.prot_encoder = encoders[kwargs["model"]["prot"]["method"]](**kwargs["model"]["prot"])
        self.drug_encoder = encoders[kwargs["model"]["drug"]["method"]](**kwargs["model"]["drug"])
        # Both poolers L2-normalise, so each tower emits a unit vector. The merge
        # operators then land on very different scales - the joint embedding's norm is
        # ~1.41 for concat and the two element-wise differences, but ~0.088 for `mult`,
        # a 16x gap that the MLP sees as a much weaker signal. Without this the
        # `feat_method` ablation measures input scale as much as merge semantics.
        self.joint_norm = nn.LayerNorm(self.embed_dim)
        self.mlp = MLP(input_dim=self.embed_dim, out_dim=1, **kwargs["model"]["mlp"])
        self._setup_metrics()

    def _setup_metrics(self):
        """Which metric collection this model reports. Overridden by subclasses."""
        self._set_class_metrics()

    def forward(self, prot: dict, drug: dict) -> Tensor:
        """"""
        prot_embed = self.prot_encoder(prot)
        drug_embed = self.drug_encoder(drug)
        joint_embedding = self.joint_norm(self.merge_features(drug_embed, prot_embed))
        return dict(
            pred=self.mlp(joint_embedding),
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
        bce_loss = F.binary_cross_entropy_with_logits(fwd_dict["pred"], labels.float())
        return dict(loss=bce_loss, preds=fwd_dict["pred"].detach(), labels=labels.detach())
