from torch import Tensor, nn

from ...data import TwoGraphData
from ...layers.encoder import GraphEncoder
from ...layers.other import MLP
from ..base_model import BaseModel


class ProtClassModel(BaseModel):
    """Model for basic protein classification. Data in self.forward has to contain data.y, which will be the label we aim to predict."""

    def __init__(self, **kwargs):
        kwargs = super().__init__(**kwargs)
        self.encoder = GraphEncoder(**kwargs["encoder"])
        self.noise_predict = MLP(kwargs["hidden_dim"], out_dim=3, **kwargs["mlp"])
        self.mlp = MLP(input_dim=kwargs["hidden_dim"], out_dim=kwargs["num_classes"], **kwargs["mlp"])
        self.loss = nn.CrossEntropyLoss()

    def forward(self, data: dict) -> Tensor:
        encoded = self.encoder(data)
        noise_preds = self.noise_predict(encoded["node"])
        preds = self.mlp(encoded["graph"])
        return preds, encoded["noise"], noise_preds

    def shared_step(self, data: TwoGraphData) -> dict:
        preds, noise, noise_preds = self.forward(data)
        cross_entropy_loss = self.loss(preds, data.y)
        denoising_loss = (noise - noise_preds).pow(2).mean()
        loss = cross_entropy_loss + 1 * denoising_loss
        return {
            "loss": loss,
            "cross_entropy_loss": cross_entropy_loss,
            "denoising_loss": denoising_loss,
            "preds": preds.detach(),
            "labels": data.y,
        }
