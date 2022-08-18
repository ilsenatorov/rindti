from torch import Tensor, nn

from ...data import TwoGraphData
from ...layers.encoder import GraphEncoder
from ...layers.other import MLP
from ..base_model import BaseModel


class ProtClassModel(BaseModel):
    """
    Model for basic protein classification. Data in self.forward has to contain data.y, which will be the label we aim
    to predict.
    """

    def __init__(self, **kwargs):
        kwargs = super().__init__(**kwargs)
        self.encoder = GraphEncoder(**kwargs["encoder"])
        self.mlp = MLP(input_dim=kwargs["hidden_dim"], out_dim=kwargs["num_classes"], **kwargs["mlp"])
        self.loss = nn.CrossEntropyLoss()
        self.train_metrics, self.val_metrics, self.test_metrics = self._set_class_metrics(kwargs["num_classes"])

    def forward(self, data: dict) -> Tensor:
        """Forward the data through the model"""
        graphs, _ = self.encoder(data)
        preds = self.mlp(graphs)
        return preds

    def shared_step(self, data: TwoGraphData) -> dict:
        """Perform the shared step for this model"""
        preds = self.forward(data)
        loss = self.loss(preds, data.y)
        return {"loss": loss, "preds": preds.detach(), "labels": data.y}
