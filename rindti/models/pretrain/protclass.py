from torch import Tensor

from ...data import TwoGraphData
from ...losses import CrossEntropyLoss, NodeLoss
from ..base_model import BaseModel
from ..encoder import Encoder


class ProtClassModel(BaseModel):
    """Model for basic protein classification. Data in self.forward has to contain data.y, which will be the label we aim to predict."""

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        if kwargs.get("pretrain"):
            self.encoder = self.load_from_checkpoint(kwargs["pretrain"]).encoder
            self.hparams.lr *= 0.001
        else:
            self.encoder = Encoder(**kwargs)
        self.loss = CrossEntropyLoss(return_nodes=True, **kwargs)
        self.node_loss = NodeLoss(weighted=False)

    def forward(self, data: dict) -> Tensor:
        """"""
        graphs, nodes = self.encoder(data)
        return graphs, nodes

    def shared_step(self, data: TwoGraphData) -> dict:
        """"""
        graphs, nodes = self.forward(data)
        metrics = self.loss(graphs, data.y)
        return {k: v.detach() if k != "loss" else v for k, v in metrics.items()}
