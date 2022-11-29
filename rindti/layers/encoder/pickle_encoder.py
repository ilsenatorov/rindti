import pickle
from typing import Tuple, Union, Optional

from pytorch_lightning import LightningModule
import torch
from torch import Tensor
from torch_geometric.data import Data

from rindti.layers.other import MLP


class PickleEncoder(LightningModule):
    def __init__(self, **kwargs):
        super(PickleEncoder, self).__init__()
        self.data = pickle.load(open(kwargs["file"], "rb"))
        self.mlp = MLP(
            len(list(self.data.items())[0][1]),
            kwargs["hidden_dim"],
        )

    def forward(self, data: Union[dict, Data]) -> Union[Tensor, Tuple[Tensor, Optional[Tensor]]]:
        embed = torch.stack([torch.tensor(self.data[idx]) for idx in data["id"]]).cuda()
        return self.mlp(embed)[0], None
