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
        # print("Data:", len(list(self.data.items())[0][1]))
        # print(list(self.data.items())[0][1])
        self.mlp = MLP(
            len(list(self.data.items())[0][1]),
            kwargs["hidden_dim"],
            num_layers=4,
            hidden_dim=512,
        )

    def forward(self, data: Union[dict, Data]) -> Union[Tensor, Tuple[Tensor, Optional[Tensor]]]:
        # print("Sample:", self.data[data["id"][0]].shape)
        embed = torch.stack([torch.tensor(self.data[idx]) for idx in data["id"]]).cuda()
        return self.mlp(embed)[0], None
