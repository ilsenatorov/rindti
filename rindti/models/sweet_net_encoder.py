import warnings
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from glycowork.glycan_data.loader import lib
from glycowork.ml.models import SweetNet, init_weights
from glycowork.motif.graph import glycan_to_graph
from torch.functional import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap

if torch.cuda.is_available():
    from glycowork.ml.models import SweetNet, init_weights, trained_SweetNet
else:
    trained_SweetNet, SweetNet, init_weights = None, None, None
    warnings.warn("GPU not available")

from rindti.models.base_model import BaseModel


class SweetNetEncoder(BaseModel):
    def __init__(self, trainable=False, **kwargs):
        super().__init__()
        self.sweetnet = SweetNetAdapter(trainable, **kwargs).cuda()

    def forward(self, data: Union[dict, Data], **kwargs) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if not isinstance(data, dict):
            data = data.to_dict()

        return self.sweetnet(data["IUPAC"])

    def embed(self, data: Data, **kwargs):
        embed = self.forward(data)
        return embed.detach()


class SweetNetAdapter(SweetNet):
    def __init__(self, trainable=False, **kwargs):
        super().__init__(len(lib), 970)
        self.trainable = trainable
        self.apply(lambda module: init_weights(module, mode="sparse"))
        self.load_state_dict(trained_SweetNet)
        self.lin4 = torch.nn.Linear(256, kwargs["hidden_dim"])

    def forward(self, x, edge_index=None, batch=None, inference=False):
        x, edge_index = list(zip(*[glycan_to_graph(iupac) for iupac in x]))
        embeddings = []
        for y, edges in zip(x, edge_index):
            y = self.item_embedding(torch.tensor(y).cuda())
            y = y.squeeze(1)
            edges = torch.tensor(edges).cuda()

            y = F.leaky_relu(self.conv1(y, edges))

            y, edges, _, batch, _, _ = self.pool1(y, edges, None)
            y1 = torch.cat([gmp(y, batch), gap(y, batch)], dim=1)

            y = F.leaky_relu(self.conv2(y, edges))

            y, edges, _, batch, _, _ = self.pool2(y, edges, None, batch)
            y2 = torch.cat([gmp(y, batch), gap(y, batch)], dim=1)

            y = F.leaky_relu(self.conv3(y, edges))

            y, edges, _, batch, _, _ = self.pool3(y, edges, None, batch)
            y3 = torch.cat([gmp(y, batch), gap(y, batch)], dim=1)

            y = y1 + y2 + y3

            if self.trainable:
                embeddings.append(self.lin4(y).squeeze())
            embeddings.append(self.lin4(y.detach()).squeeze())
        return torch.stack(embeddings)
