import warnings
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from glycowork.glycan_data.loader import lib
from glycowork.motif.graph import glycan_to_graph
from pytorch_lightning import LightningModule
from torch.functional import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap

if torch.cuda.is_available():
    from glycowork.ml.models import SweetNet, init_weights, trained_SweetNet
else:
    trained_SweetNet, SweetNet, init_weights = torch.nn.Module, torch.nn.Module, torch.nn.Module
    warnings.warn("GPU not available")


class SweetNetEncoder(LightningModule):
    """Uses SweetNet to encode a glycan."""

    def __init__(self, trainable=False, **kwargs):
        super().__init__()
        self.sweetnet = SweetNetAdapter(trainable, **kwargs).cuda()

    def forward(self, data: Union[dict, Data], **kwargs) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if not isinstance(data, dict):
            data = data.to_dict()
        return self.sweetnet(data["IUPAC"])


class SweetNetAdapter(SweetNet):
    """Wrapper for SweetNet that can be used with Lightning."""

    def __init__(self, trainable=False, **kwargs):
        super().__init__(len(lib), 970)
        self.trainable = trainable
        self.apply(lambda module: init_weights(module, mode="sparse"))
        self.load_state_dict(trained_SweetNet)
        self.lin4 = torch.nn.Linear(256, kwargs["hidden_dim"])
        # self.single_embeds = torch.nn.Linear(16, kwargs["hidden_dim"])
        self.hidden_dim = kwargs["hidden_dim"]
        self.cache = {}

    def forward(self, x, edge_index=None, batch=None, inference=False):
        """Forward a glycan through the Sweetnet"""
        x_tmp, edge_index_tmp = list(zip(*[glycan_to_graph(iupac) for iupac in x]))
        embeddings = []
        for y, edges, iupac in zip(x_tmp, edge_index_tmp, x):
            if iupac in self.cache:
                embeddings.append(torch.tensor(self.cache[iupac]))
                continue
            y = self.item_embedding(torch.tensor(y).cuda())
            # y = self.item_embedding(torch.tensor(y))
            y = y.squeeze(1)
            if len(edges) == 0:
                # embeddings.append(self.single_embeds(y.detach().squeeze()))
                # embeddings.append(y.detach().squeeze().cuda())
                # embeddings.append(y.detach().squeeze())
                embeddings.append(torch.zeros(self.hidden_dim).cuda())
                self.cache[iupac] = torch.tensor(embeddings[-1])
                continue
            edges = torch.tensor(edges).cuda().to(torch.long)
            # edges = torch.tensor(edges).to(torch.long)
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
                self.cache[iupac] = torch.tensor(embeddings[-1])
            else:
                embeddings.append(self.lin4(y.detach()).squeeze())
                self.cache[iupac] = torch.tensor(embeddings[-1])
        return torch.stack(embeddings), None
