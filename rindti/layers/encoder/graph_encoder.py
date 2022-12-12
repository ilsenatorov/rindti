from typing import Tuple, Union

from pytorch_lightning import LightningModule
from torch import nn
from torch.functional import Tensor
from torch_geometric.data import Data

from ..graphconv import ChebConvNet, FilmConvNet, GatConvNet, GINConvNet, TransformerNet
from ..graphconv.gcnet import GCNet
from ..graphconv.sagenet import SAGENet
from ..graphpool import DiffPoolNet, GMTNet, MeanPool

node_embedders = {
    "gcn": GCNet,
    "ginconv": GINConvNet,
    "sage": SAGENet,
    "chebconv": ChebConvNet,
    "gatconv": GatConvNet,
    "filmconv": FilmConvNet,
    "transformer": TransformerNet,
}
poolers = {"gmt": GMTNet, "diffpool": DiffPoolNet, "mean": MeanPool}


class GraphEncoder(LightningModule):
    r"""Encoder for graphs.

    Args:
        return_nodes (bool, optional): Return node embeddings as well. Defaults to False.
    """

    def __init__(self, return_nodes: bool = False, **kwargs):
        super().__init__()
        self.update_params(kwargs)
        self.feat_embed = self._get_feat_embed(kwargs)
        self.node_embed = self._get_node_embed(kwargs["node"])
        self.pool = self._get_pooler(kwargs["pool"])
        self.return_nodes = return_nodes

    def update_params(self, kwargs: dict):
        """Update the params to connect parts of the encoder together (hidden dims)."""
        data_params = kwargs["data"]
        kwargs["pool"]["max_nodes"] = data_params["max_nodes"]
        kwargs.update(data_params)
        kwargs["node"]["input_dim"] = kwargs["hidden_dim"]
        kwargs["node"]["output_dim"] = kwargs["hidden_dim"]
        kwargs["pool"]["input_dim"] = kwargs["hidden_dim"]
        kwargs["pool"]["output_dim"] = kwargs["hidden_dim"]

    def _get_node_embed(self, params: dict) -> nn.Module:
        """Get node embedding module based on its name"""
        return node_embedders[params["module"]](**params)

    def _get_pooler(self, params: dict) -> nn.Module:
        """Get pooling module based on its name"""
        return poolers[params["module"]](**params)

    def _get_label_embed(self, params: dict) -> nn.Embedding:
        """Get label embedding module based on its name"""
        return nn.Embedding(params["feat_dim"] + 1, params["hidden_dim"], padding_idx=0)

    def _get_onehot_embed(self, params: dict) -> nn.Linear:
        """Get onehot embedding module based on its name"""
        return nn.Linear(params["feat_dim"], params["hidden_dim"], bias=False)

    def _get_feat_embed(self, params: dict) -> Union[nn.Embedding, nn.Linear]:
        """Get feature embedding module based on its name"""
        if params["feat_type"] == "onehot":
            return self._get_onehot_embed(params)
        elif params["feat_type"] == "label":
            return self._get_label_embed(params)
        else:
            raise ValueError("Unknown feature type!")

    def forward(
        self,
        data: Union[dict, Data],
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        r"""Encode a graph.

        Args:
            data (Union[dict, Data]): Graph to encode. Must contain the following keys:
                - x: Node features
                - edge_index: Edge indices
                - batch: Batch indices
        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: Either graph of graph+node embeddings
        """
        if not isinstance(data, dict):
            data = data.to_dict()
        x, edge_index, batch, edge_feats = (
            data["x"],
            data["edge_index"],
            data["batch"],
            data.get("edge_feats"),
        )
        feat_embed = self.feat_embed(x)
        node_embed = self.node_embed(
            x=feat_embed,
            edge_index=edge_index,
            edge_feats=edge_feats,
            batch=batch,
        )
        graph_embed = self.pool(x=node_embed, edge_index=edge_index, batch=batch)
        return graph_embed, node_embed

    def embed(self, data: Data, **kwargs):
        """Generate an embedding for a graph."""
        self.return_nodes = False
        embed = self.forward(data)
        return embed.detach()
