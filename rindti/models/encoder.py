from typing import Tuple, Union

from torch.functional import Tensor
from torch_geometric.data import Data

from .base_model import BaseModel


class Encoder(BaseModel):
    """Encoder for graphs"""

    def __init__(self, feat_embed: bool = True, return_nodes: bool = False, **kwargs):
        super().__init__()
        self.feat_embed = self._get_feat_embed(kwargs) if feat_embed else None
        self.node_embed = self._get_node_embed(kwargs)
        self.pool = self._get_pooler(kwargs)
        self.return_nodes = return_nodes

    def forward(
        self,
        data: Data,
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Encode a graph

        Args:
            data (Data): torch_geometric - 'x', 'edge_index' etc

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: Either graph of graph+node embeddings
        """
        x, edge_index, batch = data["x"], data["edge_index"], data["batch"]
        if self.feat_embed is not None:
            x = self.feat_embed(x)
        x = self.node_embed(x, edge_index)
        embed = self.pool(x, edge_index, batch)
        if self.return_nodes:
            return embed, x
        return embed
