from copy import deepcopy
from typing import Tuple, Union

from torch.functional import Tensor

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
        data: dict,
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Encode a graph

        Args:
            data (dict): Dict of all data - 'x', 'edge_index' etc

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: Either graph of graph+node embeddings
        """
        if self.feat_embed is not None:
            x = self.feat_embed(data["x"])
        data["x"] = self.node_embed(x, **{k: v for k, v in data.items() if k != "x"})
        embed = self.pool(x, **{k: v for k, v in data.items() if k != "x"})
        if self.return_nodes:
            return embed, data["x"]
        return embed
