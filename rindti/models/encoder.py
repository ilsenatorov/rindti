from rindti.layers.other.mlp import MLP
from typing import Tuple, Union

from torch.functional import Tensor
from .base_model import BaseModel


class Encoder(BaseModel):
    def __init__(self, feat_embed=True, **kwargs):
        super().__init__()
        if feat_embed:
            self.feat_embed = self._get_feat_embed(kwargs)
        else:
            self.feat_embed = None
        self.node_embed = self._get_node_embed(kwargs)
        self.pool = self._get_pooler(kwargs)

    def forward(
        self,
        data: dict,
        return_nodes: bool = False,
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if self.feat_embed is not None:
            data["x"] = self.feat_embed(data["x"])
        data["x"] = self.node_embed(**data)
        embed = self.pool(**data)
        if return_nodes:
            return embed, data["x"]
        return embed
