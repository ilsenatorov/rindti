from typing import Tuple, Union

from pytorch_lightning import LightningModule
from torch import nn
from torch.functional import Tensor
from torch_geometric.data import Data

from ..other import MLP


class PretrainedEncoder(LightningModule):
    r"""Encoder for pretrained models (ESM etc). Assumes data only has x."""

    def __init__(self, **kwargs):
        super().__init__()
        self.update_params(kwargs)
        self.mlp = MLP(
            kwargs["data"]["feat_dim"],
            kwargs["hidden_dim"],
            kwargs["hidden_dim"],
        )

    def forward(
        self,
        data: Union[dict, Data],
        **kwargs,
    ) -> Tensor:
        r"""Encode an entry.

        Args:
            data (Union[dict, Data]): Entry to encode. Must contain the following keys:
                - x: Node features
        Returns:
            Tensor: Encoded entry.
        """
        return self.mlp(data["x"])
