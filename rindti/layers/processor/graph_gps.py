import torch
from graphgps.layer.gps_layer import GPSLayer
from torch_geometric.data import Batch

from ..base_layer import BaseLayer


class GraphGPSNet(BaseLayer):
    """Uses GraphGPS transformer layers to encode the graph and predict noise and node type."""

    def __init__(
        self,
        local_module: str = "GAT",
        global_module: str = "Performer",
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        self.save_hyperparameters()
        self.node_encode = torch.nn.Sequential(
            *[
                GPSLayer(
                    hidden_dim,
                    local_module,
                    global_module,
                    num_heads,
                    dropout=dropout,
                    attn_dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, batch: Batch) -> Batch:
        """Return updated batch with noise and node type predictions."""
        feat_encode = self.feat_encode(batch.x)
        pos_encode = self.pos_encode(batch.pos)
        batch.x = torch.cat([feat_encode, pos_encode], dim=1)
        batch = self.node_encode(batch)
        batch.noise_pred = self.noise_pred(batch.x)
        batch.type_pred = self.type_pred(batch.x[batch.mask])
        return batch
