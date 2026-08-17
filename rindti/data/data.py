from typing import Any

import torch
from torch_geometric.data import Data


class TwoGraphData(Data):
    """Subclass of torch_geometric.data.Data for protein and drug data."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> dict:
        """How to increment values during batching."""
        if not key.endswith("edge_index"):
            return super().__inc__(key, value, *args, **kwargs)
        lenedg = len("edge_index")
        prefix = key[:-lenedg]
        return self[prefix + "x"].size(0)

    def n_nodes(self, prefix: str) -> int:
        """Return number of nodes for graph with prefix."""
        return self[prefix + "x"].size(0)

    def n_edges(self, prefix: str) -> int:
        """Return number of edges for graph with prefix."""
        return self[prefix + "edge_index"].size(1)


# torch>=2.6 loads checkpoints with weights_only=True by default; the processed
# datasets contain TwoGraphData instances, so it has to be allowlisted.
torch.serialization.add_safe_globals([TwoGraphData])
