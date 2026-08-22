from typing import Any

from torch_geometric.data import Data


class TwoGraphData(Data):
    """Subclass of torch_geometric.data.Data for protein and drug data. The helper function __inc__ is responsible for correctly incrementing drug and target protein graph edge indices. The helper methods n_nodes(prefix) and n_edges(prefix) provide a convenient interface for querying graph sizes."""

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
