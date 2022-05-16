from typing import Any

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

    def n_node_feats(self, prefix: str) -> int:
        """Calculate the feature dimension of one of the graphs.

        If the features are index-encoded (dtype long, single number for each node, for use with Embedding),
        then return the max. Otherwise return size(1)
        """
        x = self[prefix + "x"]
        if len(x.size()) == 1:
            return x.max().item() + 1
        if len(x.size()) == 2:
            return x.size(1)
        raise ValueError("Too many dimensions in input features")

    def n_edge_feats(self, prefix: str) -> int:
        """Return number of different edges for graph with prefix."""
        if prefix + "edge_feats" not in self._store:
            return 1
        if self[prefix + "edge_feats"] is None:
            return 1
        edge_feats = self._store[prefix + "edge_feats"]
        if len(edge_feats.size()) == 1:
            return edge_feats.max().item() + 1
        if len(edge_feats.size()) == 2:
            return edge_feats.size(1)
        raise ValueError("Too many dimensions in input features")
