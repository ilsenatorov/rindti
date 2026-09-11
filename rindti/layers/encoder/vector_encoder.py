from __future__ import annotations

from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from ..other import MLP


class VectorEncoder(nn.Module):
    r"""Encoder for entities that are a single feature vector rather than a graph.

    This is what the ``esm`` protein featurisation produces: ``prot_esm.py`` writes one
    mean-pooled ESM-1b representation of shape ``(1, 1280)`` per protein, with no
    ``edge_index`` at all. :class:`~rindti.layers.encoder.GraphEncoder` dereferences
    ``data["edge_index"]`` unconditionally, so it cannot consume that input - which is
    why the ESM arm has never actually been runnable. Pair ``features.method: esm`` on
    the workflow side with ``model.prot.method: vector`` here.

    The output is a ``(batch_size, hidden_dim)`` embedding, exactly what
    :class:`~rindti.layers.encoder.GraphEncoder` returns, so the merge operators in
    ``BaseModel._determine_feat_method`` need no special case.

    Args:
        hidden_dim (int, optional): Size of the output embedding. Defaults to 128.
        num_layers (int, optional): Number of layers in the MLP. Defaults to 2.
        dropout (float, optional): Dropout ratio. Defaults to 0.1.
        return_nodes (bool, optional): Return the per-row embeddings as well, for
            interface parity with :class:`GraphEncoder`. Defaults to False.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        return_nodes: bool = False,
        **kwargs,
    ):
        super().__init__()
        data = kwargs["data"]
        if data["feat_type"] != "onehot":
            # `onehot` is what workflow/scripts/utils.py:get_type calls any float
            # tensor, ESM embeddings included. A `label` tensor is integer vocabulary
            # indices, which a Linear cannot consume.
            raise ValueError(
                f"VectorEncoder needs continuous features, got feat_type "
                f"{data['feat_type']!r}. Use method: graph for label features."
            )
        self.mlp = MLP(
            input_dim=data["feat_dim"],
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.return_nodes = return_nodes

    def forward(self, data: dict | Data, **kwargs) -> Tensor | tuple[Tensor, Tensor]:
        r"""Encode a batch of feature vectors.

        Args:
            data (Union[dict, Data]): Must contain ``x``; ``batch`` is used if present.
        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: Either the graph or graph+node embeddings
        """
        if not isinstance(data, dict):
            data = data.to_dict()
        x, batch = data["x"], data.get("batch")
        row_embed = self.mlp(x)
        # ESM contributes exactly one row per protein, so this pool is the identity
        # there. It is here so that a multi-row vector representation (per-residue
        # embeddings, say) still yields one embedding per graph rather than silently
        # misaligning with the drug side.
        embed = row_embed if batch is None else global_mean_pool(row_embed, batch)
        if self.return_nodes:
            return embed, row_embed
        return embed

    def embed(self, data: Data, **kwargs) -> Tensor:
        """Generate an embedding for a single entity."""
        self.return_nodes = False
        return self.forward(data).detach()
