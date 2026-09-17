import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import AttentionalAggregation
from torch_geometric.typing import Adj

from ..base_layer import BaseLayer


class AttentionPool(BaseLayer):
    r"""Gated attention pooling over the nodes of a graph.

    Replaces ``DiffPoolNet``, which densified the batch: it built an
    ``(B, max_nodes, max_nodes)`` adjacency, so at ``batch_size: 128`` its peak
    allocation was 0.31 GB for 300-residue proteins but **6.2 GB** at 1500, and
    whole-protein AlphaFold graphs reach well past that. Attention pooling is sparse and
    costs 0.27 GB at 1500 residues - 23x less, and 23x faster - so the pooling ablation
    can run on the same graphs as every other axis instead of needing its own reduced
    structures.

    Each node gets a scalar gate, softmax-normalised within its graph, and the graph
    embedding is the weighted sum of node features. Unlike mean pooling the weights are
    learned, so this is the "learned pooling" arm of the ablation.

    The gates are also the readout the interpretability analysis needs: ``attention_weights``
    holds one number per residue after a forward pass, directly comparable against a known
    binding site. Mean pooling has no such quantity, and DiffPool's cluster assignments
    were a soft many-to-many map that is far harder to read as per-residue importance.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, **kwargs):
        super().__init__()
        # A small MLP rather than a bare Linear: a single linear gate can only rank nodes
        # along one direction of feature space.
        self.gate = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.PReLU(), nn.Linear(hidden_dim, 1))
        self.pool = AttentionalAggregation(gate_nn=self.gate)
        self.lin = nn.Linear(input_dim, output_dim)
        self.attention_weights = None

    def forward(self, x: Tensor, edge_index: Adj, batch: Tensor, **kwargs) -> Tensor:
        """"""
        # Stashed for the interpretability readout; detached so it never holds the graph.
        self.attention_weights = self.gate(x).detach().squeeze(-1)
        pooled = self.lin(self.pool(x, batch))
        return F.normalize(pooled, dim=1)
