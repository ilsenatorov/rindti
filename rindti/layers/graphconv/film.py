from torch import Tensor
from torch.nn import ModuleList
from torch_geometric.nn import FiLMConv
from torch_geometric.typing import Adj

from ..base_layer import BaseLayer


class FilmConvNet(BaseLayer):
    r"""FiLM Convolution.

    Refer to :class:`torch_geometric.nn.conv.FiLMConv` for more details.


    Args:
        input_dim (int): Size of the input vector
        output_dim (int): Size of the output vector
        hidden_dim (int, optional): Size of the hidden layer(s). Defaults to 32.
        edge_dim (int, optional): Size of the edge input vector. Defaults to None.
        num_layers (int, optional): Number of layers. Defaults to 10.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 32,
        edge_dim: int = None,
        num_layers: int = 10,
        **kwargs,
    ):
        super().__init__()
        # `not edge_dim`, not `is None`: workflow/scripts/utils.py reports edge_dim 0
        # for a dataset with no edge features, which is the shipped default. Passing
        # num_relations=0 to FiLMConv builds no relation weights at all and raises
        # `IndexError: index 0 is out of range` on the first forward pass.
        if not edge_dim:
            edge_dim = 1
        self.edge_dim = edge_dim
        self.inp = FiLMConv(input_dim, hidden_dim, num_relations=edge_dim)
        mid_layers = [FiLMConv(hidden_dim, hidden_dim, num_relations=edge_dim) for _ in range(num_layers - 2)]
        self.mid_layers = ModuleList(mid_layers)

        self.out = FiLMConv(hidden_dim, output_dim, num_relations=edge_dim)

    def forward(self, x: Tensor, edge_index: Adj, edge_feats: Tensor = None, **kwargs) -> Tensor:
        """"""
        # FiLMConv takes discrete relation types, not a continuous attribute, so a
        # 1-dimensional edge feature (`prots.features.edge_feats: distance`) is
        # dropped here. A filmconv run with distance edges is therefore identical to
        # one without - do not read the null difference as a scientific result; use
        # `transformer` to actually measure continuous edge features.
        if self.edge_dim <= 1:
            edge_feats = None
        x = self.inp(x, edge_index, edge_feats)
        for module in self.mid_layers:
            x = module(x, edge_index, edge_feats)
        x = self.out(x, edge_index, edge_feats)
        return x
