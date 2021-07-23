from ..base_layer import BaseLayer
from torch_geometric.nn import GATv2Conv, GraphSizeNorm


class GatConvNet(BaseLayer):
    def __init__(self, input_dim, output_dim, hidden_dim=32, heads=4, **kwargs):
        super().__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads)
        self.conv2 = GATv2Conv(hidden_dim * heads, output_dim, heads=1)
        self.norm = GraphSizeNorm()

    def forward(self, x, edge_index, batch, **kwargs):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.norm(x, batch)
        return x

    @staticmethod
    def add_arguments(group):
        group.add_argument('node_embed', default='gatconv', type=str)
        group.add_argument('hidden_dim', default=32, type=int, help='Number of hidden dimensions')
        group.add_argument('dropout', default=0.2, type=float, help='Dropout')
        return group
