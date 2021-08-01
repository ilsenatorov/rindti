from torch_geometric.nn import ChebConv

from ..base_layer import BaseLayer


class ChebConvNet(BaseLayer):
    def __init__(self, input_dim, output_dim, hidden_dim=32, K=1, **kwargs):
        super().__init__()
        self.conv1 = ChebConv(input_dim, hidden_dim, K)
        self.conv2 = ChebConv(hidden_dim, output_dim, K)

    def forward(self, x, edge_index, **kwargs):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

    @staticmethod
    def add_arguments(group):
        group.add_argument('node_embed', default='chebconv', type=str)
        group.add_argument('hidden_dim', default=32, type=int, help='Number of hidden dimensions')
        group.add_argument('dropout', default=0.2, type=float, help='Dropout')
        group.add_argument('K', default=1, type=int, help='K argument of chebconv')
        return group
