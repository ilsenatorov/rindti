from torch_geometric.nn import global_mean_pool

from ..base_layer import BaseLayer


class MeanPool(BaseLayer):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, batch, **kwargs):
        return global_mean_pool(x, batch)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('pool', default='mean', type=str)
        return parser
