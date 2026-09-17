import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from rindti.layers.graphpool import AttentionPool, MeanPool, Set2SetPool

default_config = {
    "K": 1,
    "deg": torch.randint(low=0, high=5, size=(10,)),
    "dropout": 0.2,
    "edge_dim": 6,
    "hidden_dim": 64,
    "input_dim": 16,
    "num_heads": 4,
    "output_dim": 32,
}


fake_data = {
    "edge_feats": torch.randint(low=0, high=5, size=(10,)),
    "edge_index": torch.randint(low=0, high=5, size=(2, 10)),
    "x": torch.rand(size=(13, 16)),
}

fake_data = next(iter(DataLoader([Data(**fake_data)] * 10, batch_size=5, num_workers=1)))


class BaseTestGraphPool:
    def test_init(self):
        self.module(**default_config)

    def test_forward(self):
        module = self.module(**default_config)
        output = module.forward(fake_data.x, fake_data.edge_index, fake_data.batch)
        assert output.size(0) == 5
        assert output.size(1) == 32

    # def test_args(self):
    #     parser = MyArgParser()
    #     self.module.add_arguments(parser)

    def test_norm(self):
        """Pooling should return vector of length 1 for each graph"""
        module = self.module(**default_config)
        output = module.forward(fake_data.x, fake_data.edge_index, fake_data.batch)
        length = output.detach().norm(dim=1)
        assert ((length - 1.0).abs() < 1e-6).all()  # soft equal


class TestAttentionPool(BaseTestGraphPool):
    module = AttentionPool

    def test_exposes_one_weight_per_node(self):
        """The interpretability readout: a scalar per residue after a forward pass."""
        module = self.module(**default_config)
        module.forward(fake_data.x, fake_data.edge_index, fake_data.batch)
        assert module.attention_weights.shape == (fake_data.x.size(0),)
        assert not module.attention_weights.requires_grad


class TestSet2SetPool(BaseTestGraphPool):
    module = Set2SetPool


class TestMeanPool(BaseTestGraphPool):
    module = MeanPool

    def test_forward(self):
        """MeanPool always return same dim as input"""
        module = self.module(**default_config)
        output = module.forward(fake_data.x, fake_data.edge_index, fake_data.batch)
        assert output.size(0) == 5
        assert output.size(1) == 16
