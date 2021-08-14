import torch

from ..layers import DiffPoolNet, GMTNet, MeanPool

default_config = {
    "input_dim": 16,
    "output_dim": 32,
    "hidden_dim": 64,
    "num_heads": 4,
    "K": 1,
    "dropout": 0.2,
    "edge_dim": 6,
    "deg": torch.randint(low=0, high=5, size=(10,)),
}


fake_data = {
    "x": torch.rand(size=(13, 16)),
    "edge_index": torch.randint(low=0, high=5, size=(2, 10)),
    "edge_feats": torch.randint(low=0, high=5, size=(10,)),
    "batch": torch.zeros((13,), dtype=torch.long),
}


class BaseTestGraphPool:
    def test_init(self):
        """Tests .__init__"""
        self.module(**default_config)

    def test_forward(self):
        """Tests .forward"""
        module = self.module(**default_config)
        output = module.forward(**fake_data)
        assert output.size(0) == 1
        assert output.size(1) == 32


class TestGMTNet(BaseTestGraphPool):
    """Tests GMTNet"""

    module = GMTNet


class TestDiffPool(BaseTestGraphPool):
    """Tests DiffPool"""

    module = DiffPoolNet


class TestMeanPool(BaseTestGraphPool):
    """Tests MeanPool"""

    module = MeanPool

    def test_forward(self):
        """MeanPool always return same dim as input"""
        module = self.module(**default_config)
        output = module.forward(**fake_data)
        assert output.size(0) == 1
        assert output.size(1) == 16