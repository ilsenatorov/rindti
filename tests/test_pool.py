import torch

from rindti.layers import DiffPoolNet, GMTNet, MeanPool
from rindti.utils import MyArgParser
from torch_geometric.data import Data, DataLoader

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
        """Tests .__init__"""
        self.module(**default_config)

    def test_forward(self):
        """Tests .forward"""
        module = self.module(**default_config)
        output = module.forward(**fake_data.__dict__)
        assert output.size(0) == 5
        assert output.size(1) == 32

    def test_args(self):
        """Test arguments parsing"""
        parser = MyArgParser()
        self.module.add_arguments(parser)

    def test_norm(self):
        """Pooling should return vector of length 1 for each graph"""
        module = self.module(**default_config)
        output = module.forward(**fake_data.__dict__)
        length = output.detach().norm(dim=1)
        print(length)
        assert ((length - 1.0).abs() < 1e-6).all()  # soft equal


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
        output = module.forward(**fake_data.__dict__)
        assert output.size(0) == 5
        assert output.size(1) == 16
