import torch
from torch_geometric.data import Data

from ..models import GraphLogModel, InfoGraphModel, PfamModel
from ..utils.data import TwoGraphData

default_config = {
    "alpha": 1.0,
    "batch_size": 512,
    "beta": 1.0,
    "data": "kek",
    "decay_ratio": 0.5,
    "dropout": 0.2,
    "early_stop_patience": 60,
    "feat_embed_dim": 32,
    "feat_dim": 20,
    "feat_method": "element_l1",
    "frac": 0.15,
    "gamma": 0.1,
    "gpus": 1,
    "gradient_clip_val": 10,
    "hidden_dim": 32,
    "hierarchy": 3,
    "lr": 0.0005,
    "mask_rate": 0.3,
    "max_epochs": 1000,
    "model": "graphlog",
    "momentum": 0.3,
    "node_embed": "ginconv",
    "num_layers": 3,
    "num_proto": 8,
    "num_workers": 4,
    "optimiser": "adamw",
    "pool": "gmt",
    "pooling_method": "mincut",
    "ratio": 0.25,
    "reduce_lr_factor": 0.1,
    "reduce_lr_patience": 20,
    "seed": 42,
    "weight_decay": 0.01,
    "weighted": 1,
}

fake_data = {
    "x": torch.randint(low=0, high=5, size=(15,)),
    "edge_index": torch.randint(low=0, high=5, size=(2, 10)),
    "edge_feats": torch.randint(low=0, high=5, size=(10,)),
    "batch": torch.zeros((15,), dtype=torch.long),
    "label": torch.tensor([1]),
}

fake_pfam_data = {
    "a_x": torch.randint(low=0, high=5, size=(15,)),
    "b_x": torch.randint(low=0, high=5, size=(15,)),
    "a_edge_index": torch.randint(low=0, high=5, size=(2, 10)),
    "b_edge_index": torch.randint(low=0, high=5, size=(2, 10)),
    "a_edge_feats": torch.randint(low=0, high=5, size=(10,)),
    "b_edge_feats": torch.randint(low=0, high=5, size=(10,)),
    "a_x_batch": torch.zeros((15,), dtype=torch.long),
    "b_x_batch": torch.zeros((15,), dtype=torch.long),
    "label": torch.tensor([1]),
}


class BaseTestModel:
    """Abstract class for model testing"""

    def test_init(self):
        """Test .__init__"""
        self.model(**default_config)

    def test_shared_step(self):
        """Test .__shared_step"""
        model = self.model(**default_config)
        model.shared_step(Data(**fake_data))


class TestGraphLogModel(BaseTestModel):
    """GraphLog"""

    model = GraphLogModel


class TestInfoGraphModel(BaseTestModel):
    """InfoGraph"""

    model = InfoGraphModel


class TestPfamModel(BaseTestModel):
    model = PfamModel

    def test_shared_step(self):
        """Test .__shared_step"""
        model = self.model(**default_config)
        model.shared_step(Data(**fake_pfam_data))
