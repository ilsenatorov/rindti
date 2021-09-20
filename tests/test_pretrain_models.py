from copy import deepcopy

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from rindti.models import BGRLModel, GraphLogModel, InfoGraphModel, PfamModel
from rindti.models.base_model import node_embedders, poolers
from rindti.utils.data import TwoGraphData

fake_data = {
    "edge_feats": torch.randint(low=0, high=5, size=(10,)),
    "edge_index": torch.randint(low=0, high=5, size=(2, 10)),
    "label": torch.tensor([1]),
    "x": torch.ones(size=(15,), dtype=torch.long),
}


fake_pfam_data = {
    "a_edge_feats": torch.randint(low=0, high=5, size=(10,)),
    "a_edge_index": torch.randint(low=0, high=5, size=(2, 10)),
    "a_x": torch.randint(low=0, high=5, size=(15,)),
    "b_edge_feats": torch.randint(low=0, high=5, size=(10,)),
    "b_edge_index": torch.randint(low=0, high=5, size=(2, 10)),
    "b_x": torch.randint(low=0, high=5, size=(15,)),
    "label": torch.tensor([1]),
}

fake_data = next(iter(DataLoader([Data(**fake_data)] * 10, batch_size=5, num_workers=1)))
fake_pfam_data = next(
    iter(DataLoader([TwoGraphData(**fake_pfam_data)] * 10, batch_size=5, num_workers=1, follow_batch=["a_x", "b_x"]))
)


class BaseTestModel:
    """Abstract class for model testing"""

    default_config = {
        "alpha": 1.0,
        "batch_size": 512,
        "beta": 1.0,
        "corruption": "mask",
        "data": "kek",
        "decay_ratio": 0.5,
        "dropout": 0.2,
        "early_stop_patience": 60,
        "edge_dim": 5,
        "feat_dim": 20,
        "feat_embed_dim": 32,
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
        "momentum": 0.3,
        "num_layers": 3,
        "num_proto": 8,
        "num_workers": 4,
        "optimiser": "adamw",
        "pooling_method": "mincut",
        "ratio": 0.25,
        "reduce_lr_factor": 0.1,
        "reduce_lr_patience": 20,
        "seed": 42,
        "weight_decay": 0.01,
        "weighted": 1,
    }

    @pytest.mark.parametrize("node_embed", list(node_embedders.keys()))
    @pytest.mark.parametrize("pool", list(poolers.keys()))
    def test_init(self, node_embed, pool):
        """Test .__init__"""
        self.default_config["node_embed"] = node_embed
        self.default_config["pool"] = pool
        self.model(**self.default_config)

    @pytest.mark.parametrize("node_embed", list(node_embedders.keys()))
    @pytest.mark.parametrize("pool", list(poolers.keys()))
    def test_shared_step(self, node_embed, pool):
        """Test .shared_step"""
        self.default_config["node_embed"] = node_embed
        self.default_config["pool"] = pool
        model = self.model(**self.default_config)
        data = deepcopy(fake_data)
        model.shared_step(data)


class TestGraphLogModel(BaseTestModel):
    """GraphLog"""

    model = GraphLogModel


class TestInfoGraphModel(BaseTestModel):
    """InfoGraph"""

    model = InfoGraphModel


class TestBGRLModel(BaseTestModel):
    """BGRL"""

    model = BGRLModel


class TestPfamModel(BaseTestModel):
    model = PfamModel

    @pytest.mark.parametrize("node_embed", list(node_embedders.keys()))
    @pytest.mark.parametrize("pool", list(poolers.keys()))
    def test_shared_step(self, node_embed, pool):
        """Test .shared_step"""
        self.default_config["node_embed"] = node_embed
        self.default_config["pool"] = pool
        model = self.model(**self.default_config)
        model.shared_step(fake_pfam_data)
