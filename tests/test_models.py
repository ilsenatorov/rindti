from copy import deepcopy

import pytest
import torch
from torch_geometric.loader import DataLoader

from rindti.models import ClassificationModel, NoisyNodesClassModel, NoisyNodesRegModel, RegressionModel
from rindti.models.base_model import node_embedders, poolers
from rindti.utils import MyArgParser
from rindti.utils.data import TwoGraphData

# "drug_node_embed": list(node_embedders.keys()),
# "prot_node_embed": list(node_embedders.keys()),
# "drug_pool": list(poolers.keys()),
# "prot_pool": list(poolers.keys()),


fake_data = {
    "prot_x": torch.randint(low=1, high=5, size=(15,)),
    "drug_x": torch.randint(low=1, high=5, size=(15,)),
    "prot_edge_index": torch.randint(low=0, high=5, size=(2, 10)),
    "drug_edge_index": torch.randint(low=0, high=5, size=(2, 10)),
    "prot_edge_feats": torch.randint(low=0, high=5, size=(10,)),
    "drug_edge_feats": torch.randint(low=0, high=5, size=(10,)),
    "label": torch.tensor([1]),
}

dl = DataLoader([TwoGraphData(**fake_data)] * 10, batch_size=5, num_workers=1, follow_batch=["prot_x", "drug_x"])
fake_data = next(iter(dl))


class BaseTestModel:
    """Abstract class for model testing"""

    default_config = {
        "corruption": "mask",
        "drug_alpha": 1,
        "drug_deg": torch.zeros((100), dtype=torch.long),
        "drug_dropout": 0.2,
        "drug_edge_dim": 3,
        "drug_feat_dim": 9,
        "drug_feat_embed_dim": 32,
        "drug_frac": 0.05,
        "drug_hidden_dim": 32,
        "drug_max_nodes": 78,
        "drug_node_embed": "ginconv",
        "drug_num_heads": 4,
        "drug_num_layers": 3,
        "drug_pool": "gmt",
        "drug_pretrain": False,
        "drug_ratio": 0.25,
        "early_stop_patience": 60,
        "feat_method": "element_l1",
        "lr": 0.0005,
        "mlp_dropout": 0.2,
        "mlp_hidden_dim": 64,
        "momentum": 0.3,
        "optimiser": "adamw",
        "prot_alpha": 1,
        "prot_deg": torch.zeros((100), dtype=torch.long),
        "prot_dropout": 0.2,
        "prot_edge_dim": 5,
        "prot_feat_dim": 20,
        "prot_feat_embed_dim": 32,
        "prot_frac": 0.05,
        "prot_hidden_dim": 32,
        "prot_max_nodes": 593,
        "prot_node_embed": "ginconv",
        "prot_num_heads": 4,
        "prot_num_layers": 3,
        "prot_pool": "gmt",
        "prot_pretrain": False,
        "prot_ratio": 0.25,
        "reduce_lr_factor": 0.1,
        "reduce_lr_patience": 20,
        "seed": 42,
        "weight_decay": 0.01,
        "weighted": 0,
    }

    @pytest.mark.parametrize("prot_node_embed", list(node_embedders.keys()))
    @pytest.mark.parametrize("drug_node_embed", list(node_embedders.keys()))
    @pytest.mark.parametrize("prot_pool", list(poolers.keys()))
    @pytest.mark.parametrize("drug_pool", list(poolers.keys()))
    def test_shared_step(self, drug_node_embed, prot_node_embed, drug_pool, prot_pool):
        """Test .__init__"""
        self.default_config["prot_node_embed"] = prot_node_embed
        self.default_config["prot_pool"] = prot_pool
        self.default_config["drug_node_embed"] = drug_node_embed
        self.default_config["drug_pool"] = drug_pool
        model = self.model(**self.default_config)
        data = deepcopy(fake_data)
        model.shared_step(data)

    def test_arg_parser(self):
        parser = MyArgParser()
        self.model.add_arguments(parser)


class TestClassModel(BaseTestModel):
    """Classification Model"""

    model = ClassificationModel

    @pytest.mark.parametrize("feat_method", ["element_l1", "element_l2", "mult", "concat"])
    def test_feat_methods(self, feat_method: str):
        """Test feature concatenation"""
        self.default_config["feat_method"] = feat_method
        model = self.model(**self.default_config)
        prot = torch.rand((32, 64), dtype=torch.float32)
        drug = torch.rand((32, 64), dtype=torch.float32)
        combined = model.merge_features(drug, prot)
        assert combined.size(0) == 32
        if feat_method == "concat":
            assert combined.size(1) == 128
        else:
            assert combined.size(1) == 64


class TestNoisyNodesClassModel(BaseTestModel):
    """Noisy Nodes"""

    model = NoisyNodesClassModel


class TestNoisyNodesRegModel(BaseTestModel):
    """Noisy Nodes"""

    model = NoisyNodesRegModel


class TestRegressionModel(BaseTestModel):
    """Regression Model"""

    model = RegressionModel
