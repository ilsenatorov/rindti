from copy import deepcopy

import pytest
import torch

from ..models import ClassificationModel, NoisyNodesModel
from ..models.base_model import node_embedders, poolers
from ..utils.data import TwoGraphData

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
    "prot_x_batch": torch.zeros((15,), dtype=torch.long),
    "drug_x_batch": torch.zeros((15,), dtype=torch.long),
    "label": torch.tensor([1]),
}


class BaseTestModel:
    """Abstract class for model testing"""

    default_config = {
        "drug_alpha": 1,
        "drug_deg": torch.zeros((100), dtype=torch.long),
        "drug_dropout": 0.2,
        "drug_edge_dim": 3,
        "drug_feat_dim": 9,
        "drug_feat_embed_dim": 32,
        "drug_frac": 0.05,
        "drug_hidden_dim": 32,
        "drug_max_nodes": 78,
        "drug_num_heads": 4,
        "drug_num_layers": 3,
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
        "prot_num_heads": 4,
        "prot_num_layers": 3,
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
    def test_init(self, drug_node_embed, prot_node_embed, drug_pool, prot_pool):
        """Test .__init__"""
        self.default_config["prot_node_embed"] = prot_node_embed
        self.default_config["prot_pool"] = prot_pool
        self.default_config["drug_node_embed"] = drug_node_embed
        self.default_config["drug_pool"] = drug_pool
        self.model(**self.default_config)

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
        model.shared_step(TwoGraphData(**data))


class TestClassModel(BaseTestModel):
    """Classification Model"""

    model = ClassificationModel


class TestNoisyNodesModel(BaseTestModel):
    """Noisy Nodes"""

    model = NoisyNodesModel
