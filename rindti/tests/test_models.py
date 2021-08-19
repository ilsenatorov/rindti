import torch

from ..models import ClassificationModel, NoisyNodesModel
from ..utils.data import TwoGraphData

default_config = {
    "batch_size": 256,
    "drug_deg": torch.tensor([1, 1, 0], dtype=torch.long),
    "drug_dropout": 0.2,
    "drug_edge_dim": 3,
    "drug_feat_dim": 9,
    "drug_feat_embed_dim": 32,
    "drug_hidden_dim": 32,
    "drug_max_nodes": 78,
    "drug_node_embed": "ginconv",
    "drug_num_heads": 4,
    "drug_num_layers": 3,
    "drug_pool": "gmt",
    "drug_ratio": 0.25,
    "early_stop_patience": 60,
    "feat_method": "element_l1",
    "lr": 0.0005,
    "mlp_dropout": 0.2,
    "mlp_hidden_dim": 64,
    "momentum": 0.3,
    "optimiser": "adamw",
    "prot_deg": torch.tensor([1, 1, 0], dtype=torch.long),
    "prot_dropout": 0.2,
    "prot_edge_dim": 5,
    "prot_feat_dim": 20,
    "prot_feat_embed_dim": 32,
    "prot_hidden_dim": 32,
    "prot_max_nodes": 593,
    "prot_node_embed": "ginconv",
    "prot_num_heads": 4,
    "prot_num_layers": 3,
    "prot_pool": "gmt",
    "prot_ratio": 0.25,
    "reduce_lr_factor": 0.1,
    "reduce_lr_patience": 20,
    "seed": 42,
    "weight_decay": 0.01,
    "weighted": 0,
    "prot_frac": 0.05,
    "drug_frac": 0.05,
    "prot_alpha": 1,
    "drug_alpha": 1,
    "prot_pretrain": False,
    "drug_pretrain": False,
}

fake_data = {
    "prot_x": torch.randint(low=0, high=5, size=(15,)),
    "drug_x": torch.randint(low=0, high=5, size=(15,)),
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

    def test_init(self):
        """Test .__init__"""
        self.model(**default_config)

    def test_shared_step(self):
        """Test .__shared_step"""
        model = self.model(**default_config)
        model.shared_step(TwoGraphData(**fake_data))


class TestClassModel(BaseTestModel):
    """Classification Model"""

    model = ClassificationModel


class TestNoisyNodesModel(BaseTestModel):
    """Noisy Nodes"""

    model = NoisyNodesModel
