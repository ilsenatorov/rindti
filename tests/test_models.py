import pytest
import torch

from rindti.models import ClassificationModel, RegressionModel
from rindti.models.base_model import node_embedders, poolers
from rindti.utils import MyArgParser


@pytest.fixture
def default_config():
    return {
        "alpha": 1,
        "corruption": "mask",
        "drug_alpha": 1,
        "drug_dropout": 0.2,
        "drug_frac": 0.05,
        "drug_hidden_dim": 32,
        "drug_node_embed": "ginconv",
        "prot_feat_type": "label",
        "prot_edge_type": "none",
        "drug_feat_type": "label",
        "drug_edge_type": "none",
        "drug_num_heads": 4,
        "drug_num_layers": 3,
        "drug_pool": "gmt",
        "drug_pretrain": False,
        "drug_ratio": 0.25,
        "early_stop_patience": 60,
        "feat_method": "element_l1",
        "lr": 0.001,
        "mlp_dropout": 0.2,
        "momentum": 0.3,
        "optimiser": "adamw",
        "prot_alpha": 1,
        "prot_dropout": 0.2,
        "prot_frac": 0.05,
        "prot_hidden_dim": 32,
        "prot_node_embed": "ginconv",
        "prot_num_heads": 4,
        "prot_num_layers": 3,
        "prot_pool": "gmt",
        "prot_pretrain": False,
        "prot_ratio": 0.25,
        "reduce_lr_factor": 0.1,
        "reduce_lr_patience": 20,
        "temperature": 1,
        "seed": 42,
        "weight_decay": 0.01,
        "weighted": 0,
    }


class BaseTestModel:
    @pytest.mark.parametrize("prot_node_embed", list(node_embedders.keys()))
    @pytest.mark.parametrize("prot_pool", list(poolers.keys()))
    def test_no_edge_shared_step(
        self,
        prot_node_embed,
        prot_pool,
        dti_dataset,
        dti_batch,
        default_config,
    ):
        default_config["prot_node_embed"] = prot_node_embed
        default_config["prot_pool"] = prot_pool
        default_config.update(dti_dataset.config)
        default_config["prot_edge_type"] = "none"
        default_config["drug_edge_type"] = "none"
        model = self.model(**default_config)
        print(default_config)
        if default_config["prot_pool"] == "filmconv" and default_config["prot_edge_type"] == "onehot":
            with pytest.raises(AssertionError):
                model.shared_step(dti_batch)
        model.shared_step(dti_batch)


class TestClassModel(BaseTestModel):

    model = ClassificationModel

    @pytest.mark.parametrize("feat_method", ["element_l1", "element_l2", "mult", "concat"])
    def test_feat_methods(self, feat_method, default_config):
        """Test feature concatenation"""
        default_config["feat_method"] = feat_method
        default_config["prot_feat_dim"] = 1
        default_config["drug_feat_dim"] = 1
        model = self.model(**default_config)
        prot = torch.rand((32, 64), dtype=torch.float32)
        drug = torch.rand((32, 64), dtype=torch.float32)
        combined = model.merge_features(drug, prot)
        assert combined.size(0) == 32
        if feat_method == "concat":
            assert combined.size(1) == 128
        else:
            assert combined.size(1) == 64


class TestRegressionModel(BaseTestModel):

    model = RegressionModel
