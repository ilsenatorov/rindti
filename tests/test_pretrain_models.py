import pytest

from rindti.models import BGRLModel, DistanceModel, GraphLogModel, InfoGraphModel
from rindti.models.encoder import node_embedders, poolers

from .conftest import BATCH_SIZE, PROT_EDGE_DIM, PROT_FEAT_DIM, PROT_PER_FAM


@pytest.fixture
def default_config():
    return {
        "alpha": 1.0,
        "beta": 1.0,
        "batch_size": BATCH_SIZE,
        "corruption": "mask",
        "edge_type": "none",
        "decay_ratio": 0.5,
        "dropout": 0.2,
        "early_stop_patience": 60,
        "edge_dim": 5,
        "feat_dim": 20,
        "feat_type": "label",
        "feat_method": "element_l1",
        "frac": 0.15,
        "gamma": 0.1,
        "gpus": 1,
        "gradient_clip_val": 10,
        "hidden_dim": 32,
        "hierarchy": 3,
        "lr": 0.0005,
        "margin": 1.0,
        "mask_rate": 0.3,
        "max_epochs": 1000,
        "loss": "snnl",
        "temp": 1,
        "optim_temp": True,
        "momentum": 0.3,
        "num_layers": 3,
        "num_proto": 8,
        "num_workers": 4,
        "optimiser": "adamw",
        "pooling_method": "mincut",
        "prot_per_fam": PROT_PER_FAM,
        "ratio": 0.25,
        "reduce_lr_factor": 0.1,
        "reduce_lr_patience": 20,
        "seed": 42,
        "weight_decay": 0.01,
        "weighted": 1,
    }


class BaseTestModel:
    @pytest.mark.parametrize("node_embed", list(node_embedders.keys()))
    @pytest.mark.parametrize("pool", list(poolers.keys()))
    def test_init(self, node_embed, pool, default_config):
        default_config["node_embed"] = node_embed
        default_config["pool"] = pool
        self.model(**default_config)

    @pytest.mark.parametrize("node_embed", list(node_embedders.keys()))
    @pytest.mark.parametrize("pool", list(poolers.keys()))
    def test_shared_step(self, node_embed, pool, pretrain_batch, default_config, pretrain_dataset):
        default_config["node_embed"] = node_embed
        default_config["pool"] = pool
        default_config["feat_dim"] = PROT_FEAT_DIM
        default_config["edge_dim"] = PROT_EDGE_DIM
        default_config.update(pretrain_dataset.config)
        model = self.model(**default_config)
        model.shared_step(pretrain_batch)


class TestGraphLogModel(BaseTestModel):
    model = GraphLogModel


class TestInfoGraphModel(BaseTestModel):

    model = InfoGraphModel


class TestBGRLModel(BaseTestModel):

    model = BGRLModel


class TestDistanceModel(BaseTestModel):

    model = DistanceModel
