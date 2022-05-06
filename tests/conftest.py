import pickle
from random import choice, randint

import pandas as pd
import pytest
import torch
from pytorch_lightning.utilities.seed import seed_everything
from torch.functional import Tensor
from torch_geometric.loader import DataLoader

from rindti.data import DTIDataModule, DTIDataset, PfamSampler, PreTrainDataModule, PreTrainDataset

PROT_FEAT_DIM = 20
PROT_EDGE_DIM = 5
DRUG_FEAT_DIM = 14
DRUG_EDGE_DIM = 3
MIN_NODES = 50
MAX_NODES = 100
MIN_EDGES = 50
MAX_EDGES = 100
N_PROTS = 50
N_DRUGS = 50
N_INTER = 100
BATCH_SIZE = 16
PROT_PER_FAM = 8
BATCH_PER_EPOCH = 10


def create_features(feat_type: str, n: int, feat_dim: int) -> Tensor:
    if feat_type == "onehot":
        return torch.rand((n, feat_dim))
    elif feat_type == "label":
        return torch.randint(low=1, high=feat_dim, size=(n,), dtype=torch.long)
    else:
        return None


def create_fake_graph(
    node_attr_type: str,
    node_dim: int,
    edge_attr_type: str,
    edge_dim: int,
    fam: list = None,
    split: list = None,
):
    n_nodes = randint(MIN_NODES, MAX_NODES)
    n_edges = randint(MIN_EDGES, MAX_EDGES)
    d = dict(
        x=create_features(node_attr_type, n_nodes, node_dim),
        edge_index=torch.randint(0, n_nodes, (2, n_edges)),
        edge_attr=create_features(edge_attr_type, n_edges, edge_dim),
    )
    if fam:
        d["y"] = choice(fam)
    return d


def generate_params():
    for edge_type in ["label", "onehot", "none"]:
        for node_type in ["label", "onehot"]:
            yield {"edge_type": edge_type, "node_type": node_type}


def fake_dataset(params):
    prots = pd.Series(
        [
            create_fake_graph(
                params["prot_node_type"],
                PROT_FEAT_DIM,
                params["prot_edge_type"],
                PROT_EDGE_DIM,
            )
            for _ in range(N_PROTS)
        ],
        name="data",
    )
    drugs = pd.Series(
        [
            create_fake_graph(
                "label",
                DRUG_FEAT_DIM,
                "label",
                DRUG_EDGE_DIM,
            )
            for _ in range(N_DRUGS)
        ],
        name="data",
    )
    prots = pd.DataFrame(prots)
    drugs = pd.DataFrame(drugs)
    prots["count"] = 1
    drugs["count"] = 1
    inter = [
        {
            "prot_id": randint(0, N_PROTS - 1),
            "drug_id": randint(0, N_DRUGS - 1),
            "split": choice(["train", "val", "test"]),
            "label": randint(0, 1),
        }
        for _ in range(N_INTER)
    ]
    return dict(
        prots=prots,
        drugs=drugs,
        data=inter,
        config={
            "prot_feat_dim": PROT_FEAT_DIM,
            "drug_feat_dim": DRUG_FEAT_DIM,
            "prot_edge_dim": PROT_EDGE_DIM,
            "drug_edge_dim": DRUG_EDGE_DIM,
        },
    )


@pytest.fixture(scope="session", params=[{"prot_" + k: v for k, v in p.items()} for p in generate_params()])
def dti_pickle(tmpdir_factory, request):
    """Create a pickle file with fake DTI dataset"""
    params = request.param
    ds = fake_dataset(params)
    fn = tmpdir_factory.mktemp("data").join("temp_data.pkl")
    with open(fn, "wb") as file:
        pickle.dump(ds, file)
    print(fn)
    return fn


@pytest.fixture(scope="session", params=generate_params())
def pretrain_pickle(tmpdir_factory, request):
    """Create a pickle file with fake protein dataset"""
    params = request.param
    ds = pd.Series(
        [
            create_fake_graph(
                params["node_type"],
                PROT_FEAT_DIM,
                params["edge_type"],
                PROT_EDGE_DIM,
                fam=["a", "b", "c", "a;b"],
            )
            for _ in range(N_PROTS)
        ],
        name="data",
    )
    ds = pd.DataFrame(ds)
    fn = tmpdir_factory.mktemp("data").join("temp_pretrain_data.pkl")
    ds.to_pickle(fn)
    return fn


@pytest.fixture()
def dti_dataset(dti_pickle):
    return DTIDataset(dti_pickle, "test")


@pytest.fixture()
def dti_datamodule(dti_pickle):
    return DTIDataModule(dti_pickle, "test", batch_size=BATCH_SIZE)


@pytest.fixture()
def dti_dataloader(dti_dataset):
    return DataLoader(dti_dataset, batch_size=BATCH_SIZE, follow_batch=["prot_x", "drug_x"])


@pytest.fixture
def dti_batch(dti_dataloader):
    return next(iter(dti_dataloader))


@pytest.fixture()
def pretrain_dataset(pretrain_pickle):
    return PreTrainDataset(pretrain_pickle)


@pytest.fixture()
def pretrain_datamodule(pretrain_pickle):
    return PreTrainDataModule(pretrain_pickle, "test")


@pytest.fixture()
def pfam_sampler(pretrain_dataset):
    return PfamSampler(
        pretrain_dataset,
        batch_size=BATCH_SIZE,
        prot_per_fam=PROT_PER_FAM,
        batch_per_epoch=BATCH_PER_EPOCH,
    )


@pytest.fixture()
def pretrain_dataloader(pretrain_dataset, pfam_sampler):
    return DataLoader(pretrain_dataset, batch_sampler=pfam_sampler)


@pytest.fixture
def pretrain_batch(pretrain_dataloader):
    return next(iter(pretrain_dataloader))


@pytest.fixture(autouse=True)
def seed():
    seed_everything(42)
