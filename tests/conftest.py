import os.path as osp
import pickle
import string
from random import choice, randint

import pandas as pd
import pytest
import torch
from torch.functional import Tensor
from torch_geometric.loader import DataLoader

from rindti.data import DTIDataset, PfamSampler, PreTrainDataset

MIN_NODES = 100
MAX_NODES = 200
N_PROTS = 100
N_DRUGS = 100
N_INTER = 250
BATCH_SIZE = 16
PROT_PER_FAM = 8
BATCH_PER_EPOCH = 10


def randomword(length: int) -> str:
    letters = string.ascii_lowercase
    return "".join(choice(letters) for i in range(length))


def create_features(feat_type: str, dim: tuple) -> Tensor:
    if feat_type == "onehot":
        return torch.rand(dim)
    elif feat_type == "label":
        return torch.randint(1, dim[1], (dim[0],), dtype=torch.long)
    else:
        return None


def create_fake_graph(
    node_attr_type: str,
    node_dim: int,
    n_nodes: int,
    edge_attr_type: str,
    edge_dim: int,
    n_edges: int,
    fam: list = None,
):
    d = (
        dict(
            x=create_features(node_attr_type, (n_nodes, node_dim)),
            edge_index=torch.randint(0, n_nodes, (2, n_edges)),
            edge_attr=create_features(edge_attr_type, (n_edges, edge_dim)),
        ),
    )
    if fam:
        d["fam"] = choice(fam)
    return d


def generate_params():
    pass


@pytest.fixture
def create_fake_dataset():
    prots = pd.Series(
        [
            create_fake_graph(
                randint(MIN_NODES, MAX_NODES),
                20,
                randint(100, 100),
            )
            for _ in range(n_prots)
        ],
        name="data",
    )
    drugs = pd.Series([create_fake_graph(randint(100, 200), 9) for _ in range(n_drugs)], name="data")
    prots = pd.DataFrame(prots)
    drugs = pd.DataFrame(drugs)
    prots["count"] = 1
    drugs["count"] = 1
    inter = [
        {
            "prot_id": randint(0, n_prots - 1),
            "drug_id": randint(0, n_drugs - 1),
            "split": "train",
            "label": randint(0, 1),
        }
        for _ in range(n_interactions)
    ]
    return dict(prots=prots, drugs=drugs, data=inter, config={"prot_feat_dim": 20, "drug_feat_dim": 9})


@pytest.fixture(scope="session")
def dti_pickle(tmpdir_factory):
    """Create a pickle file with fake DTI dataset"""
    fake_dataset = create_fake_dataset(N_PROTS, N_DRUGS, N_INTER)
    fn = tmpdir_factory.mktemp("data").join("temp_data.pkl")
    with open(fn, "wb") as file:
        pickle.dump(fake_dataset, file)
    return fn


@pytest.fixture(scope="session")
def pretrain_pickle(tmpdir_factory):
    """Create a pickle file with fake protein dataset"""
    ds = pd.Series([create_fake_graph(randint(100, 200), 20, fam=["a", "b"]) for _ in range(N_PROTS)], name="data")
    ds = pd.DataFrame(ds)
    fn = tmpdir_factory.mktemp("data").join("temp_pretrain_data.pkl")
    ds.to_pickle(fn)
    return fn


@pytest.fixture(scope="session")
def sharded_pretrain_pickle(tmpdir_factory):
    """Create a collection of pickled files with fake protein dataset"""
    ds = pd.Series([create_fake_graph(randint(100, 200), 20, fam=["a", "b"]) for _ in range(N_PROTS)], name="data")
    ds = pd.DataFrame(ds)
    fn = tmpdir_factory.mktemp("shards")
    for i in range(0, len(ds), N_PROTS // 2):
        ds.iloc[i : i + N_PROTS // 2].to_pickle(fn.join("data{}.pkl".format(i)))
    return str(fn)


@pytest.fixture(scope="session")
def dti_dataset(dti_pickle):
    return DTIDataset(dti_pickle)


@pytest.fixture(scope="session")
def dti_dataloader(dti_dataset):
    return DataLoader(dti_dataset, batch_size=BATCH_SIZE, follow_batch=["prot_x", "drug_x"])


@pytest.fixture
def dti_batch(dti_dataloader):
    return next(iter(dti_dataloader))


@pytest.fixture(scope="session")
def pretrain_dataset(pretrain_pickle):
    return PreTrainDataset(pretrain_pickle)


@pytest.fixture(scope="session")
def pfam_sampler(pretrain_dataset):
    return PfamSampler(
        pretrain_dataset,
        batch_size=BATCH_SIZE,
        prot_per_fam=PROT_PER_FAM,
        batch_per_epoch=BATCH_PER_EPOCH,
    )


@pytest.fixture(scope="session")
def pretrain_dataloader(pretrain_dataset, pfam_sampler):
    return DataLoader(pretrain_dataset, batch_sampler=pfam_sampler)


@pytest.fixture
def pretrain_batch(pretrain_dataloader):
    return next(iter(pretrain_dataloader))
