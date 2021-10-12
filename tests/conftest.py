import pickle
import string
from random import choice, randint

import pandas as pd
import pytest
import torch
from torch_geometric.loader import DataLoader

from rindti.data import DTIDataset, PfamSampler, PreTrainDataset

N_PROTS = 100
N_DRUGS = 100
N_INTER = 250
BATCH_SIZE = 16
PROT_PER_FAM = 8
BATCH_PER_EPOCH = 10


def randomword(length: int) -> str:
    letters = string.ascii_lowercase
    return "".join(choice(letters) for i in range(length))


def create_fake_graph(n_nodes, n_features, fam: list = None):
    d = {
        "x": torch.randint(1, n_features, (n_nodes,), dtype=torch.long),
        "edge_index": torch.randint(low=0, high=5, size=(2, 10)),
        "id": randomword(20),
    }
    if fam:
        d["fam"] = choice(fam)
    return d


def create_fake_dataset(n_prots, n_drugs, n_interactions):
    prots = pd.Series([create_fake_graph(randint(100, 200), 20) for _ in range(n_prots)], name="data")
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
def dti_dataset(dti_pickle):
    ds = DTIDataset(dti_pickle)
    return ds


@pytest.fixture(scope="session")
def dti_dataloader(dti_dataset):
    dl = DataLoader(dti_dataset, batch_size=BATCH_SIZE, follow_batch=["prot_x", "drug_x"])
    return dl


@pytest.fixture
def dti_batch(dti_dataloader):
    batch = next(iter(dti_dataloader))
    return batch


@pytest.fixture(scope="session")
def pretrain_dataset(pretrain_pickle):
    ds = PreTrainDataset(pretrain_pickle)
    return ds


@pytest.fixture(scope="session")
def pfam_sampler(pretrain_dataset):
    sampler = PfamSampler(
        pretrain_dataset,
        batch_size=BATCH_SIZE,
        prot_per_fam=PROT_PER_FAM,
        batch_per_epoch=BATCH_PER_EPOCH,
    )
    return sampler


@pytest.fixture(scope="session")
def pretrain_dataloader(pretrain_dataset, pfam_sampler):
    dl = DataLoader(pretrain_dataset, batch_sampler=pfam_sampler)
    return dl


@pytest.fixture
def pretrain_batch(pretrain_dataloader):
    batch = next(iter(pretrain_dataloader))
    return batch
