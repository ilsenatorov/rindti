import pickle
from pprint import pprint
from random import randint

import pandas as pd
import pytest
import torch

from rindti.utils.data import Dataset, PreTrainDataset

N_PROTS = 10
N_DRUGS = 10
N_INTER = 40


def create_fake_graph(n_nodes, n_features):
    """Fake graph data"""
    return {
        "x": torch.randint(1, n_features, (n_nodes,), dtype=torch.long),
        "edge_index": torch.randint(low=0, high=5, size=(2, 10)),
    }


def create_fake_dataset(n_prots, n_drugs, n_interactions):
    """Create fake dict that imitates DTI dataset"""
    prots = pd.Series([create_fake_graph(randint(100, 200), 20) for _ in range(n_prots)], name="data")
    drugs = pd.Series([create_fake_graph(randint(100, 200), 20) for _ in range(n_drugs)], name="data")
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
    return dict(prots=prots, drugs=drugs, data=inter, config={"prot_feat_dim": 20, "drug_feat_dim": 20})


@pytest.fixture(scope="session")
def fake_dataset(tmpdir_factory):
    """Create a pickle file with fake DTI dataset"""
    fake_dataset = create_fake_dataset(N_PROTS, N_DRUGS, N_INTER)
    fn = tmpdir_factory.mktemp("data").join("temp_data.pkl")
    with open(fn, "wb") as file:
        pickle.dump(fake_dataset, file)
    # pprint(fake_dataset)
    return fn


@pytest.fixture(scope="session")
def fake_pretrain_dataset(tmpdir_factory):
    """Create a pickle file with fake protein dataset"""
    ds = pd.Series([create_fake_graph(randint(100, 200), 20) for _ in range(N_PROTS)], name="data")
    ds = pd.DataFrame(ds)
    fn = tmpdir_factory.mktemp("data").join("temp_pretrain_data.pkl")
    with open(fn, "wb") as file:
        pickle.dump(ds, file)
    return fn


def test_dataset(fake_dataset):
    """Test normal dataset"""
    ds = Dataset(fake_dataset)
    assert len(ds) == N_INTER


def test_pretrain_dataset(fake_pretrain_dataset):
    """Test pretrain dataset"""
    ds = PreTrainDataset(fake_pretrain_dataset)
    assert len(ds) == N_PROTS
