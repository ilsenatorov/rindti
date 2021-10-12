from rindti.data import DTIDataset, PreTrainDataset

from .conftest import N_INTER, N_PROTS


def test_dataset(dti_pickle):
    ds = DTIDataset(dti_pickle)
    assert len(ds) == N_INTER


def test_pretrain_dataset(pretrain_pickle):
    ds = PreTrainDataset(pretrain_pickle)
    assert len(ds) == N_PROTS
