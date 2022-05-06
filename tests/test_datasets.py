import pytest

from rindti.data import DTIDataset, LargePreTrainDataset, PreTrainDataset

from .conftest import N_INTER, N_PROTS


def test_dti_dataset(dti_pickle):
    ds = DTIDataset(dti_pickle, "test")
    ds.process()
    assert len(ds) <= N_INTER


def test_pretrain_dataset(pretrain_pickle):
    ds = PreTrainDataset(pretrain_pickle)
    ds.process()
    assert len(ds) == N_PROTS


def test_large_pretrain_dataset(sharded_pretrain_pickle):
    ds = LargePreTrainDataset(sharded_pretrain_pickle)
    ds.process()
    assert len(ds) == N_PROTS
