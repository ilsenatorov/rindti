import pytest

from rindti.data import DTIDataset, LargePreTrainDataset, PreTrainDataset

from .conftest import N_INTER, N_PROTS, dti_pickle, pretrain_pickle, sharded_pretrain_pickle


@pytest.mark.parametrize(
    "ds_pickle, ds_class, expected",
    [
        ("dti_pickle", DTIDataset, N_INTER),
        ("pretrain_pickle", PreTrainDataset, N_PROTS),
        ("sharded_pretrain_pickle", LargePreTrainDataset, N_PROTS),
    ],
)
def test_ds(ds_pickle, ds_class, expected, request):
    ds = ds_class(request.getfixturevalue(ds_pickle))
    ds.process()
    assert len(ds) == expected
