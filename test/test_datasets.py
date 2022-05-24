from rindti.data import DTIDataset, PreTrainDataset


def test_dti_dataset(dti_pickle):
    ds = DTIDataset(dti_pickle, "test")
    ds.process()
    assert len(ds) == 25


def test_pretrain_dataset(pretrain_pickle):
    ds = PreTrainDataset(pretrain_pickle)
    ds.process()
    assert len(ds) == 5
