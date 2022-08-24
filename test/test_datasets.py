from rindti.data import DTIDataset, ProtPreTrainDataset


def test_dti_dataset(dti_pickle):
    ds = DTIDataset(dti_pickle, "test")
    ds.process()
    assert len(ds) == int(25 * 0.7)


def test_pretrain_dataset(pretrain_pickle):
    ds = ProtPreTrainDataset(pretrain_pickle)
    ds.process()
    assert len(ds) == 5
