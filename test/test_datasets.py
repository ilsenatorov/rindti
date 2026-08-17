import pytest

from rindti.data import DTIDataset


@pytest.mark.snakemake
def test_dti_dataset(dti_pickle):
    ds = DTIDataset(dti_pickle, "test")
    assert len(ds) == int(25 * 0.7)
    assert "snakemake" in ds.config
