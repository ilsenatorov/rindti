import os

import pytest

from rindti.data import DTIDataset


@pytest.mark.snakemake
def test_dti_dataset(dti_pickle):
    ds = DTIDataset(dti_pickle, "test")
    assert len(ds) == int(25 * 0.7)
    assert "snakemake" in ds.config


@pytest.mark.snakemake
class TestProcessLock:
    """Concurrent jobs sharing an (exp_name, dataset) pair used to race.

    The cache key is the directory, so submitting several model configs against one
    dataset had them all find the cache cold and all write into the same `processed/`.
    `InMemoryDataset.save` is not atomic, so a loser could read a half-written `.pt`.
    """

    def test_cache_is_reused_not_rebuilt(self, dti_pickle, tmp_path, monkeypatch):
        """Guards the naming trap: `_process` is torch_geometric's own method, the one
        that decides whether the cache is warm. Defining it on the subclass overrides
        that check, so every construction rebuilds and `process()` never runs."""
        monkeypatch.chdir(tmp_path)
        builds = []
        original = DTIDataset._build_splits
        monkeypatch.setattr(
            DTIDataset,
            "_build_splits",
            lambda self: (builds.append(1), original(self))[1],
        )

        DTIDataset(dti_pickle, "locktest", split="train")
        assert len(builds) == 1
        DTIDataset(dti_pickle, "locktest", split="train")
        assert len(builds) == 1, "second construction rebuilt instead of reusing the cache"

    def test_lock_is_removed_after_processing(self, dti_pickle, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        dataset = DTIDataset(dti_pickle, "locktest", split="train")
        assert not os.path.exists(dataset._process_lock())

    def test_stale_lock_is_taken_over(self, dti_pickle, tmp_path, monkeypatch):
        """A holder that died must not deadlock every later job on its leftover file."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(DTIDataset, "_wait_for_other_process", staticmethod(lambda *a, **k: None))
        root = os.path.join("data", "locktest", os.path.splitext(os.path.basename(dti_pickle))[0])
        os.makedirs(root, exist_ok=True)
        open(os.path.join(root, "processing.lock"), "w").close()

        dataset = DTIDataset(dti_pickle, "locktest", split="train")
        assert len(dataset) > 0
        assert not os.path.exists(dataset._process_lock())

    def test_lock_is_removed_when_processing_fails(self, dti_pickle, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(DTIDataset, "_build_splits", lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(RuntimeError, match="boom"):
            DTIDataset(dti_pickle, "locktest", split="train")
        root = os.path.join("data", "locktest", os.path.splitext(os.path.basename(dti_pickle))[0])
        assert not os.path.exists(os.path.join(root, "processing.lock"))
