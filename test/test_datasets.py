import os

import pytest

from rindti.data import DTIDataset


@pytest.mark.snakemake
def test_dti_dataset(dti_pickle):
    """`split` defaults to train, which is the 70% the length below assumes.

    This used to read `DTIDataset(dti_pickle, "test")`, where "test" was the dropped
    `exp_name` argument rather than the split.
    """
    ds = DTIDataset(dti_pickle)
    assert len(ds) == int(25 * 0.7)
    assert "snakemake" in ds.config


@pytest.mark.snakemake
def test_cache_root_is_the_dataset_alone(dti_pickle):
    """The cache key is the pickle stem - itself the md5 of the config that built it.

    An `exp_name` level above it made every experiment rebuild a byte-identical copy.
    """
    stem = os.path.splitext(os.path.basename(dti_pickle))[0]
    assert DTIDataset(dti_pickle).root == os.path.join("data", stem)


@pytest.mark.snakemake
class TestProcessLock:
    """Concurrent jobs built from one dataset pickle used to race.

    The cache key is the directory, so submitting several model configs against one
    dataset has them all find the cache cold and all write into the same `processed/`.
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

        DTIDataset(dti_pickle, split="train")
        assert len(builds) == 1
        DTIDataset(dti_pickle, split="train")
        assert len(builds) == 1, "second construction rebuilt instead of reusing the cache"

    def test_lock_is_removed_after_processing(self, dti_pickle, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        dataset = DTIDataset(dti_pickle, split="train")
        assert not os.path.exists(dataset._process_lock())

    def test_stale_lock_is_taken_over(self, dti_pickle, tmp_path, monkeypatch):
        """A holder that died must not deadlock every later job on its leftover file."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(DTIDataset, "_wait_for_other_process", staticmethod(lambda *a, **k: None))
        root = os.path.join("data", os.path.splitext(os.path.basename(dti_pickle))[0])
        os.makedirs(root, exist_ok=True)
        open(os.path.join(root, "processing.lock"), "w").close()

        dataset = DTIDataset(dti_pickle, split="train")
        assert len(dataset) > 0
        assert not os.path.exists(dataset._process_lock())

    def test_lock_is_removed_when_processing_fails(self, dti_pickle, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(DTIDataset, "_build_splits", lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(RuntimeError, match="boom"):
            DTIDataset(dti_pickle, split="train")
        root = os.path.join("data", os.path.splitext(os.path.basename(dti_pickle))[0])
        assert not os.path.exists(os.path.join(root, "processing.lock"))
