"""Tests for the training entry point's guards."""

import pytest

from rindti.cli import check_task_matches


class _FakeDataModule:
    def __init__(self, task):
        config = {"snakemake": {}}
        if task is not None:
            config["snakemake"]["parse_dataset"] = {"task": task}
        self.config = config


class TestCheckTaskMatches:
    """`parse_dataset.task` and `model.module` live in separate config files.

    Nothing related them, so a `reg` dataset trained as `class` ran
    binary_cross_entropy_with_logits against continuous affinities and reported AUROC on
    them without error.
    """

    @pytest.mark.parametrize("task", ["class", "reg"])
    def test_matching_task_passes(self, task):
        check_task_matches(_FakeDataModule(task), task)

    @pytest.mark.parametrize(("task", "module"), [("reg", "class"), ("class", "reg")])
    def test_mismatch_raises(self, task, module):
        with pytest.raises(ValueError, match="parse_dataset.task"):
            check_task_matches(_FakeDataModule(task), module)

    def test_dataset_without_recorded_task_is_allowed(self):
        """Datasets built before the task was recorded must still load."""
        check_task_matches(_FakeDataModule(None), "class")
