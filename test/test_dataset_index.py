"""Tests for the pickle -> config resolver.

`prepare_all` names its output by a hash of the whole snakemake config, so the only way
to tell twenty-six built datasets apart is to read the config back out of them. Two
behaviours matter enough to pin down: that the table collapses to the axes that actually
vary, and that a mistyped or unsplit filter fails loudly rather than quietly returning
every dataset - which would put the wrong number of jobs in a queue file.
"""

import pickle
import subprocess
import sys
from pathlib import Path

import pytest
from dataset_index import _dataset_name, _dig, describe, exp_name, varying

SCRIPT = Path("workflow/scripts/dataset_index.py")


def _config(source="datasets/davis/resources", split="random", **over):
    config = {
        "source": source,
        "prots": {
            "structs": {"method": "whole"},
            "features": {"method": "distance", "node_feats": "label", "edge_feats": "none"},
        },
        "drugs": {"node_feats": "label"},
        "split_data": {"method": split},
        "parse_dataset": {"task": "class", "filtering": "posneg", "sampling": "none", "unit": "nM"},
    }
    for path, value in over.items():
        node = config
        keys = path.split(".")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    return config


def _pickle(tmp_path, name, config):
    """A file shaped like a prepare_all output: only the `config` key is read."""
    path = tmp_path / name
    with open(path, "wb") as handle:
        pickle.dump({"config": config, "data": [], "prots": None, "drugs": None}, handle)
    return path


class TestDig:
    def test_reads_a_dotted_path(self):
        assert _dig(_config(), "prots.features.node_feats") == "label"

    def test_missing_level_is_none_not_an_error(self):
        # A dataset built before an axis existed simply has no value for it.
        assert _dig(_config(), "prots.nonexistent.thing") is None


class TestDatasetName:
    @pytest.mark.parametrize(
        "source,expected",
        [
            ("datasets/davis/resources", "davis"),
            ("datasets/bindingdb_kd/resources/", "bindingdb_kd"),
            ("test/test_data/resources", "test_data"),
        ],
    )
    def test_names_the_dataset_directory(self, source, expected):
        assert _dataset_name(source) == expected

    def test_no_source_does_not_raise(self):
        assert _dataset_name(None) == "unknown"


class TestDescribe:
    def test_reads_the_config_back_out(self, tmp_path):
        path = _pickle(tmp_path, "a.pkl", _config(split="cluster_target"))
        row = describe(str(path))
        assert row["dataset"] == "davis"
        assert row["split"] == "cluster_target"
        assert row["drug_node"] == "label"


class TestVarying:
    def test_keeps_only_the_axes_that_differ(self):
        rows = [describe_like(split="random"), describe_like(split="drug")]
        columns = varying(rows)
        assert columns == ["dataset", "split"]

    def test_surfaces_an_axis_once_it_moves(self):
        rows = [describe_like(drug_node="label"), describe_like(drug_node="rich")]
        assert "drug_node" in varying(rows)

    def test_dataset_and_split_survive_even_when_constant(self):
        # They name the run, so they stay in the table regardless.
        rows = [describe_like(), describe_like()]
        assert varying(rows) == ["dataset", "split"]


def describe_like(**over):
    row = {
        "dataset": "davis",
        "split": "random",
        "task": "class",
        "filtering": "posneg",
        "sampling": "none",
        "unit": "nM",
        "structs": "whole",
        "protfeat": "distance",
        "prot_node": "label",
        "prot_edge": "none",
        "drug_node": "label",
    }
    row.update(over)
    return row


class TestExpName:
    def test_names_by_dataset_and_split(self):
        assert exp_name(describe_like(), ["dataset", "split"]) == "davis_random"

    def test_carries_a_moved_axis(self):
        name = exp_name(describe_like(drug_node="rich"), ["dataset", "split", "drug_node"])
        assert name == "davis_random_drug_node=rich"

    def test_contains_no_comma_or_space(self):
        # The name goes into a column of a comma-separated condor queue file.
        name = exp_name(describe_like(drug_node="rich"), ["dataset", "split", "drug_node"])
        assert "," not in name and " " not in name


def _run(tmp_path, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "*.pkl"), *args],
        capture_output=True,
        text=True,
    )


class TestCli:
    @pytest.fixture()
    def built(self, tmp_path):
        _pickle(tmp_path, "a.pkl", _config(split="random"))
        _pickle(tmp_path, "b.pkl", _config(split="drug"))
        _pickle(tmp_path, "c.pkl", _config(split="random", **{"drugs.node_feats": "rich"}))
        return tmp_path

    def test_lists_every_pickle(self, built):
        result = _run(built)
        assert result.returncode == 0
        assert result.stdout.count(".pkl") == 3

    def test_where_filters(self, built):
        result = _run(built, "--where", "split=random")
        assert result.returncode == 0
        assert result.stdout.count(".pkl") == 2

    def test_where_clauses_are_anded(self, built):
        result = _run(built, "--where", "split=random", "--where", "drug_node=rich")
        assert result.stdout.count(".pkl") == 1

    def test_unknown_column_is_rejected(self, built):
        result = _run(built, "--where", "bogus=x")
        assert result.returncode == 1
        assert "no such column" in result.stdout + result.stderr

    def test_filtering_everything_out_is_an_error(self, built):
        # Better than emitting an empty queue file and reporting "0 jobs submitted"
        # several minutes later.
        result = _run(built, "--where", "split=nope")
        assert result.returncode == 1

    def test_unsplit_option_string_is_rejected(self, built):
        """The zsh trap: an expansion that was never word-split.

        argparse treats any token containing a space as positional, so this used to
        arrive as a glob pattern, match nothing, and leave the filter silently
        unapplied - generating a queue file with every dataset in it.
        """
        result = _run(built, "--where split=random --where drug_node=rich")
        assert result.returncode == 1
        assert "parsed as a path" in result.stdout + result.stderr

    def test_emit_runs_produces_four_columns(self, built):
        result = _run(built, "--emit-runs", "config/dti/base.yaml", "--runs", "3")
        assert result.returncode == 0
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        assert len(lines) == 3
        for line in lines:
            assert len(line.split(",")) == 4
            assert line.endswith("--set runs=3")

    def test_emit_runs_names_are_unique(self, built):
        """Two jobs sharing an exp_name race on next_version() and overwrite each other."""
        result = _run(built, "--emit-runs", "config/dti/base.yaml")
        names = [line.split(",")[2].strip() for line in result.stdout.splitlines() if line.strip()]
        assert len(set(names)) == len(names)

    def test_extra_containing_a_comma_is_rejected(self, built):
        result = _run(built, "--emit-runs", "config/dti/base.yaml", "--extra", "--set a=[x, y]")
        assert result.returncode == 1
        assert "comma" in result.stdout + result.stderr

    def test_table_is_written(self, built, tmp_path):
        out = tmp_path / "index.tsv"
        result = _run(built, "--table", str(out))
        assert result.returncode == 0
        header = out.read_text().splitlines()[0].split("\t")
        assert header[0] == "path" and "split" in header
