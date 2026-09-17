"""Tests for the dataset description tool.

Its two checks are the interesting part: a cold split that leaks an entity across the
train/test boundary, and a split so small that `split_groups` sent everything to train.
Both used to be invisible until a training run behaved strangely.
"""

import pandas as pd
import pytest
import torch
from dataset_stats import _labels, _sizes, _splits, collect, render


def _graph_frame(sizes, edges=True):
    """An entity table shaped like the one `prepare_all` pickles."""
    rows = []
    for n in sizes:
        graph = {"x": torch.zeros(n, 4)}
        if edges:
            # Both directions stored, as the pipeline writes them.
            graph["edge_index"] = torch.zeros(2, n * 4, dtype=torch.long)
        rows.append(graph)
    return pd.DataFrame({"data": rows})


def _interactions(rows):
    return pd.DataFrame(rows, columns=["prot_id", "drug_id", "label", "split"])


class TestSizes:
    def test_node_summary(self):
        stats = _sizes(_graph_frame([10, 20, 30]), "prot")
        assert stats["prot_count"] == 3
        assert stats["prot_nodes_mean"] == 20
        assert stats["prot_nodes_min"] == 10
        assert stats["prot_nodes_max"] == 30

    def test_edges_are_halved_for_undirected_graphs(self):
        stats = _sizes(_graph_frame([10]), "prot")
        assert stats["prot_edges_mean"] == 20  # 40 stored directions
        assert stats["prot_degree_mean"] == 4

    def test_vector_features_have_no_edges(self):
        """The ESM featurisation is one pooled vector per protein, with no graph."""
        stats = _sizes(_graph_frame([1, 1], edges=False), "prot")
        assert "prot_edges_mean" not in stats
        assert stats["prot_count"] == 2


class TestSplits:
    def test_cold_split_leakage_is_reported(self):
        frame = _interactions(
            [("P1", "D1", 1, "train"), ("P1", "D2", 0, "test"), ("P2", "D3", 1, "val")],
        )
        stats = _splits(frame, "target")
        assert stats["leakage_column"] == "prot_id"
        assert stats["leakage_overlap"] == 1

    def test_clean_cold_split_has_no_overlap(self):
        frame = _interactions(
            [("P1", "D1", 1, "train"), ("P2", "D2", 0, "test"), ("P3", "D3", 1, "val")],
        )
        assert _splits(frame, "cluster_target")["leakage_overlap"] == 0

    def test_random_split_is_not_checked_for_overlap(self):
        """Entities on both sides is what `random` means; it is not a defect."""
        frame = _interactions([("P1", "D1", 1, "train"), ("P1", "D2", 0, "test")])
        assert "leakage_overlap" not in _splits(frame, "random")

    def test_empty_splits_are_reported(self):
        """split_groups takes int(10 * frac) per bin of ten, so a handful of groups
        all land in train and the val/test sets come back empty."""
        frame = _interactions([("P1", "D1", 1, "train"), ("P2", "D2", 0, "train")])
        assert _splits(frame, "target")["empty_splits"] == ["val", "test"]

    def test_healthy_split_reports_none_empty(self):
        frame = _interactions(
            [("P1", "D1", 1, "train"), ("P2", "D2", 0, "val"), ("P3", "D3", 1, "test")],
        )
        assert _splits(frame, "random")["empty_splits"] == []


class TestLabels:
    def test_classification_reports_positive_rate(self):
        frame = _interactions([("P1", "D1", 1, "train"), ("P2", "D2", 0, "train")])
        config = {"parse_dataset": {"task": "class", "unit": "nM", "threshold": 100}}
        stats = _labels(frame, config)
        assert stats["positives"] == 1
        assert stats["positive_rate"] == 0.5

    def test_regression_reports_the_value_distribution(self):
        frame = _interactions([("P1", "D1", 5.0, "train"), ("P2", "D2", 7.0, "train")])
        config = {"parse_dataset": {"task": "reg", "unit": "nM", "log": True}}
        stats = _labels(frame, config)
        assert stats["label_mean"] == 6.0
        assert (stats["label_min"], stats["label_max"]) == (5.0, 7.0)
        assert "positive_rate" not in stats


@pytest.mark.snakemake
class TestAgainstABuiltDataset:
    """End to end on the dataset the pipeline actually produces."""

    def test_collect_and_render(self, dti_pickle):
        stats = collect(dti_pickle)
        assert stats["final_interactions"] > 0
        assert stats["final_targets"] > 0
        assert stats["prot_count"] == stats["final_targets"]
        assert stats["drug_count"] == stats["final_drugs"]
        # Every interaction is in exactly one split.
        assert sum(stats[f"{s}_interactions"] for s in ("train", "val", "test")) == stats["final_interactions"]
        assert "Attrition" in render(stats)
