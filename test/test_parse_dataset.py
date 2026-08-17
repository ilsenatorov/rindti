"""Regression tests for interaction-table parsing.

The bug these guard against: a single global threshold interpreted as nM silently
labelled every KIBA interaction positive, `posneg` filtering then removed everything,
and the pipeline wrote a header-only TSV without complaining.
"""

import numpy as np
import pandas as pd
import pytest
from parse_dataset import (
    MAX_VALID_AFFINITY,
    balanced_filter,
    binarize,
    check_not_degenerate,
    drop_invalid_affinities,
    posneg_filter,
)


def _frame(y):
    """Interactions over 4 drugs x 3 targets with the given Y values."""
    n = len(y)
    return pd.DataFrame(
        {
            "Drug_ID": [f"D{i % 4}" for i in range(n)],
            "Target_ID": [f"T{i % 3}" for i in range(n)],
            "Y": y,
        }
    )


class TestBinarize:
    def test_nm_lower_is_positive(self):
        """Davis-like: Kd in nM, lower means a stronger interaction."""
        out = binarize(_frame([1.0, 50.0, 100.0, 10000.0]), threshold=100, unit="nM")
        assert out["Y"].tolist() == [1, 1, 0, 0]

    def test_score_higher_is_positive(self):
        """KIBA-like: unitless score, higher means a stronger interaction."""
        out = binarize(_frame([0.0, 11.0, 12.1, 17.2]), threshold=12.1, unit="score")
        assert out["Y"].tolist() == [0, 0, 1, 1]

    def test_unknown_unit_rejected(self):
        with pytest.raises(ValueError, match="Unknown affinity unit"):
            binarize(_frame([1.0]), threshold=1, unit="pKd")

    def test_kiba_scale_under_nm_threshold_is_caught(self):
        """The original bug: KIBA values against the nM default are all one class.

        Previously this produced an empty file downstream; now it raises.
        """
        kiba_like = _frame(np.linspace(0.0, 17.2, 40))
        out = binarize(kiba_like, threshold=100, unit="nM")
        assert out["Y"].nunique() == 1, "precondition: the wrong unit collapses the labels"
        with pytest.raises(ValueError, match="single-class"):
            check_not_degenerate(out, "class", "Binarization")

    def test_kiba_scale_with_correct_unit_is_usable(self):
        kiba_like = _frame(np.linspace(0.0, 17.2, 40))
        out = binarize(kiba_like, threshold=12.1, unit="score")
        check_not_degenerate(out, "class", "Binarization")
        assert out["Y"].nunique() == 2
        assert not posneg_filter(out).empty


class TestGuards:
    def test_empty_is_rejected(self):
        empty = _frame([]).astype({"Y": float})
        with pytest.raises(ValueError, match="empty dataset"):
            check_not_degenerate(empty, "class", "Filtering")

    def test_single_class_is_rejected(self):
        with pytest.raises(ValueError, match="single-class"):
            check_not_degenerate(_frame([1, 1, 1, 1]), "class", "Filtering")

    def test_regression_skips_the_class_check(self):
        """Continuous targets legitimately have many distinct values."""
        check_not_degenerate(_frame([1.0, 2.0, 3.0]), "reg", "Binarization")

    def test_healthy_dataset_passes(self):
        check_not_degenerate(_frame([0, 1, 0, 1]), "class", "Filtering")


class TestInvalidAffinities:
    def test_drops_out_of_range(self):
        """GLASS carries negative affinities, exact zeros and values up to 1e28."""
        out = drop_invalid_affinities(_frame([-41900.0, 0.0, 1e28, 105.0, 3.98]))
        assert out["Y"].tolist() == [105.0, 3.98]

    def test_keeps_the_boundaries(self):
        out = drop_invalid_affinities(_frame([1e-3, MAX_VALID_AFFINITY]))
        assert len(out) == 2


class TestBalancedFilter:
    def test_equalises_the_classes(self):
        """`balanced` was permitted by the schema but raised ValueError."""
        out = balanced_filter(_frame([1] * 20 + [0] * 5))
        assert out["Y"].value_counts().to_dict() == {1: 5, 0: 5}

    def test_single_class_yields_empty(self):
        assert balanced_filter(_frame([1, 1, 1])).empty
