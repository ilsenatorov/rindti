"""Tests for the regression metrics the affinity literature reports.

Neither had any coverage before: `_set_reg_metrics` was assembled entirely from
torchmetrics, and torchmetrics' `ConcordanceCorrCoef` is a *different statistic*
(Lin's CCC) from the concordance index used here.
"""

import math

import pytest
import torch

from rindti.models.metrics import (
    RM2,
    ConcordanceIndex,
    concordance_index,
    monitor_mode,
    rm2,
)


def ci(preds, target) -> float:
    return concordance_index(torch.tensor(preds).float(), torch.tensor(target).float()).item()


class TestConcordanceIndex:
    """Hand-computable cases. With n targets all distinct there are n(n-1)/2 pairs."""

    def test_perfect_ranking(self):
        assert ci([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) == 1.0

    def test_perfectly_inverted_ranking(self):
        assert ci([4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]) == 0.0

    def test_monotone_but_not_linear_is_still_perfect(self):
        """CI is rank-based: only the ordering matters, not the scale."""
        assert ci([0.1, 100.0, 1e6], [1.0, 2.0, 3.0]) == 1.0

    def test_one_discordant_pair(self):
        """3 targets -> 3 pairs. Swapping the top two makes exactly one discordant."""
        assert ci([1.0, 3.0, 2.0], [1.0, 2.0, 3.0]) == pytest.approx(2 / 3)

    def test_all_predictions_tied_is_chance(self):
        """Every comparable pair scores 0.5, so the whole statistic is 0.5."""
        assert ci([7.0, 7.0, 7.0, 7.0], [1.0, 2.0, 3.0, 4.0]) == 0.5

    def test_single_tied_prediction_pair(self):
        """3 pairs: (1,2) tied -> 0.5, (1,3) and (2,3) concordant -> 2.5/3."""
        assert ci([1.0, 1.0, 2.0], [1.0, 2.0, 3.0]) == pytest.approx(2.5 / 3)

    def test_ties_in_the_target_are_not_comparable(self):
        """Targets 1,1,2: the (1,1) pair is dropped, the other two are concordant."""
        assert ci([5.0, 9.0, 10.0], [1.0, 1.0, 2.0]) == 1.0

    def test_degenerate_target_is_nan_not_chance(self):
        """No comparable pair at all. NaN makes a degenerate split visible; 0.5
        would look like an honest chance-level result."""
        assert math.isnan(ci([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]))

    def test_single_sample_is_nan(self):
        assert math.isnan(ci([1.0], [1.0]))

    def test_chunking_does_not_change_the_answer(self):
        """The pairwise walk is chunked to bound memory; the result must not depend
        on where the chunk boundaries fall."""
        from rindti.models import metrics

        torch.manual_seed(0)
        target = torch.randn(500)
        preds = target + 0.5 * torch.randn(500)
        unchunked = concordance_index(preds, target).item()
        original = metrics._MAX_PAIRS_IN_FLIGHT
        try:
            metrics._MAX_PAIRS_IN_FLIGHT = 500  # one row at a time
            assert concordance_index(preds, target).item() == pytest.approx(unchunked)
        finally:
            metrics._MAX_PAIRS_IN_FLIGHT = original


class TestConcordanceIndexMetric:
    """The torchmetrics wrapper, which is what MetricCollection actually holds."""

    def test_accumulates_across_batches(self):
        """Shapes are (B, 1), as `shared_step` produces them."""
        metric = ConcordanceIndex()
        target = torch.tensor([1.0, 2.0, 3.0, 4.0])
        metric.update(target[:2].unsqueeze(1), target[:2].unsqueeze(1))
        metric.update(target[2:].unsqueeze(1), target[2:].unsqueeze(1))
        assert metric.compute().item() == 1.0

    def test_is_higher_is_better(self):
        """MetricCollection and the checkpoint-mode lookup both rely on this."""
        assert ConcordanceIndex.higher_is_better is True

    def test_reset_clears_state(self):
        metric = ConcordanceIndex()
        metric.update(torch.tensor([[1.0], [2.0]]), torch.tensor([[1.0], [2.0]]))
        metric.reset()
        assert math.isnan(metric.compute().item())


class TestRM2:
    def test_perfect_prediction(self):
        preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert rm2(preds, preds).item() == pytest.approx(1.0)

    def test_matches_the_reference_formula(self):
        """Checked against the DeepDTA reference implementation's arithmetic."""
        preds = torch.tensor([1.0, 3.0, 2.0, 5.0]).double()
        target = torch.tensor([1.0, 2.0, 3.0, 4.0]).double()
        p_c, t_c = preds - preds.mean(), target - target.mean()
        r2 = ((p_c * t_c).sum() / (p_c.norm() * t_c.norm())) ** 2
        k = (target * preds).sum() / (preds**2).sum()
        r02 = 1 - ((target - k * preds) ** 2).sum() / (t_c**2).sum()
        expected = r2 * (1 - torch.sqrt(torch.abs(r2 - r02)))
        assert rm2(preds.float(), target.float()).item() == pytest.approx(expected.item(), rel=1e-5)

    def test_constant_prediction_is_nan(self):
        assert math.isnan(rm2(torch.ones(4), torch.tensor([1.0, 2.0, 3.0, 4.0])).item())

    def test_constant_target_is_nan(self):
        assert math.isnan(rm2(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.ones(4)).item())

    def test_metric_wrapper(self):
        metric = RM2()
        preds = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        metric.update(preds, preds)
        assert metric.compute().item() == pytest.approx(1.0)
        assert RM2.higher_is_better is True


class TestMonitorMode:
    """Checkpointing, early stopping and the LR scheduler all hardcoded "min"."""

    @pytest.mark.parametrize(
        "monitor,expected",
        [
            ("val_loss", "min"),
            ("train_loss", "min"),
            ("val_MeanSquaredError", "min"),
            ("val_MeanAbsoluteError", "min"),
            ("val_AUROC", "max"),
            ("test_ConcordanceIndex", "max"),
            ("val_RM2", "max"),
            ("val_MatthewsCorrCoef", "max"),
        ],
    )
    def test_mode(self, monitor, expected):
        assert monitor_mode(monitor) == expected

    def test_unknown_metric_falls_back_to_min(self):
        """The previous behaviour, so nothing silently changes for an unlisted name."""
        assert monitor_mode("val_SomethingNew") == "min"
