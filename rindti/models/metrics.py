"""Regression metrics the affinity-prediction literature reports.

torchmetrics covers MAE/MSE/Pearson/Spearman, but not these two, and the one name
collision is a trap: ``torchmetrics.ConcordanceCorrCoef`` is *Lin's concordance
correlation coefficient*, a different statistic from the Gonen-Heller **concordance
index** that the DeepDTA -> GraphDTA -> DGraphDTA -> GEFA lineage calls "CI". Using
Lin's would put the wrong number in the column meant to compare against them.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import Metric

# Cap on the number of pairwise comparisons held in memory at once. n^2 over a whole
# test split is millions of pairs, so the comparison is walked in bounded row chunks
# rather than materialised as one n x n matrix.
_MAX_PAIRS_IN_FLIGHT = 4_000_000


def _flat(preds: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Both arrive as ``(B, 1)`` from the models' ``shared_step``."""
    return preds.detach().reshape(-1).float(), target.detach().reshape(-1).float()


def _gathered(state: list[Tensor] | Tensor) -> Tensor:
    """Flatten an accumulated ``add_state`` list.

    ``dist_reduce_fx="cat"`` replaces the list with a single tensor after a
    distributed sync, so both shapes have to be handled - and an empty list, which
    happens whenever ``compute`` is reached without an ``update`` (a reset metric, or
    a validation loop that ran no batches).
    """
    if isinstance(state, Tensor):
        return state.reshape(-1)
    return torch.cat(state) if state else torch.empty(0)


class ConcordanceIndex(Metric):
    r"""Gonen-Heller concordance index.

    The fraction of comparable pairs - pairs whose true affinities differ - that the
    model ranks the right way round, with exactly-tied predictions counting a half:

    .. math::
        CI = \frac{\sum_{y_i > y_j} \mathbb{1}[\hat{y}_i > \hat{y}_j]
                   + 0.5 \cdot \mathbb{1}[\hat{y}_i = \hat{y}_j]}
                  {\sum_{y_i > y_j} 1}

    0.5 is chance, 1.0 is a perfect ranking. When no pair is comparable (every target
    identical) the statistic is undefined and this returns ``nan`` rather than 0.5, so
    a degenerate split shows up in the results table instead of looking like chance.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Accumulate a batch."""
        preds, target = _flat(preds, target)
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        """Concordance over everything accumulated so far."""
        preds, target = _gathered(self.preds), _gathered(self.target)
        return concordance_index(preds, target)


def concordance_index(preds: Tensor, target: Tensor) -> Tensor:
    """Functional form of :class:`ConcordanceIndex`, on flat 1-D tensors."""
    if preds.numel() < 2:
        return torch.tensor(float("nan"), device=preds.device)

    n = preds.numel()
    chunk = max(1, _MAX_PAIRS_IN_FLIGHT // n)
    concordant = torch.zeros((), dtype=torch.float64, device=preds.device)
    comparable = torch.zeros((), dtype=torch.float64, device=preds.device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        # Only i > j, so every unordered pair is visited exactly once.
        t_row, p_row = target[start:stop, None], preds[start:stop, None]
        t_col, p_col = target[None, :stop], preds[None, :stop]
        lower = torch.arange(start, stop, device=preds.device)[:, None] > torch.arange(stop, device=preds.device)
        greater = (t_row > t_col) & lower
        smaller = (t_row < t_col) & lower
        comparable += (greater | smaller).sum(dtype=torch.float64)
        concordant += ((greater & (p_row > p_col)) | (smaller & (p_row < p_col))).sum(dtype=torch.float64)
        # Ties in the prediction over a comparable pair are worth half a point.
        concordant += 0.5 * ((greater | smaller) & (p_row == p_col)).sum(dtype=torch.float64)

    if comparable == 0:
        return torch.tensor(float("nan"), device=preds.device)
    return (concordant / comparable).float()


class RM2(Metric):
    r"""Roy's :math:`r_m^2`, as reported alongside CI across the same lineage.

    .. math::
        r_m^2 = r^2 \left(1 - \sqrt{|r^2 - r_0^2|}\right)

    where :math:`r^2` is the squared Pearson correlation and :math:`r_0^2` the squared
    correlation of the regression through the origin. Undefined for a constant
    prediction or a constant target, where it returns ``nan``.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Accumulate a batch."""
        preds, target = _flat(preds, target)
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        """rm^2 over everything accumulated so far."""
        preds, target = _gathered(self.preds), _gathered(self.target)
        return rm2(preds, target)


def rm2(preds: Tensor, target: Tensor) -> Tensor:
    """Functional form of :class:`RM2`, on flat 1-D tensors."""
    nan = torch.tensor(float("nan"), device=preds.device)
    if preds.numel() < 2:
        return nan
    preds, target = preds.double(), target.double()

    p_centred, t_centred = preds - preds.mean(), target - target.mean()
    denom = p_centred.norm() * t_centred.norm()
    if denom == 0:  # constant predictions or constant targets
        return nan
    r2 = ((p_centred * t_centred).sum() / denom) ** 2

    # r0^2: the regression of observed on predicted, forced through the origin.
    # Slope and residuals follow the DeepDTA reference implementation, since that is
    # what the reported numbers this is compared against were produced by.
    p_ss = (preds**2).sum()
    total = (t_centred**2).sum()
    if p_ss == 0 or total == 0:
        return nan
    k = (target * preds).sum() / p_ss
    residual = ((target - k * preds) ** 2).sum()
    r02 = 1 - residual / total

    return (r2 * (1 - torch.sqrt(torch.abs(r2 - r02)))).float()


# Metrics whose *higher* value is better. Losses and error metrics are absent, so an
# unrecognised monitor defaults to minimisation - the previous hardcoded behaviour.
_MAXIMISE = frozenset(
    {
        "Accuracy",
        "AUROC",
        "AveragePrecision",
        "MatthewsCorrCoef",
        "ExplainedVariance",
        "PearsonCorrCoef",
        "SpearmanCorrCoef",
        "ConcordanceIndex",
        "RM2",
    }
)


def monitor_mode(monitor: str) -> str:
    """Whether ``monitor`` should be maximised or minimised.

    ModelCheckpoint, EarlyStopping and ReduceLROnPlateau all had this hardcoded to
    ``"min"``. That is right for ``val_loss`` but silently inverts anything else:
    checkpointing the *worst* AUROC, stopping early when concordance improves, and
    cutting the learning rate on every gain. Names are the torchmetrics class names
    that MetricCollection uses, minus the train_/val_/test_ prefix it prepends.
    """
    for prefix in ("train_", "val_", "test_"):
        if monitor.startswith(prefix):
            monitor = monitor[len(prefix) :]
            break
    return "max" if monitor in _MAXIMISE else "min"
