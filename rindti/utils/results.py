"""Collect metrics from TensorBoard runs into a tidy table.

``rindti-train`` writes one TensorBoard run per seed under
``tb_logs/dti_<exp>/<dataset>/version_<n>/<seed>/``. Reading numbers back out of
those event files by hand does not scale past a handful of runs, so this turns a
log tree into one row per (run, metric) that pandas can group and aggregate.
"""

from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Metrics are logged per epoch; "last" is the final epoch, which for an
# early-stopped run is the one that triggered the stop, not the best epoch.
REDUCTIONS = ("last", "best_max", "best_min")


def _scalars(event_file: Path) -> dict:
    """All scalar tags in one event file, mapped to their value series."""
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    return {tag: [e.value for e in accumulator.Scalars(tag)] for tag in accumulator.Tags()["scalars"]}


def _reduce(values: list, how: str) -> float:
    if not values:
        return float("nan")
    if how == "last":
        return values[-1]
    if how == "best_max":
        return max(values)
    if how == "best_min":
        return min(values)
    raise ValueError(f"Unknown reduction {how!r}, expected one of {REDUCTIONS}")


def collect(logdir: str = "tb_logs", reduction: str = "last") -> pd.DataFrame:
    """Walk a TensorBoard log tree and return one row per run and metric.

    Args:
        logdir: root of the log tree, i.e. what you passed to ``tensorboard --logdir``.
        reduction: how to collapse a metric's epoch series to a single number -
            ``last``, ``best_max`` (e.g. AUROC) or ``best_min`` (e.g. loss).

    Returns:
        A frame with columns ``experiment, dataset, version, seed, metric, value``.
        Empty if no event files were found.
    """
    root = Path(logdir)
    rows = []
    for event_file in sorted(root.rglob("events.out.tfevents*")):
        # tb_logs/dti_<experiment>/<dataset>/version_<n>/<seed>/events.out.tfevents.*
        parts = event_file.relative_to(root).parts[:-1]
        experiment, dataset, version, seed = (list(parts) + [None] * 4)[:4]
        for metric, values in _scalars(event_file).items():
            if metric == "epoch":
                continue
            rows.append(
                {
                    "experiment": experiment,
                    "dataset": dataset,
                    "version": version,
                    "seed": seed,
                    "metric": metric,
                    "value": _reduce(values, reduction),
                    "epochs": len(values),
                }
            )
    return pd.DataFrame(rows, columns=["experiment", "dataset", "version", "seed", "metric", "value", "epochs"])


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate over seeds: mean, std and n per experiment/dataset/metric."""
    if results.empty:
        return results
    grouped = results.groupby(["experiment", "dataset", "version", "metric"])["value"]
    return grouped.agg(mean="mean", std="std", n="count").reset_index()


def main(
    logdir: str = "tb_logs",
    output: str = "results.csv",
    reduction: str = "last",
    summary: bool = False,
) -> None:
    """Write collected TensorBoard metrics to a CSV.

    Args:
        logdir: root of the TensorBoard log tree.
        output: where to write the CSV.
        reduction: ``last``, ``best_max`` or ``best_min``.
        summary: aggregate over seeds into mean/std instead of one row per run.
    """
    results = collect(logdir, reduction=reduction)
    if results.empty:
        print(f"No TensorBoard event files found under {logdir}/")
        return
    if summary:
        results = summarise(results)
    results.to_csv(output, index=False)
    print(f"Wrote {len(results)} rows to {output}")
    print(results.to_string(index=False, max_rows=25))


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
