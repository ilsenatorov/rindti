"""Collect metrics from TensorBoard runs into a tidy table.

``rindti-train`` writes one TensorBoard run per seed under
``tb_logs/dti_<exp>/<dataset>/version_<n>/<seed>/``. Reading numbers back out of
those event files by hand does not scale past a handful of runs, so this turns a
log tree into one row per (run, metric) that pandas can group and aggregate.
"""

from pathlib import Path

import pandas as pd
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Metrics are logged per epoch; "last" is the final epoch, which for an
# early-stopped run is the one that triggered the stop, not the best epoch.
REDUCTIONS = ("last", "best_max", "best_min")


# Never useful as an ablation column: run-specific, or constant across a sweep.
UNINFORMATIVE_HPARAMS = {"seed", "git_hash", "datamodule,exp_name"}


def _flatten(config: dict, prefix: str = "") -> dict:
    """Flatten nested hparams to comma-joined keys."""
    flat = {}
    for key, value in config.items():
        path = f"{prefix},{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _hparams(run_dir: Path) -> dict:
    """Config Lightning saved in the run directory, flattened.

    This is what makes a sweep analysable: it records which settings produced
    each row, including the dataset dims the snakemake config contributed.
    """
    path = run_dir / "hparams.yaml"
    if not path.exists():
        return {}
    try:
        with open(path) as file:
            return _flatten(yaml.safe_load(file) or {})
    except yaml.YAMLError as err:
        # One unreadable config should cost that run its columns, not abort the
        # collection of an entire sweep.
        print(f"Skipping unreadable {path}: {err.__class__.__name__}")
        return {}


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


def collect(logdir: str = "tb_logs", reduction: str = "last", hparams: bool = True) -> pd.DataFrame:
    """Walk a TensorBoard log tree and return one row per run and metric.

    Args:
        logdir: root of the log tree, i.e. what you passed to ``tensorboard --logdir``.
        reduction: how to collapse a metric's epoch series to a single number -
            ``last``, ``best_max`` (e.g. AUROC) or ``best_min`` (e.g. loss).
        hparams: attach the settings each run used, keeping only those that vary
            across the tree - the columns an ablation is grouped by.

    Returns:
        A frame with one row per run and metric, plus a column per varying
        hyperparameter. Empty if no event files were found.
    """
    root = Path(logdir)
    rows = []
    # Lightning writes a separate event file per fit/test stage, so group by the
    # run directory - otherwise one run is counted as several.
    run_dirs = sorted({f.parent for f in root.rglob("events.out.tfevents*")})
    for run_dir in run_dirs:
        # tb_logs/dti_<experiment>/<dataset>/version_<n>/<seed>/
        parts = run_dir.relative_to(root).parts
        experiment, dataset, version, seed = (list(parts) + [None] * 4)[:4]
        config = _hparams(run_dir) if hparams else {}
        merged = {}
        for event_file in sorted(run_dir.glob("events.out.tfevents*")):
            merged.update(_scalars(event_file))
        for metric, values in merged.items():
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
                    **{f"hp:{k}": v for k, v in config.items()},
                }
            )
    frame = pd.DataFrame(rows)
    return _drop_constant_hparams(frame) if hparams else frame


def _drop_constant_hparams(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only the hyperparameter columns that actually vary.

    A full config has hundreds of entries; an ablation table is only readable if
    it shows the handful that differ between runs.
    """
    if frame.empty:
        return frame
    hp_cols = [c for c in frame.columns if c.startswith("hp:")]
    keep = [
        c for c in hp_cols if frame[c].astype(str).nunique(dropna=False) > 1 and c[3:] not in UNINFORMATIVE_HPARAMS
    ]
    return frame.drop(columns=[c for c in hp_cols if c not in keep])


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate over seeds, grouping by every varying hyperparameter."""
    if results.empty:
        return results
    keys = ["experiment", "dataset", "version", "metric"]
    keys += [c for c in results.columns if c.startswith("hp:")]
    grouped = results.groupby(keys, dropna=False)["value"]
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
