import json

from ...metrics import ORACLE, best_baseline, boss
from .max_likelihood import Max
from .prot_drug_max_likelihood import ProtDrugMax

models = {"max": Max, "prot_drug_max": ProtDrugMax}


def run(
    model: str,
    filename: str,
    split: str = "test",
    n_runs: int = 10,
    which: str = "both",
    prob: bool = False,
    task: str = "class",
) -> dict:
    """Assess the performance of a baseline model on a dataset.

    Args:
        model: which baseline, one of ``max`` or ``prot_drug_max``.
        filename: split_data TSV with a ``split`` column.
        split: which held-out split to report on. Defaults to ``test``.
        n_runs: repeats, only meaningful for ``prob=True``.
        which: for ``prot_drug_max``, use ``prot``, ``drug`` or ``both`` priors.
        prob: sample predictions instead of returning the mean.
        task: ``class`` or ``reg``, as ``parse_dataset.task``; picks the metric set.
    """
    model = models[model](which=which, prob=prob, task=task)
    return model.assess_dataset(filename, split=split, n_runs=n_runs)


def report(filename: str, task: str = "class", split: str = "test") -> dict:
    """Every deterministic baseline on one split, as ``{name: {metric: mean}}``.

    What the pipeline writes next to each split: the floor a trained model has to beat
    on exactly that split. ``std`` is dropped since it is always 0 without ``prob``.
    """
    variants = {"max": ("max", "both")} | {
        f"prot_drug_max_{w}": ("prot_drug_max", w) for w in ("both", "prot", "drug")
    }
    return {
        name: {k: mean for k, (mean, _) in run(model, filename, split=split, which=which, task=task).items()}
        for name, (model, which) in variants.items()
    }


def score(model_metrics: dict, baselines: str | dict) -> dict:
    """BOSS of a model against the label-only floor, as ``{metric: {...}}``.

    Turns a metric a reader cannot calibrate ("AUROC 0.89") into one they can ("the model
    closes 28% of the gap the labels alone leave"). ``baselines`` is the JSON the pipeline
    writes beside each split, by path or already loaded; ``model_metrics`` is a test-metric
    dict keyed the same short way (``auc``, ``mcc``, ``ci``, ...).

    Only metrics present on both sides and holding a finite best value are scored, so
    passing a classification model against a regression floor yields an empty result
    rather than nonsense.
    """
    if isinstance(baselines, str):
        with open(baselines) as handle:
            baselines = json.load(handle)
    scored = {}
    for metric, value in model_metrics.items():
        if metric not in ORACLE:
            continue
        try:
            base = best_baseline(metric, baselines)
        except KeyError:
            continue
        scored[metric] = {"model": value, "baseline": base, "boss": boss(metric, value, base)}
    return scored


if __name__ == "__main__":
    from jsonargparse import CLI

    cli = CLI([run, report, score])
