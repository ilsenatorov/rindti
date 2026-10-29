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


if __name__ == "__main__":
    from jsonargparse import CLI

    cli = CLI(run)
