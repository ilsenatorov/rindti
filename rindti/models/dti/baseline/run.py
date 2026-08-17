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
) -> dict:
    """Assess the performance of a baseline model on a dataset.

    Args:
        model: which baseline, one of ``max`` or ``prot_drug_max``.
        filename: split_data TSV with a ``split`` column.
        split: which held-out split to report on. Defaults to ``test``.
        n_runs: repeats, only meaningful for ``prob=True``.
        which: for ``prot_drug_max``, use ``prot``, ``drug`` or ``both`` priors.
        prob: sample predictions instead of returning the mean.
    """
    model = models[model](which=which, prob=prob)
    return model.assess_dataset(filename, split=split, n_runs=n_runs)


if __name__ == "__main__":
    from jsonargparse import CLI

    cli = CLI(run)
