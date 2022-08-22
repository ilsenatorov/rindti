from jsonargparse import CLI

from .max_likelihood import Max
from .prot_drug_max_likelihood import ProtDrugMax

models = {"max": Max, "prot_drug_max": ProtDrugMax}


def run(
    model: str,
    filename: str,
    train_frac: float = 0.8,
    n_runs: int = 10,
    which: str = "both",
    prob: bool = False,
):
    """Assess the performance of the model on a dataset."""
    model = models[model](which=which, prob=prob)
    model.assess_dataset(filename, train_frac, n_runs)


if __name__ == "__main__":
    cli = CLI(run)
