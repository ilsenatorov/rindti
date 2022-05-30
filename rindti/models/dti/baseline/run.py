import pandas as pd
from max_likelihood import Max
from prot_drug_max_likelihood import ProtDrugMax

models = {"max": Max, "prot_drug_max": ProtDrugMax}


def run(model: str, filename: str, train_frac: float = 0.8, n_runs: int = 10):
    """Assess the performance of the model on a dataset."""
    model = models[model]()
    model.assess_dataset(filename, train_frac, n_runs)


if __name__ == "__main__":
    from jsonargparse import CLI

    cli = CLI(run)
