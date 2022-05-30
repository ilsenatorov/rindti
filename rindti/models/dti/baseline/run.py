from max_likelihood import Max
from prot_drug_max_likelihood import ProtDrugMax
from single_max import SingleMax

models = {"max": Max, "prot_drug_max": ProtDrugMax, "single_max": SingleMax}


def run(model: str, filename: str, train_frac: float = 0.8, n_runs: int = 10, prot: bool = True, prob: bool = False):
    """Assess the performance of the model on a dataset."""
    kwargs = {
        "prot": prot,
        "prob": prob,
    }
    model = models[model](**kwargs)
    model.assess_dataset(filename, train_frac, n_runs)


if __name__ == "__main__":
    from jsonargparse import CLI

    cli = CLI(run)
