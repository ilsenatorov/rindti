import pandas as pd

from .prot_drug_max_likelihood import ProtDrugMax


def run(
    filename: str,
    which: str = "both",
    prob: bool = False,
):
    """Assess the performance of the model on a dataset."""
    model = ProtDrugMax(which=which, prob=prob)
    data = pd.read_csv(filename, sep="\t")
    train = data[data["split"] == "train"]
    test = data[data["split"] == "test"]
    metrics = model.assess_dataset(train, test)
    print(f"Results\tAcc : {metrics['acc']:.3}\tAUROC: {metrics['auc']:.3}\tMCC: {metrics['mcc']:.3}")


if __name__ == "__main__":
    from jsonargparse import CLI

    cli = CLI(run)
