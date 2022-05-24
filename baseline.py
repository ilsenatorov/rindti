import argparse

import torch
from tqdm import tqdm
from pytorch_lightning import seed_everything

from rindti.data import DTIDataModule
from torchmetrics import (
    AUROC,
    Accuracy,
    MatthewsCorrCoef,
)

from rindti.utils import read_config, get_git_hash


def init(**kwargs):
    seed_everything(kwargs["seed"])
    datamodule = DTIDataModule(**kwargs["datamodule"])
    datamodule.setup()
    datamodule.update_config(kwargs)

    run_baseline(datamodule.train_dataloader(), datamodule.val_dataloader(), "majority", **kwargs)


def run_baseline(train_dataloader, val_dataloader, mode, **kwargs):
    """
    mode: Mode of the baseline. can be either
        * majority: Just return the majority class
    """
    if mode == "majority":
        model = get_majority_model(train_dataloader)
    else:
        raise ValueError("Unknown mode for model!")

    preds, labels = [torch.tensor(x) for x in zip(*[(model(x), x["label"].item()) for x in tqdm(val_dataloader)])]
    print(f"Baseline mode: {mode}:\n"
          f"\tMCC  : {MatthewsCorrCoef(num_classes=kwargs.get('num_classes', 2))(preds, labels)}\n"
          f"\tAcc  : {Accuracy(num_classes=kwargs.get('num_classes', 2))(preds, labels)}\n"
          f"\tAUROC: {AUROC()(0.005 + preds * 0.99, labels)}")

    return model


def get_majority_model(dataloader):
    classes = {}
    for sample in tqdm(dataloader):
        if sample["label"].item() not in classes:
            classes[sample["label"].item()] = 0
        classes[sample["label"].item()] += 1

    max_class = max(classes.items(), key=lambda y: y[1])[0]
    print(f"Most often seen class: {max_class}: {classes[max_class]}-times")
    return lambda x: max_class


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Model Trainer")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    orig_config = read_config(args.config)
    orig_config["git_hash"] = get_git_hash()  # to know the version of the code
    init(**orig_config)
