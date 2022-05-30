import numpy as np
import pandas as pd
import torch
from torchmetrics.functional import accuracy, auroc, matthews_corrcoef


class BaseBaseline:
    """Parent of all baseline models."""

    def __init__(self, **kwargs):
        self.prob = kwargs["prob"]

    def fit(self, train: pd.DataFrame):
        """Fit the model to the training dataframe. Has to have 'Drug_ID', 'Target_ID' and 'Y' columns."""
        raise NotImplementedError()

    def predict_pair(self, prot_id: str, drug_id: str) -> float:
        """Predict the outcome for a pair of a protein and a drug."""
        raise NotImplementedError()

    def test_metrics(self, test: pd.DataFrame) -> dict:
        """Calculate the metrics for the test dataframe."""
        pred = self.predict(test)
        y_hat = torch.tensor(pred["pred"].values)
        y = torch.tensor(pred["Y"].values)
        return dict(
            acc=accuracy(y_hat, y).float(),
            auc=auroc(y_hat, y).float(),
            mcc=matthews_corrcoef(y_hat, y, num_classes=2).float(),
        )

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        """Apply prediction to the whole test dataframe."""
        test["pred"] = test.apply(lambda x: self.predict_pair(x["Target_ID"], x["Drug_ID"]), axis=1)
        return test

    def assess_dataset(self, filename: str, train_frac: float = 0.8, n_runs: int = 10):
        """Assess the performance of the model on a dataset."""
        dataset = pd.read_csv(filename, sep="\t")
        if "split" in dataset.columns:
            train = dataset[dataset["split"] == "train"]
            val = dataset[dataset["split"] == "val"]
            self.fit(train)
            metrics = self.test_metrics(val)
            print(f"Results\tAcc : {metrics['acc']:.3}\tAUROC: {metrics['auc']:.3}\tMCC: {metrics['mcc']:.3}")
        else:
            averages = {"acc": [], "mcc": [], "auc": []}
            for i in range(n_runs):
                train = dataset.sample(frac=train_frac)
                test = dataset.drop(train.index)
                self.fit(train)
                metrics = self.test_metrics(test)
                averages["acc"].append(metrics["acc"])
                averages["mcc"].append(metrics["mcc"])
                averages["auc"].append(metrics["auc"])
                print(f"Run {i}\tAcc : {metrics['acc']:.3}\tAUROC: {metrics['auc']:.3}\tMCC: {metrics['mcc']:.3}")
            print(
                f"Average\tAcc : {np.mean(averages['acc']):.3}\tAUROC: {np.mean(averages['auc']):.3}\tMCC: {np.mean(averages['mcc']):.3}"
            )
