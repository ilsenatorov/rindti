import pandas as pd
import torch
from torchmetrics.functional import accuracy, auroc, matthews_corrcoef


class BaseBaseline:
    """Base (Parent) class for all baseline models that use simple, non-neural network based strategies to predict the outcome for a pair of a protein and a drug interaction."""

    def __init__(self, prob: bool = False, **kwargs):
        self.prob = prob

    def fit(self, train: pd.DataFrame):
        """Fit the model to the training dataframe. Has to have 'Drug_ID', 'Target_ID' and 'Y' columns. Implemented by sub-classes."""
        raise NotImplementedError()

    def predict_pair(self, prot_id: str, drug_id: str) -> float:
        """Predict the outcome for a pair of a protein and a drug. Implemented by sub-classes."""
        raise NotImplementedError()

    def test_metrics(self, test: pd.DataFrame) -> dict:
        """Calculate the metrics for the test dataframe. Available options: Accuracy, AUROC, Matthews correlation coefficient"""
        pred = self.predict(test)
        y_hat = torch.tensor(pred["pred"].values)
        y = torch.tensor(pred["Y"].values)
        return dict(
            acc=accuracy(y_hat, y, task="binary").float(),
            auc=auroc(y_hat, y, task="binary").float(),
            mcc=matthews_corrcoef(y_hat, y, num_classes=2, task="binary").float(),
        )

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        """Apply prediction to the whole test dataframe."""
        test["pred"] = test.apply(
            lambda x: self.predict_pair(x["Target_ID"], x["Drug_ID"]), axis=1
        )
        return test

    def assess_dataset(self, filename: str, train_frac: float = 0.8, n_runs: int = 10):
        """Assess the performance of the model on a dataset and output tab-separated metrics with accuracy of 3 decimal digits."""
        dataset = pd.read_csv(filename, sep="\t")
        train = dataset[dataset["split"] == "train"]
        val = dataset[dataset["split"] == "val"]
        self.fit(train)
        metrics = self.test_metrics(val)
        print(
            f"Results\tAcc : {metrics['acc']:.3}\tAUROC: {metrics['auc']:.3}\tMCC: {metrics['mcc']:.3}"
        )
