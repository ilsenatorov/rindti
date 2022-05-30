import pandas as pd
import torch
from torchmetrics.functional import accuracy, auroc, matthews_corrcoef


class BaseBaseline:
    """Parent of all baseline models."""

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

    def assess_dataset(self, filename: str, train_frac: float = 0.8, n_runs: int = 10):
        """Assess the performance of the model on a dataset."""
        dataset = pd.read_csv(filename, sep="\t")
        for i in range(n_runs):
            train = dataset.sample(frac=train_frac)
            test = dataset.drop(train.index)
            self.fit(train)
            metrics = self.test_metrics(test)
            print(f"Run {i}\tAcc : {metrics['acc']:.3}\tAUROC: {metrics['auc']:.3}\tMCC: {metrics['mcc']:.3}")
