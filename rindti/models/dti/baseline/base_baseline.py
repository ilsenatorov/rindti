import pandas as pd
import torch
from torchmetrics.functional import accuracy, auroc, matthews_corrcoef


class BaseBaseline:
    """Parent of all baseline models."""

    def __init__(self, prob: bool = False, **kwargs):
        self.prob = prob

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
            acc=accuracy(y_hat, y).float().item(),
            auc=auroc(y_hat, y).float().item(),
            mcc=matthews_corrcoef(y_hat, y, num_classes=2).float().item(),
        )

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        """Apply prediction to the whole test dataframe."""
        test["pred"] = test.apply(lambda x: self.predict_pair(x["Target_ID"], x["Drug_ID"]), axis=1)
        return test

    def assess_dataset(self, train: pd.DataFrame, test: pd.DataFrame):
        """Assess the performance of the model on a dataset."""
        self.fit(train)
        return self.test_metrics(test)
