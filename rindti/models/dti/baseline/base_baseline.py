import numpy as np
import pandas as pd
import torch
from torchmetrics.functional import (
    accuracy,
    auroc,
    average_precision,
    explained_variance,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    pearson_corrcoef,
    spearman_corrcoef,
)

from ...metrics import concordance_index, rm2


class BaseBaseline:
    """Parent of all baseline models."""

    def __init__(self, prob: bool = False, task: str = "class", **kwargs):
        if task not in ("class", "reg"):
            raise ValueError(f"Unknown task {task!r}, expected 'class' or 'reg'")
        self.prob = prob
        self.task = task

    def fit(self, train: pd.DataFrame):
        """Fit the model to the training dataframe. Has to have 'Drug_ID', 'Target_ID' and 'Y' columns."""
        raise NotImplementedError()

    def predict_pair(self, prot_id: str, drug_id: str) -> float:
        """Predict the outcome for a pair of a protein and a drug."""
        raise NotImplementedError()

    def test_metrics(self, test: pd.DataFrame) -> dict:
        """Calculate the metrics for the test dataframe.

        Regression reports the same set ``RegressionModel`` logs, so a baseline row sits
        directly under a model row. A constant predictor (``Max``) has no correlation, so
        pearson, spearman and rm2 come out NaN and CI 0.5 - that is the honest answer.
        """
        pred = self.predict(test)
        y_hat = torch.tensor(pred["pred"].values, dtype=torch.float)
        y = torch.tensor(pred["Y"].values)
        if self.task == "reg":
            y = y.float()
            return dict(
                mae=mean_absolute_error(y_hat, y).item(),
                mse=mean_squared_error(y_hat, y).item(),
                explained_variance=explained_variance(y_hat, y).item(),
                pearson=pearson_corrcoef(y_hat, y).item(),
                spearman=spearman_corrcoef(y_hat, y).item(),
                ci=concordance_index(y_hat, y).item(),
                rm2=rm2(y_hat, y).item(),
            )
        return dict(
            acc=accuracy(y_hat, y, task="binary").float().item(),
            auc=auroc(y_hat, y, task="binary").float().item(),
            auprc=average_precision(y_hat, y, task="binary").float().item(),
            mcc=matthews_corrcoef(y_hat, y, task="binary").float().item(),
        )

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        """Apply prediction to the whole test dataframe."""
        test["pred"] = test.apply(lambda x: self.predict_pair(x["Target_ID"], x["Drug_ID"]), axis=1)
        return test

    def assess_dataset(self, filename: str, split: str = "test", n_runs: int = 10) -> dict:
        """Assess the performance of the model on a dataset.

        Evaluates on the held-out ``test`` split by default. This used to report on
        ``val``, and to accept ``train_frac``/``n_runs`` while ignoring both, so it
        produced a single number with no error bars.

        ``n_runs`` only varies the result for probabilistic models (``prob=True``);
        the deterministic frequency priors are re-run anyway so the reported std is
        an honest 0 rather than an absent one.

        Returns:
            dict: ``{metric: (mean, std)}`` over ``n_runs`` repeats.
        """
        dataset = pd.read_csv(filename, sep="\t")
        train = dataset[dataset["split"] == "train"]
        held_out = dataset[dataset["split"] == split]
        if held_out.empty:
            raise ValueError(f"No rows with split={split!r} in {filename}")

        runs = []
        for _ in range(n_runs if self.prob else 1):
            self.fit(train)
            runs.append(self.test_metrics(held_out.copy()))

        summary = {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in runs[0]}
        pretty = "\t".join(f"{k.upper()}: {m:.3f}+-{s:.3f}" for k, (m, s) in summary.items())
        print(f"Results ({split}, n={len(runs)})\t{pretty}")
        return summary
