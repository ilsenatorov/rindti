import numpy as np
import pandas as pd

from .base_baseline import BaseBaseline


class ProtDrugMax(BaseBaseline):
    """Take the average of the drug and prot labels."""

    def __init__(self, which: str = "both", prob: bool = False):
        super().__init__(prob)
        self.which = which

    def fit(self, train: pd.DataFrame):
        """Fit the model to the training dataframe. Has to have 'Drug_ID', 'Target_ID' and 'Y' columns."""
        train["count"] = 1
        self.protgroup = train.groupby("Target_ID").agg({"Y": ["mean", "std"], "count": "sum"}).loc[:, "Y"]
        self.druggroup = train.groupby("Drug_ID").agg({"Y": ["mean", "std"], "count": "sum"}).loc[:, "Y"]

    def _get_val(self, group: pd.DataFrame, id: str) -> float:
        if id not in group.index:
            return 0.5
        elif self.prob:
            return np.random.normal(loc=group.at[id, "mean"], scale=group.at[id, "std"])
        return group.at[id, "mean"]

    def predict_pair(self, prot_id: str, drug_id: str) -> float:
        """Predict the outcome for a pair of a protein and a drug."""
        if self.which == "both":
            return (self._get_val(self.protgroup, prot_id) + self._get_val(self.druggroup, drug_id)) / 2
        elif self.which == "prot":
            return self._get_val(self.protgroup, prot_id)
        elif self.which == "drug":
            return self._get_val(self.druggroup, drug_id)
        else:
            raise ValueError(f"Unknown value for which: {self.which}")
