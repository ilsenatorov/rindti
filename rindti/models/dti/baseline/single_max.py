import numpy as np
import pandas as pd

from base_baseline import BaseBaseline


class SingleMax(BaseBaseline):
    def __init__(self, **kwargs):
        """Init if the model should be protein-based or ligand based"""
        super(SingleMax, self).__init__(**kwargs)
        self.key = "Target_ID" if kwargs["prot"] else "Drug_ID"

    def fit(self, train: pd.DataFrame):
        """Fit the model to the training dataframe. Has to have 'Drug_ID', 'Target_ID' and 'Y' columns."""
        self.group = train.groupby(self.key).agg({"Y": ["mean", "std"]})

    def predict_pair(self, prot_id: str, drug_id: str) -> float:
        """Predict the outcome for a protein or a drug."""
        try:
            if self.prob:
                return np.random.normal(
                    loc=self.group.loc[prot_id if self.key == "Target_ID" else drug_id, "Y"],
                    scale=self.group.loc[prot_id if self.key == "Target_ID" else drug_id, "std"],
                )
            return self.group.loc[prot_id if self.key == "Target_ID" else drug_id, "Y"]
        except KeyError:
            return 0.5
