import numpy as np
import pandas as pd
from base_baseline import BaseBaseline


class ProtDrugMax(BaseBaseline):
    """Take the average of the drug and prot labels."""

    def fit(self, train: pd.DataFrame):
        """Fit the model to the training dataframe. Has to have 'Drug_ID', 'Target_ID' and 'Y' columns."""
        train["count"] = 1
        self.protgroup = train.groupby("Target_ID").agg({"Y": ["mean", "std"], "count": "sum"})
        self.druggroup = train.groupby("Drug_ID").agg({"Y": ["mean", "std"], "count": "sum"})

    def predict_pair(self, prot_id: str, drug_id: str) -> float:
        """Predict the outcome for a pair of a protein and a drug."""
        try:
            if self.prob:
                prot_mean = np.random.normal(
                    loc=self.protgroup.loc[prot_id, "Y"], scale=self.protgroup.loc[prot_id, "std"]
                )
            else:
                prot_mean = self.protgroup.loc[prot_id, "Y"]
        except KeyError:
            prot_mean = 0.5

        try:
            if self.prob:
                drug_mean = np.random.normal(
                    loc=self.druggroup.loc[drug_id, "Y"], scale=self.druggroup.loc[drug_id, "std"]
                )
            else:
                drug_mean = self.druggroup.loc[drug_id, "Y"]
        except KeyError:
            drug_mean = 0.5

        return (prot_mean + drug_mean) / 2
