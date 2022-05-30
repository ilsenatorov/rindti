import pandas as pd
from base_baseline import BaseBaseline


class ProtDrugMax(BaseBaseline):
    """Take the average of the drug and prot labels."""

    def fit(self, train: pd.DataFrame):
        """Fit the model to the training dataframe. Has to have 'Drug_ID', 'Target_ID' and 'Y' columns."""
        train["count"] = 1
        self.protgroup = train.groupby("Target_ID").agg({"Y": "mean", "count": "sum"})
        self.druggroup = train.groupby("Drug_ID").agg({"Y": "mean", "count": "sum"})

    def predict_pair(self, prot_id: str, drug_id: str) -> float:
        """Predict the outcome for a pair of a protein and a drug."""
        try:
            prot_mean = self.protgroup.loc[prot_id, "Y"]
        except KeyError:
            prot_mean = 0.5
        try:
            drug_mean = self.druggroup.loc[drug_id, "Y"]
        except KeyError:
            drug_mean = 0.5
        return (prot_mean + drug_mean) / 2

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        """Apply prediction to the whole test dataframe."""
        test["pred"] = test.apply(lambda x: self.predict_pair(x["Target_ID"], x["Drug_ID"]), axis=1)
        return test
