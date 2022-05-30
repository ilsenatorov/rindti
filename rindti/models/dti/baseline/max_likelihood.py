import pandas as pd
from base_baseline import BaseBaseline


class Max(BaseBaseline):
    """Take the most popular label."""

    def fit(self, train: pd.DataFrame):
        """Fit the model to the training dataframe. Has to have 'Drug_ID', 'Target_ID' and 'Y' columns."""
        self.mean = train["Y"].mean()

    def predict_pair(self, prot_id: str, drug_id: str) -> float:
        """Predict the outcome for a pair of a protein and a drug."""
        return self.mean

    def predict(self, test: pd.DataFrame) -> pd.DataFrame:
        """Apply prediction to the whole test dataframe."""
        test["pred"] = test.apply(lambda x: self.predict_pair(x["Target_ID"], x["Drug_ID"]), axis=1)
        return test


if __name__ == "__main__":
    from jsonargparse import CLI

    cli = CLI(Max.assess_dataset)
