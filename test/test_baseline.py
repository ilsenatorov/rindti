import json

import pandas as pd
import pytest

from rindti.models.dti.baseline import ProtDrugMax
from rindti.models.dti.baseline.run import report, run, score


@pytest.fixture
def train() -> pd.DataFrame:
    """Contains 10 interactions between 3 drugs and 5 prots."""

    return pd.DataFrame(
        {
            "Drug_ID": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C"],
            "Target_ID": ["P1", "P2", "P3", "P4", "P5", "P1", "P2", "P3", "P4", "P5"],
            "Y": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        }
    )


@pytest.fixture
def test() -> pd.DataFrame:
    """Contains 5 interactions. One drug and one prot are not known."""
    return pd.DataFrame(
        {
            "Drug_ID": ["C", "C", "C", "A", "D"],
            "Target_ID": ["P1", "P2", "P3", "P6", "P6"],
            "Y": [0, 1, 0, 1, 0],
        }
    )


# `split_data` is produced by the full snakemake pipeline.
@pytest.mark.snakemake
@pytest.mark.parametrize("model", ["max", "prot_drug_max"])
@pytest.mark.parametrize("prob", [True, False])
@pytest.mark.parametrize("which", ["both", "prot", "drug"])
def test_baselines(
    split_data: str,
    model: str,
    prob: bool,
    which: str,
):
    summary = run(model, split_data, prob=prob, which=which, n_runs=3)
    assert set(summary) == {"acc", "auc", "auprc", "mcc"}
    for metric, (mean, std) in summary.items():
        assert std >= 0, metric
        if metric in ("acc", "auc", "auprc"):
            assert 0 <= mean <= 1, f"{metric} out of range: {mean}"
        else:  # MCC
            assert -1 <= mean <= 1, f"{metric} out of range: {mean}"


def test_regression_baseline(train: pd.DataFrame, test: pd.DataFrame, tmp_path):
    """Affinities, not 0/1: averaged priors, and the train mean for unseen IDs, not 0.5."""
    train = train.assign(Y=[5.0, 7.0, 6.0, 8.0, 5.5, 6.5, 7.5, 6.0, 8.5, 5.0])
    test = test.assign(Y=[6.0, 7.0, 6.5, 7.5, 6.0])

    model = ProtDrugMax(task="reg")
    model.fit(train.copy())
    pred = model.predict(test.copy())["pred"].tolist()
    drug_c = train[train.Drug_ID == "C"].Y.mean()
    assert pred[0] == pytest.approx((train[train.Target_ID == "P1"].Y.mean() + drug_c) / 2)
    assert pred[4] == pytest.approx(train.Y.mean())  # neither D nor P6 seen in train

    path = tmp_path / "split.tsv"
    pd.concat([train.assign(split="train"), test.assign(split="test")]).to_csv(path, sep="\t")
    summary = report(str(path), task="reg")
    assert set(summary) == {"max", "prot_drug_max_both", "prot_drug_max_prot", "prot_drug_max_drug"}
    assert set(summary["max"]) == {"mae", "mse", "explained_variance", "pearson", "spearman", "ci", "rm2"}
    assert summary["max"]["mse"] == pytest.approx(((test.Y - train.Y.mean()) ** 2).mean())


class TestScore:
    """Putting a model's metrics against the floor the pipeline writes per split."""

    # Davis random split, as `report` writes it.
    BASELINES = {
        "max": {"auc": 0.500, "mcc": 0.066},
        "prot_drug_max_both": {"auc": 0.843, "mcc": 0.066},
        "prot_drug_max_prot": {"auc": 0.669, "mcc": 0.067},
        "prot_drug_max_drug": {"auc": 0.794, "mcc": 0.231},
    }

    def test_published_value(self):
        """The reference model's 0.887 AUROC over a 0.843 floor is the paper's 0.28."""
        out = score({"auc": 0.887}, self.BASELINES)
        assert out["auc"]["baseline"] == 0.843
        assert out["auc"]["boss"] == pytest.approx(0.28, abs=0.005)

    def test_floor_is_per_metric(self):
        """MCC takes the drug-only prior, which is far stronger than the average."""
        out = score({"auc": 0.887, "mcc": 0.40}, self.BASELINES)
        assert out["mcc"]["baseline"] == 0.231

    def test_metrics_without_an_oracle_are_skipped(self):
        """`loss` has no finite best value, so it is dropped rather than guessed at."""
        assert "loss" not in score({"auc": 0.887, "loss": 0.12}, self.BASELINES)

    def test_mismatched_task_scores_nothing(self):
        """A classification model against a regression floor, rather than nonsense."""
        assert score({"auc": 0.887}, {"max": {"mse": 1.0, "ci": 0.5}}) == {}

    def test_reads_a_json_path(self, tmp_path):
        """What the pipeline writes beside each split is a file, not a dict."""
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps(self.BASELINES))
        assert score({"auc": 0.887}, str(path)) == score({"auc": 0.887}, self.BASELINES)
