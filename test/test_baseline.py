import json

import numpy as np
import pandas as pd
import pytest

from rindti.models.dti.baseline import KNN, ProtDrugMax
from rindti.models.dti.baseline.knn import ecfp_matrix, kmer_matrix, tanimoto
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


class TestKNN:
    """Similarity-weighted nearest training pairs, on cases small enough to work by hand."""

    # P1 and P1m differ by one residue; P2 shares no 3-mer with either.
    SEQS = {"P1": "MKTAYIAKQRQISFVKSHFSRQ", "P1m": "MKTAYIAKQRQISFVKSHFSRA", "P2": "GGGGGGGGGG"}
    # D1 and D1m are benzene and toluene; D2 is unrelated.
    SMILES = {"D1": "c1ccccc1O", "D1m": "Cc1ccccc1O", "D2": "CCCCCCCCCC(=O)O"}

    def _fit(self, rows, **kwargs) -> KNN:
        model = KNN(self.SEQS, self.SMILES, **kwargs)
        model.fit(pd.DataFrame(rows, columns=["Target_ID", "Drug_ID", "Y"]))
        return model

    def _predict(self, model: KNN, pairs) -> list[float]:
        test = pd.DataFrame(pairs, columns=["Target_ID", "Drug_ID"]).assign(Y=0)
        return model.predict(test)["pred"].tolist()

    def test_tanimoto_by_hand(self):
        """{AAA} against {AAA, AAB} shares one of two k-mers; two empty sets score 0."""
        m = kmer_matrix(["AAAA", "AAAB", "", ""])
        sim = tanimoto(m, m)
        assert sim[0, 1] == pytest.approx(0.5)
        assert sim[0, 0] == pytest.approx(1.0)
        assert sim[2, 3] == 0.0
        assert not np.isnan(sim).any()

    def test_nearest_neighbour_wins(self):
        """k=1: a slightly mutated protein and drug take the label of their close relative."""
        model = self._fit([("P1", "D1", 1), ("P2", "D2", 0)], k=1)
        assert self._predict(model, [("P1m", "D1m"), ("P2", "D2")]) == [1.0, 0.0]

    def test_weighted_mean_by_hand(self):
        """k=2 over two training pairs is their similarity-weighted mean label."""
        model = self._fit([("P1", "D1", 1), ("P1m", "D2", 0)], k=2)
        kmers = kmer_matrix([self.SEQS["P1"], self.SEQS["P1"], self.SEQS["P1m"]])
        prot = tanimoto(kmers[:1], kmers[1:])[0]
        drug = tanimoto(ecfp_matrix([self.SMILES["D1m"]]), ecfp_matrix([self.SMILES["D1"], self.SMILES["D2"]]))[0]
        w = prot * drug
        expected = w[0] / (w[0] + w[1])
        assert 0 < expected < 1  # both neighbours contribute, so this is not a trivial case
        assert self._predict(model, [("P1", "D1m")])[0] == pytest.approx(expected)

    def test_no_similarity_falls_back_to_train_mean(self):
        """A protein sharing no 3-mer with any training protein has zero weight everywhere."""
        model = self._fit([("P1", "D1", 1), ("P1m", "D1", 0), ("P1", "D2", 1)], k=2)
        assert self._predict(model, [("P2", "D1")])[0] == pytest.approx(2 / 3)

    def test_missing_id_fails_loudly(self):
        """An ID without a sequence must not silently score as dissimilar to everything."""
        model = self._fit([("P1", "D1", 1)])
        with pytest.raises(KeyError, match="protein"):
            self._predict(model, [("P9", "D1")])

    def test_run_needs_tables(self, tmp_path):
        with pytest.raises(ValueError, match="--prots and --drugs"):
            run("knn", str(tmp_path / "unused.tsv"))

    def test_run_end_to_end(self, tmp_path):
        """Through the CLI entry point, with tables on disk as the pipeline writes them."""
        prots, drugs, split = tmp_path / "prot.tsv", tmp_path / "lig.tsv", tmp_path / "split.tsv"
        pd.DataFrame({"Target_ID": list(self.SEQS), "Target": list(self.SEQS.values())}).to_csv(prots, sep="\t")
        pd.DataFrame({"Drug_ID": list(self.SMILES), "Drug": list(self.SMILES.values())}).to_csv(drugs, sep="\t")
        pd.DataFrame(
            [("P1", "D1", 1, "train"), ("P2", "D2", 0, "train"), ("P1m", "D1m", 1, "test"), ("P2", "D1m", 0, "test")],
            columns=["Target_ID", "Drug_ID", "Y", "split"],
        ).to_csv(split, sep="\t")
        summary = run("knn", str(split), prots=str(prots), drugs=str(drugs), k=1)
        assert summary["auc"][0] == pytest.approx(1.0)
