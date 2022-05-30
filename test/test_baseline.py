import pandas as pd
import pytest

from rindti.models.dti.baseline.run import run


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


@pytest.mark.parametrize("model", ["max", "prot_drug_max"])
@pytest.mark.parametrize("prob", [True, False])
@pytest.mark.parametrize("which", ["both", "prot", "drug"])
def test_baselines(
    split_data: str,
    model: str,
    prob: bool,
    which: str,
):
    run(model, split_data, prob=prob, which=which)
