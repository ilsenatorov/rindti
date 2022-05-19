import subprocess

import pytest

from rindti.utils import read_config, write_config


@pytest.fixture
def config():
    return read_config("tests/configs/default_snakemake.yaml")


def run_snakemake(config: dict, tmp_path: str):
    """Run snakemake with the given config."""
    write_config(tmp_path / "tmp_config.yaml", config)
    subprocess.run(
        [
            "snakemake",
            "-j",
            "4",
            "--forceall",
            "--use-conda",
            "--configfile",
            f"{tmp_path / 'tmp_config.yaml'}",
        ],
        check=True,
    )


@pytest.mark.snakemake
@pytest.mark.slow
class TestSnakeMake:
    """Runs all snakemake tests."""

    @pytest.mark.parametrize("method", ["whole", "plddt", "bsite", "template"])
    def test_structures(self, method: str, config: dict, tmp_path: str):
        """Test the various structure-parsing methods."""
        config["prots"]["structs"]["method"] = method
        run_snakemake(config, tmp_path)

    @pytest.mark.parametrize("features", ["rinerator", "distance"])
    def test_features(self, features: str, config: dict, tmp_path: str):
        """Test the graph creation methods."""
        config["prots"]["features"]["method"] = features
        run_snakemake(config, tmp_path)

    @pytest.mark.parametrize("node_feats", ["label", "onehot"])
    @pytest.mark.parametrize("edge_feats", ["label", "onehot", "none"])
    @pytest.mark.parametrize("which", ["prots", "drugs"])
    def test_encodings(self, node_feats: str, edge_feats: str, which: str, config: dict, tmp_path: str):
        """Test the encoding methods for nodes and edges."""
        if which == "prots":
            config[which]["features"]["node_feats"] = node_feats
            config[which]["features"]["edge_feats"] = edge_feats
        else:
            config[which]["node_feats"] = node_feats
            config[which]["edge_feats"] = edge_feats
        run_snakemake(config, tmp_path)

    @pytest.mark.parametrize("split", ["random", "drug", "target"])
    def test_splits(self, split: str, config: dict, tmp_path: str):
        """Test the dataset splitting methods."""
        config["split_data"]["method"] = split
        run_snakemake(config, tmp_path)

    @pytest.mark.parametrize("filtering", ["all", "posneg"])
    @pytest.mark.parametrize("sampling", ["none", "over", "under"])
    def test_parse_dataset(self, filtering: str, sampling: str, config: dict, tmp_path: str):
        """Test the dataset filtering and sampling methods."""
        config["parse_dataset"]["filtering"] = filtering
        config["parse_dataset"]["sampling"] = sampling
        run_snakemake(config, tmp_path)
