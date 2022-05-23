import os
import subprocess

import pytest
from snakemake.utils import validate

from rindti.utils import read_config, write_config

SNAKEMAKE_CONFIG_DIR = "config/snakemake"
DEFAULT_CONFIG = os.path.join(SNAKEMAKE_CONFIG_DIR, "default.yaml")
TEST_CONFIG = os.path.join(SNAKEMAKE_CONFIG_DIR, "test.yaml")

snakemake_configs = [
    os.path.join(SNAKEMAKE_CONFIG_DIR, x) for x in os.listdir(SNAKEMAKE_CONFIG_DIR) if x != "default.yaml"
]


@pytest.fixture
def config():
    default_config = read_config(DEFAULT_CONFIG)
    test_config = read_config(TEST_CONFIG)
    default_config.update(test_config)
    return default_config


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

    @pytest.mark.parametrize("config_file", snakemake_configs)
    def test_configs(self, config: dict, config_file: dict):
        """Test all snakemake configs."""
        config.update(read_config(config_file))
        validate(config, "workflow/schemas/config.schema.yaml")

    @pytest.mark.parametrize("method", ["whole", "plddt", "bsite", "template"])
    def test_structures(self, method: str, config: dict, tmp_path: str):
        """Test the various structure-parsing methods."""
        config["prots"]["structs"]["method"] = method
        config["only_prots"] = True
        run_snakemake(config, tmp_path)

    @pytest.mark.parametrize("features", ["rinerator", "distance"])  # NOTE esm testing disabled
    def test_features(self, features: str, config: dict, tmp_path: str):
        """Test the graph creation methods."""
        config["prots"]["features"]["method"] = features
        config["only_prots"] = True
        run_snakemake(config, tmp_path)

    @pytest.mark.parametrize("node_feats", ["label", "onehot"])
    @pytest.mark.parametrize("edge_feats", ["label", "onehot", "none"])
    def test_prot_encodings(self, node_feats: str, edge_feats: str, config: dict, tmp_path: str):
        """Test the encoding methods for prot nodes and edges."""
        config["prots"]["features"]["node_feats"] = node_feats
        config["prots"]["features"]["edge_feats"] = edge_feats
        run_snakemake(config, tmp_path)

    @pytest.mark.parametrize("node_feats", ["label", "onehot", "glycan", "glycanone"])
    @pytest.mark.parametrize("edge_feats", ["label", "onehot", "none"])
    def test_drug_encodings(self, node_feats: str, edge_feats: str, config: dict, tmp_path: str):
        """Test the encoding methods for drug nodes and edges."""
        config["drugs"]["node_feats"] = node_feats
        config["drugs"]["edge_feats"] = edge_feats
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
