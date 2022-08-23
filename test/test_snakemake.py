import os

import pytest
from snakemake.utils import validate

from rindti.utils import read_config

from .conftest import DEFAULT_CONFIG, SNAKEMAKE_CONFIG_DIR, TEST_CONFIG, run_snakemake

snakemake_configs = [
    os.path.join(SNAKEMAKE_CONFIG_DIR, x) for x in os.listdir(SNAKEMAKE_CONFIG_DIR) if x != "default.yaml"
]


@pytest.fixture()
def snakemake_config():
    default_test_config = read_config(DEFAULT_CONFIG)
    test_config = read_config(TEST_CONFIG)
    default_test_config.update(test_config)
    return default_test_config


@pytest.mark.snakemake
@pytest.mark.slow
class TestSnakemake:
    """Runs all snakemake tests."""

    @pytest.mark.parametrize("config_file", snakemake_configs)
    def test_configs(self, snakemake_config: dict, config_file: dict):
        """Test all snakemake configs."""
        snakemake_config.update(read_config(config_file))
        validate(snakemake_config, "workflow/schemas/config.schema.yaml")

    @pytest.mark.parametrize("method", ["whole", "plddt", "bsite", "template"])
    def test_structures(self, method: str, snakemake_config: dict, tmpdir_factory: str):
        """Test the various structure-parsing methods."""
        print(snakemake_config)
        snakemake_config["prots"]["structs"]["method"] = method
        snakemake_config["only_prots"] = True
        run_snakemake(snakemake_config, tmpdir_factory)

    @pytest.mark.gpu
    def test_features_esm(self, snakemake_config: dict, tmpdir_factory: str):
        """Test the graph creation methods."""
        snakemake_config["prots"]["features"]["method"] = "esm"
        snakemake_config["only_prots"] = True
        run_snakemake(snakemake_config, tmpdir_factory)

    @pytest.mark.parametrize("features", ["rinerator", "distance"])  # NOTE esm testing disabled
    def test_features(self, features: str, snakemake_config: dict, tmpdir_factory: str):
        """Test the graph creation methods."""
        snakemake_config["prots"]["features"]["method"] = features
        snakemake_config["only_prots"] = True
        run_snakemake(snakemake_config, tmpdir_factory)

    @pytest.mark.parametrize("node_feats", ["label", "onehot"])
    @pytest.mark.parametrize("edge_feats", ["label", "onehot", "none"])
    def test_prot_encodings(self, node_feats: str, edge_feats: str, snakemake_config: dict, tmpdir_factory: str):
        """Test the encoding methods for prot nodes and edges."""
        snakemake_config["prots"]["features"]["node_feats"] = node_feats
        snakemake_config["prots"]["features"]["edge_feats"] = edge_feats
        run_snakemake(snakemake_config, tmpdir_factory)

    @pytest.mark.parametrize("node_feats", ["label", "onehot", "glycan"])
    @pytest.mark.parametrize("edge_feats", ["label", "onehot", "none"])
    def test_drug_encodings(self, node_feats: str, edge_feats: str, snakemake_config: dict, tmpdir_factory: str):
        """Test the encoding methods for drug nodes and edges."""
        snakemake_config["drugs"]["node_feats"] = node_feats
        snakemake_config["drugs"]["edge_feats"] = edge_feats
        run_snakemake(snakemake_config, tmpdir_factory)

    @pytest.mark.parametrize("split", ["random", "drug", "target"])
    def test_splits(self, split: str, snakemake_config: dict, tmpdir_factory: str):
        """Test the dataset splitting methods."""
        snakemake_config["split_data"]["method"] = split
        run_snakemake(snakemake_config, tmpdir_factory)

    @pytest.mark.parametrize("filtering", ["all", "posneg"])
    @pytest.mark.parametrize("sampling", ["none", "over", "under"])
    def test_parse_dataset(self, filtering: str, sampling: str, snakemake_config: dict, tmpdir_factory: str):
        """Test the dataset filtering and sampling methods."""
        snakemake_config["parse_dataset"]["filtering"] = filtering
        snakemake_config["parse_dataset"]["sampling"] = sampling
        run_snakemake(snakemake_config, tmpdir_factory)
