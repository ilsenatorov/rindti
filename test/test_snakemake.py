import os

import pytest
from snakemake.utils import validate

from rindti.utils import read_config

from .conftest import SNAKEMAKE_CONFIG_DIR, run_snakemake

snakemake_configs = [
    os.path.join(SNAKEMAKE_CONFIG_DIR, x) for x in os.listdir(SNAKEMAKE_CONFIG_DIR) if x != "default.yaml"
]


@pytest.mark.snakemake
@pytest.mark.slow
class TestSnakeMake:
    """Runs all snakemake tests."""

    @pytest.mark.parametrize("config_file", snakemake_configs)
    def test_configs(self, dti_snakemake_config: dict, config_file: dict):
        """Test all snakemake configs."""
        dti_snakemake_config.update(read_config(config_file))
        validate(dti_snakemake_config, "workflow/schemas/config.schema.yaml")

    @pytest.mark.parametrize("method", ["whole", "plddt", "bsite", "template"])
    def test_structures(self, method: str, dti_snakemake_config: dict, tmpdir_factory: str):
        """Test the various structure-parsing methods."""
        dti_snakemake_config["prots"]["structs"]["method"] = method
        dti_snakemake_config["only_prots"] = True
        run_snakemake(dti_snakemake_config, tmpdir_factory)

    @pytest.mark.gpu
    def test_features_esm(self, dti_snakemake_config: dict, tmpdir_factory: str):
        """Test the graph creation methods."""
        dti_snakemake_config["prots"]["features"]["method"] = "esm"
        dti_snakemake_config["only_prots"] = True
        run_snakemake(dti_snakemake_config, tmpdir_factory)

    @pytest.mark.parametrize("features", ["rinerator", "distance"])  # NOTE esm testing disabled
    def test_features(self, features: str, dti_snakemake_config: dict, tmpdir_factory: str):
        """Test the graph creation methods."""
        dti_snakemake_config["prots"]["features"]["method"] = features
        dti_snakemake_config["only_prots"] = True
        run_snakemake(dti_snakemake_config, tmpdir_factory)

    @pytest.mark.parametrize("node_feats", ["label", "onehot"])
    @pytest.mark.parametrize("edge_feats", ["label", "onehot", "none"])
    def test_prot_encodings(self, node_feats: str, edge_feats: str, dti_snakemake_config: dict, tmpdir_factory: str):
        """Test the encoding methods for prot nodes and edges."""
        dti_snakemake_config["prots"]["features"]["node_feats"] = node_feats
        dti_snakemake_config["prots"]["features"]["edge_feats"] = edge_feats
        run_snakemake(dti_snakemake_config, tmpdir_factory)

    @pytest.mark.parametrize("node_feats", ["label", "onehot", "glycan"])
    @pytest.mark.parametrize("edge_feats", ["label", "onehot", "none"])
    def test_drug_encodings(self, node_feats: str, edge_feats: str, dti_snakemake_config: dict, tmpdir_factory: str):
        """Test the encoding methods for drug nodes and edges."""
        dti_snakemake_config["drugs"]["node_feats"] = node_feats
        dti_snakemake_config["drugs"]["edge_feats"] = edge_feats
        run_snakemake(dti_snakemake_config, tmpdir_factory)

    @pytest.mark.parametrize("split", ["random", "drug", "target"])
    def test_splits(self, split: str, dti_snakemake_config: dict, tmpdir_factory: str):
        """Test the dataset splitting methods."""
        dti_snakemake_config["split_data"]["method"] = split
        run_snakemake(dti_snakemake_config, tmpdir_factory)

    @pytest.mark.parametrize("filtering", ["all", "posneg"])
    @pytest.mark.parametrize("sampling", ["none", "over", "under"])
    def test_parse_dataset(self, filtering: str, sampling: str, dti_snakemake_config: dict, tmpdir_factory: str):
        """Test the dataset filtering and sampling methods."""
        dti_snakemake_config["parse_dataset"]["filtering"] = filtering
        dti_snakemake_config["parse_dataset"]["sampling"] = sampling
        run_snakemake(dti_snakemake_config, tmpdir_factory)
