import os

import pytest
from snakemake.utils import update_config, validate

from rindti.utils import IterDict, read_config

from .conftest import SNAKEMAKE_CONFIG_DIR, run_snakemake

snakemake_configs = [
    os.path.join(SNAKEMAKE_CONFIG_DIR, x) for x in os.listdir(SNAKEMAKE_CONFIG_DIR) if x != "default.yaml"
]


@pytest.mark.snakemake
@pytest.mark.slow
class TestSnakeMake:
    """Runs all snakemake tests."""

    @pytest.mark.parametrize("config_file", snakemake_configs)
    def test_configs(self, snakemake_config: dict, config_file: dict):
        """Every shipped config must validate against the schema.

        Sweep configs hold lists where the schema wants scalars, since
        ``run_snakemake.py`` expands them into one run each. Expand them the same
        way here so that each resulting run is validated, rather than exempting
        them from the check.
        """
        update_config(snakemake_config, read_config(config_file))
        for expanded in IterDict()(snakemake_config):
            validate(expanded, "workflow/schemas/config.schema.yaml")

    @pytest.mark.parametrize("method", ["whole", "plddt", "bsite", "template"])
    def test_structures(self, method: str, snakemake_config: dict, tmpdir_factory: str):
        """Test the various structure-parsing methods."""
        snakemake_config["prots"]["structs"]["method"] = method
        run_snakemake(snakemake_config, tmpdir_factory)

    @pytest.mark.parametrize("features", ["distance"])
    def test_features(self, features: str, snakemake_config: dict, tmpdir_factory: str):
        """Test the graph creation methods.

        ``esm`` is covered separately by :meth:`test_features_esm`, which needs the
        optional ``fair-esm`` dependency and downloads a 650M-parameter checkpoint.
        """
        snakemake_config["prots"]["features"]["method"] = features
        run_snakemake(snakemake_config, tmpdir_factory)

    @pytest.mark.gpu
    def test_features_esm(self, snakemake_config: dict, tmpdir_factory: str):
        """The ESM featurisation, end to end.

        Re-enabled after being commented out: the old version set ``only_prots``,
        a key that no longer exists and that the schema now rejects outright.
        """
        pytest.importorskip("esm", reason="needs the optional `esm` extra: uv sync --extra esm")
        snakemake_config["prots"]["features"]["method"] = "esm"
        run_snakemake(snakemake_config, tmpdir_factory)

    @pytest.mark.parametrize("node_feats", ["label", "onehot"])
    @pytest.mark.parametrize("edge_feats", ["distance", "none"])
    def test_prot_encodings(
        self,
        node_feats: str,
        edge_feats: str,
        snakemake_config: dict,
        tmpdir_factory: str,
    ):
        """Test the encoding methods for prot nodes and edges."""
        snakemake_config["prots"]["features"]["node_feats"] = node_feats
        snakemake_config["prots"]["features"]["edge_feats"] = edge_feats
        run_snakemake(snakemake_config, tmpdir_factory)

    @pytest.mark.parametrize("node_feats", ["label", "onehot", "rich", "glycan"])
    @pytest.mark.parametrize("edge_feats", ["label", "onehot", "none"])
    def test_drug_encodings(
        self,
        node_feats: str,
        edge_feats: str,
        snakemake_config: dict,
        tmpdir_factory: str,
    ):
        """Test the encoding methods for drug nodes and edges."""
        snakemake_config["drugs"]["node_feats"] = node_feats
        snakemake_config["drugs"]["edge_feats"] = edge_feats
        run_snakemake(snakemake_config, tmpdir_factory)

    @pytest.mark.parametrize("split", ["random", "drug", "target", "cluster_target", "cluster_drug"])
    def test_splits(self, split: str, snakemake_config: dict, tmpdir_factory: str):
        """Test the dataset splitting methods.

        ``cluster_target`` builds the MMseqs2 conda environment on first run; the CI
        job that runs these already has miniforge.
        """
        snakemake_config["split_data"]["method"] = split
        run_snakemake(snakemake_config, tmpdir_factory)

    @pytest.mark.parametrize("log", [True, False])
    def test_task_reg(self, log: bool, snakemake_config: dict, tmpdir_factory: str):
        """The regression task, which no test covered.

        `reg` is not a drop-in swap for `class`: `parse_dataset.py` raises unless
        filtering is `all` and sampling is `none`, since both assume binary labels.
        That is exactly what `config/snakemake/{davis,kiba}_reg.yaml` pin, so pin it
        here too.
        """
        snakemake_config["parse_dataset"]["task"] = "reg"
        snakemake_config["parse_dataset"]["filtering"] = "all"
        snakemake_config["parse_dataset"]["sampling"] = "none"
        snakemake_config["parse_dataset"]["log"] = log
        run_snakemake(snakemake_config, tmpdir_factory)

    @pytest.mark.parametrize("filtering", ["all", "posneg"])
    @pytest.mark.parametrize("sampling", ["none", "over", "under"])
    def test_parse_dataset(self, filtering: str, sampling: str, snakemake_config: dict, tmpdir_factory: str):
        """Test the dataset filtering and sampling methods."""
        snakemake_config["parse_dataset"]["filtering"] = filtering
        snakemake_config["parse_dataset"]["sampling"] = sampling
        run_snakemake(snakemake_config, tmpdir_factory)
