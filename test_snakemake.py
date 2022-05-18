from nbformat import write
from rindti.utils import read_config, write_config
import pytest
import subprocess


@pytest.fixture
def config():
    return read_config("config/snakemake/default.yaml")


@pytest.mark.parametrize("method", ["whole", "plddt", "bsite", "template"])
def test_structures(method: str, config: dict, tmp_path: str):
    config["prots"]["structs"]["method"] = method
    write_config(tmp_path / "tmp_config.yaml", config)
    subprocess.run(f"snakemake -j 1 --configfile {tmp_path / 'tmp_config.yaml'} --use-conda", shell=True)


@pytest.mark.parametrize("features", ["rinerator", "distance"])
def test_features(features: str, config: dict, tmp_path: str):
    config["prots"]["features"]["method"] = features
    write_config(tmp_path / "tmp_config.yaml", config)
    subprocess.run(f"snakemake -j 1 --configfile {tmp_path / 'tmp_config.yaml'} --use-conda", shell=True)


@pytest.mark.parametrize("node_feats", ["label", "onehot"])
@pytest.mark.parametrize("which", ["prots", "drugs"])
def test_node_encodings(node_feats: str, which: str, config: dict, tmp_path: str):
    if which == "prots":
        config[which]["features"]["node_feats"] = node_feats
    else:
        config[which]["node_feats"] = node_feats
    write_config(tmp_path / "tmp_config.yaml", config)
    subprocess.run(f"snakemake -j 1 --configfile {tmp_path / 'tmp_config.yaml'} --use-conda", shell=True)

@pytest.mark.parametrize("edge_feats", ["label", "onehot", "none"])
@pytest.mark.parametrize("which", ["prots", "drugs"])
def test_edge_encodings(edge_feats: str, which: str, config: dict, tmp_path: str):
    if which == "prots":
        config[which]["features"]["edge_feats"] = edge_feats
    else:
        config[which]["edge_feats"] = edge_feats
    write_config(tmp_path / "tmp_config.yaml", config)
    subprocess.run(f"snakemake -j 1 --configfile {tmp_path / 'tmp_config.yaml'} --use-conda", shell=True)
