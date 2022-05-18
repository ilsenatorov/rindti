import subprocess

import pytest
from nbformat import write

from rindti.utils import read_config, write_config


@pytest.fixture
def config():
    return read_config("config/snakemake/default.yaml")


@pytest.mark.parametrize("method", ["whole", "plddt", "bsite", "template"])
def test_structures(method: str, config: dict, tmp_path: str):
    config["prots"]["structs"]["method"] = method
    write_config(tmp_path / "tmp_config.yaml", config)
    subprocess.run(f"snakemake --forceall -j 1 --configfile {tmp_path / 'tmp_config.yaml'} --use-conda", shell=True)


@pytest.mark.parametrize("features", ["rinerator", "distance"])
def test_features(features: str, config: dict, tmp_path: str):
    config["prots"]["features"]["method"] = features
    write_config(tmp_path / "tmp_config.yaml", config)
    subprocess.run(f"snakemake --forceall -j 1 --configfile {tmp_path / 'tmp_config.yaml'} --use-conda", shell=True)


@pytest.mark.parametrize("node_feats", ["label", "onehot"])
@pytest.mark.parametrize("edge_feats", ["label", "onehot", "none"])
@pytest.mark.parametrize("which", ["prots", "drugs"])
def test_encodings(node_feats: str, edge_feats: str, which: str, config: dict, tmp_path: str):
    if which == "prots":
        config[which]["features"]["node_feats"] = node_feats
        config[which]["features"]["edge_feats"] = edge_feats
    else:
        config[which]["node_feats"] = node_feats
        config[which]["edge_feats"] = edge_feats
    write_config(tmp_path / "tmp_config.yaml", config)
    subprocess.run(f"snakemake --forceall -j 1 --configfile {tmp_path / 'tmp_config.yaml'} --use-conda", shell=True)


@pytest.mark.parametrize("split", ["random", "drug", "target"])
def test_splits(split: str, config: dict, tmp_path: str):
    config["split_data"]["method"] = split
    write_config(tmp_path / "tmp_config.yaml", config)
    subprocess.run(f"snakemake --forceall -j 1 --configfile {tmp_path / 'tmp_config.yaml'} --use-conda", shell=True)


@pytest.mark.parametrize("filtering", ["all", "posneg", "balanced"])
@pytest.mark.parametrize("sampling", ["none", "over", "under"])
def test_parse_dataset(filtering: str, sampling: str, config: dict, tmp_path: str):
    config["parse_dataset"]["filtering"] = filtering
    config["parse_dataset"]["sampling"] = sampling
    write_config(tmp_path / "tmp_config.yaml", config)
    subprocess.run(f"snakemake --forceall -j 1 --configfile {tmp_path / 'tmp_config.yaml'} --use-conda", shell=True)
