import os
import random
import subprocess

import yaml

from rindti.utils import IterDict, read_config


def run(config_path: str, threads: int = 1) -> None:
    """Run multiple snakemake instances using subprocess with different configs."""
    tmp_config_path = os.path.join(*config_path.split("/")[:-1], f"tmp_config{random.randint(1,100)}.yaml")
    orig_config = read_config(config_path)
    all_configs = IterDict()(orig_config)
    for config in all_configs:
        with open(tmp_config_path, "w") as file:
            yaml.dump(config, file)
        subprocess.run(f"snakemake -j {threads} --configfile {tmp_config_path} --use-conda", shell=True)
    os.remove(tmp_config_path)


run("config/snakemake/glass.yaml")
