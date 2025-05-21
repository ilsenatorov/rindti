import os
import random
import subprocess

import yaml
from jsonargparse import CLI
from tqdm import tqdm

from rindti.utils import IterDict, read_config


def run(config_path: str, threads: int = 1) -> None:
    """Run multiple snakemake instances using subprocess with different configs."""
    tmp_config_path = os.path.join(
        *config_path.split("/")[:-1], f"tmp_config{random.randint(1, 100)}.yaml"
    )
    orig_config = read_config(config_path)
    all_configs = IterDict()(orig_config)
    print(f"Running {len(all_configs)} runs.")
    for i, config in tqdm(enumerate(all_configs)):
        with open(tmp_config_path, "w") as file:
            yaml.dump(config, file)
        subprocess.run(
            f"snakemake -j {threads} --configfile {tmp_config_path} --use-conda > logs/log{i}.txt 2>&1",
            shell=True,
        )
    os.remove(tmp_config_path)


cli = CLI(run)
