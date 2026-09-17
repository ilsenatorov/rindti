import os
import subprocess
import tempfile

import yaml
from jsonargparse import CLI
from tqdm import tqdm

from rindti.utils import IterDict, read_config


def run(config_path: str, threads: int = 1, conda: bool = True) -> None:
    """Run multiple snakemake instances using subprocess with different configs.

    Args:
        config_path: sweep config; any list value is expanded into one snakemake run.
        threads: cores handed to each snakemake invocation.
        conda: pass ``--software-deployment-method conda``. Needed locally, where
            MMseqs2 comes from ``workflow/envs/mmseqs.yaml``. Must be **off** inside the
            HPC image, which has MMseqs2 on ``PATH`` and no conda at all - this used to
            be hardcoded on, so the documented ablation command could not run there.
    """
    orig_config = read_config(config_path)
    all_configs = IterDict()(orig_config)
    os.makedirs("logs", exist_ok=True)
    print(f"Running {len(all_configs)} runs.")

    deployment = "--software-deployment-method conda " if conda else ""
    # A unique temporary file rather than `tmp_config<random 1-100>.yaml` next to the
    # sweep config: two concurrent sweeps could previously collide on the same name, and
    # a crash left the file behind in the repo.
    failed = []
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        tmp_config_path = handle.name
    try:
        for i, config in tqdm(enumerate(all_configs), total=len(all_configs)):
            with open(tmp_config_path, "w") as file:
                yaml.dump(config, file)
            log = f"logs/log{i}.txt"
            result = subprocess.run(
                f"snakemake -s workflow/Snakefile -j {threads} --configfile {tmp_config_path} "
                f"{deployment}> {log} 2>&1",
                shell=True,
            )
            # Failures used to be swallowed: output is redirected to a log file and the
            # return code was discarded, so a sweep could "finish" with half its datasets
            # missing and only surface hours later when training jobs could not find them.
            if result.returncode != 0:
                failed.append((i, log))
                print(f"Run {i} FAILED (exit {result.returncode}); see {log}")
    finally:
        os.remove(tmp_config_path)

    if failed:
        raise SystemExit(
            f"{len(failed)} of {len(all_configs)} snakemake runs failed: "
            + ", ".join(f"run {i} ({log})" for i, log in failed)
        )
    print(f"All {len(all_configs)} runs succeeded.")


cli = CLI(run)
