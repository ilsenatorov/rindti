import os
import subprocess

import yaml

with open("config/snakemake/standard.yaml", "r") as file:
    og_config = yaml.safe_load(file)

for split in ["random", "colddrug", "coldtarget"]:
    for filtering in ["posneg"]:
        new_config = og_config.copy()
        new_config["split"]["method"] = split
        new_config["parse_dataset"]["filtering"] = filtering
        with open("config/snakemake/tmp.yaml", "w") as file:
            yaml.dump(new_config, file)
            subprocess.run("snakemake -j 16 --configfile config/snakemake/tmp.yaml --use-conda", shell=True)
os.remove("config/snakemake/tmp.yaml")
