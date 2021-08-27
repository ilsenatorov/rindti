import os

config = snakemake.config

script = """
import psico.fullinit
from glob import glob

cmd.load("resources/structures/{protein}.pdb")
lst = glob("resources/templates/*.pdb")
templates = [x.split('/')[-1].split('.')[0] for x in lst]
for i in lst:cmd.load(i)
scores = {{x : cmd.tmalign("{protein}", x) for x in templates}}
max_score = max(scores, key=scores.get)

cmd.extra_fit("name CA", max_score, "tmalign")
"""

if config["structures"] == "bsite":
    radius = config["bsite"]["radius"]
    script += """
cmd.select("bsite", "br. {protein} within {radius} of organic")
cmd.save("results/parsed_structures_bsite/{protein}.pdb", "bsite")
"""
elif config["structures"] == "template":
    radius = config["template"]["radius"]
    script += """
select template, br. {protein} within {radius} of not {protein} and name CA
save results/parsed_structures_template/{protein}.pdb, template
"""
else:
    raise ValueError("Unknown structures type!")


for output in snakemake.output:
    protein = os.path.basename(output).split(".")[0]
    with open(output, "w") as file:
        file.write(
            script.format(
                protein=protein,
                radius=radius,
            )
        )
