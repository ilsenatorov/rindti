import os

script = """
import psico.fullinit
load resources/structures/{protein}.pdb
from glob import glob
lst = glob("resources/templates/*.pdb")
for i in lst:cmd.load(i)
extra_fit name CA, {protein}, tmalign
select template, {protein} within {radius} of not {protein}
save results/parsed_structures_template/{protein}.pdb, template
"""

outputs = snakemake.output
for output in outputs:
    protein = os.path.basename(output).split(".")[0]
    with open(output, "w") as file:
        file.write(
            script.format(
                protein=protein,
                radius=snakemake.config["template"]["radius"],
            )
        )
