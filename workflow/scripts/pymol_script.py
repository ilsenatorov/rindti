import os

script = """
import psico.fullinit
load resources/structures/{protein}.pdb
from glob import glob
lst = glob("{template_dir}/*.pdb")
for i in lst:cmd.load(i)
extra_fit name CA, {protein}, tmalign
select bsite, {protein} within {radius} of organic
a = set()
f = open("{result_dir}/{protein}_rinhelper.txt", "w")
iterate bsite and n. ca, a.add("0,{chain},{resi},{resi}".format(chain=chain, resi=resi))
f.write("\\n".join(a))
f.close()
"""

outputs = snakemake.output
for output in outputs:
    protein = os.path.basename(output).split(".")[0]
    template_dir = "resources/templates"
    result_dir = "results/run_pymol"
    with open(output, "w") as file:
        file.write(
            script.format(
                protein=protein,
                template_dir=template_dir,
                result_dir=result_dir,
                radius=snakemake.config["bsite"]["radius"],
                chain="{chain}",
                resi="{resi}",
            )
        )
