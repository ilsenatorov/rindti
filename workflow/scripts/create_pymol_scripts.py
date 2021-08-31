import os


def create_script(protein, config):
    """
    Create pymol parsing script for a protein according to the config
    """
    fmt_keywords = {"protein": protein}
    script = """
import psico.fullinit
from glob import glob

cmd.load("resources/structures/{protein}.pdb")
"""

    if config["structures"] == "plddt":
        script += """
            cmd.select("result", "b > {threshold}")
            """
        fmt_keywords["threshold"] = config["plddt"]["threshold"]
    else:
        # template-based
        assert os.path.isdir("resources/templates")
        script += """
            lst = glob("resources/templates/*.pdb")
            templates = [x.split('/')[-1].split('.')[0] for x in lst]
            for i in lst:cmd.load(i)
            scores = {{x : cmd.tmalign("{protein}", x) for x in templates}}
            max_score = max(scores, key=scores.get)

            cmd.extra_fit("name CA", max_score, "tmalign")
            """

        if config["structures"] == "bsite":
            fmt_keywords["radius"] = config["bsite"]["radius"]
            script += """
                cmd.select("result", "br. {protein} within {radius} of organic")
                """
        elif config["structures"] == "template":
            fmt_keywords["radius"] = config["template"]["radius"]
            script += """
                cmd.select("result", "br. {protein} within {radius} of not {protein} and name CA")
                """
    script += """
        cmd.save("results/parsed_structures_{structures}/{protein}.pdb", "result")"""
    fmt_keywords["structures"] = config["structures"]

    return script.format(**fmt_keywords)


if __name__ == "__main__":
    for output in snakemake.output:
        protein = os.path.basename(output).split(".")[0]

        with open(output, "w") as file:
            file.write(create_script(protein, snakemake.config))
