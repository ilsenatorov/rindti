import os


def create_script(protein: str, inp: str, params: dict):
    """
    Create pymol parsing script for a protein according to the params.
    """
    resources = params.resources
    results = params.results
    fmt_keywords = {"protein": protein, "resources": resources, "results": results}
    script = [
        "import psico.fullinit",
        "from glob import glob",
        'cmd.load("{inp}")',
    ]

    if params.method == "plddt":
        script.append('cmd.select("result", "b > {threshold}")')
        fmt_keywords["threshold"] = params.other_params[params.method]["threshold"]
    else:
        # template-based
        script += [
            'lst = glob("{resources}/templates/*.pdb")',
            'templates = [x.split("/")[-1].split(".")[0] for x in lst]',
            "for i in lst:cmd.load(i)",
            'scores = {{x : cmd.tmalign("{protein}", x) for x in templates}}',
            "max_score = max(scores, key=scores.get)",
            'cmd.extra_fit("name CA", max_score, "tmalign")',
        ]

        fmt_keywords["radius"] = params.other_params[params.method]["radius"]
        if params.method == "bsite":
            script.append(
                'cmd.select("result", "br. {protein} within {radius} of organic")'
            )
        elif params.method == "template":
            script.append(
                'cmd.select("result", "br. {protein} within {radius} of not {protein} and name CA")'
            )
    script.append('cmd.save("{parsed_structs_dir}/{protein}.pdb", "result")')
    fmt_keywords["parsed_structs_dir"] = params.parsed_structs_dir
    fmt_keywords["structs"] = params.method
    fmt_keywords["inp"] = inp
    return "\n".join(script).format(**fmt_keywords)


if __name__ == "__main__":
    for inp, out in zip(snakemake.input, snakemake.output):
        protein = os.path.basename(out).split(".")[0]

        with open(out, "w") as file:
            file.write(create_script(protein, inp, snakemake.params))
