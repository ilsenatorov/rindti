use_pymol = True if config["prots"]["structs"]["method"] != "whole" else False

parsed_structs_dir = sh._target(
    "parsed_structs",
    sh.namer(config["prots"]["structs"]),
)
pymol_scripts = sh._target(
    "pymol_scripts",
    sh.namer(config["prots"]["structs"]),
    "{prot}.pml",
)

if not use_pymol:
    parsed_structs = sh._source("structures", "{prot}.pdb")
else:
    parsed_structs = os.path.join(parsed_structs_dir, "{prot}.pdb")


rule create_pymol_scripts:
    input:
        sh.raw_structs,
    output:
        scripts=expand(pymol_scripts, prot=sh.prot_ids),
    params:
        parsed_structs_dir=parsed_structs_dir,
        resources=sh.source,
        results=sh.target,
        method=config["prots"]["structs"]["method"],
        method_params=config["prots"]["structs"].get(config["prots"]["structs"]["method"]),
    script:
        "../scripts/create_pymol_scripts.py"


rule run_pymol:
    input:
        script=pymol_scripts,
        struct=sh._source("structures", "{prot}.pdb"),
    output:
        structs=parsed_structs,
    log:
        os.path.join(parsed_structs_dir + "_log", "{prot}.log"),
    conda:
        "../envs/pymol.yaml"
    shell:
        "pymol -k -y -c {input.script} > {log} 2>&1"


# rule save_structure_info:
#     input:
#         structs=expand(structs, protein=targets),
#     output:
#         tsv="{results}/structure_info/{type}_info.tsv".format(
#             results=target,
#             type=config["structs"],
#         ),
#     script:
#         "scripts/save_structure_info.py"
