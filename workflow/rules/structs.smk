parsed_structs_dir = sh._target(
    "parsed_structs",
    sh.namer(config["prots"]["structs"]),
)
pymol_scripts = sh._target(
    "pymol_scripts",
    sh.namer(config["prots"]["structs"]),
    "{prot}.pml",
)

if config["prots"]["structs"]["method"] == "whole":
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
        resources=sh.source_dir,
        results=sh.target_dir,
        method=config["prots"]["structs"]["method"],
        other_params=config["prots"]["structs"],
    script:
        "../scripts/create_pymol_scripts.py"


rule run_pymol:
    input:
        script=pymol_scripts,
        struct=sh._source("structures", "{prot}.pdb"),
    output:
        structs=parsed_structs,
    log:
        sh._target("pymol_logs", sh.namer(config["prots"]["structs"]), "{prot}.log"),
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
