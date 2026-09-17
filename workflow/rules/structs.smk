parsed_structs_dir = sh._target(
    "parsed_structs",
    sh.namer(config["prots"]["structs"]),
)
structs_method = config["prots"]["structs"]["method"]

if structs_method == "whole":
    parsed_structs = sh._source("structures", "{prot}.pdb")
else:
    parsed_structs = os.path.join(parsed_structs_dir, "{prot}.pdb")

# Declared as rule inputs rather than globbed inside the script: the template library
# is data the selection depends on, so adding or removing a template has to invalidate
# the parsed structures. The old PyMOL script globbed at runtime and the DAG never
# knew the templates existed.
templates = (
    sorted(glob.glob(sh._source("templates", "*.pdb")))
    if structs_method in ("bsite", "template")
    else []
)


rule parse_structs:
    input:
        struct=sh._source("structures", "{prot}.pdb"),
        templates=templates,
    output:
        struct=parsed_structs,
    params:
        method=structs_method,
        other_params=config["prots"]["structs"],
    script:
        "../scripts/parse_structs.py"
