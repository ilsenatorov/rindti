final_output = sh._target("prepare_all", sh.namer(config) + ".pkl")
summary = sh._target("summary", sh.namer(config) + ".pkl")


rule parse_dataset:
    input:
        inter=sh.tables["inter"],
    output:
        inter=sh._target("parse_dataset", sh.namer(config["parse_dataset"]) + ".tsv"),
    script:
        "../scripts/parse_dataset.py"


rule split_data:
    input:
        inter=rules.parse_dataset.output.inter,
    output:
        split_data=sh._target(
            "split_data",
            sh.namer({**config["split_data"], **config["parse_dataset"]}) + ".tsv",
        ),
    params:
        method=config["split_data"]["method"],
        train=config["split_data"]["train"],
        val=config["split_data"]["val"],
    script:
        "../scripts/split_data.py"


rule prepare_all:
    input:
        drugs=rules.prepare_drugs.output.pickle,
        prots=prot_data,
        inter=rules.split_data.output.split_data,
    output:
        combined_pickle=final_output,
    script:
        "../scripts/prepare_all.py"


rule summary:
    input:
        pickle=rules.prepare_all.output.combined_pickle,
        structs=expand(parsed_structs, prot=sh.prot_ids),
    output:
        summary=summary,
    params:
        struct_dir=parsed_structs_dir,
    script:
        "../scripts/summary.py"
