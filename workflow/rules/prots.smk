include: "structs.smk"


parsed_graphs = sh._target("parsed_graphs", sh.namer(config["prots"]) + ".pkl")
rinerator_dir = sh._target("rinerator", sh.namer(config["prots"]))
rinerator_output = os.path.join(rinerator_dir, "{prot}/{prot}_h.sif")

if config["prots"]["features"]["method"] == "rinerator":

    ruleorder: parse_rinerator > distance_based


else:

    ruleorder: distance_based > parse_rinerator


rule rinerator:
    input:
        pdb=parsed_structs,
    output:
        sif=rinerator_output,
    log:
        os.path.join(rinerator_dir, "{prot}/log.txt"),
    params:
        dir=rinerator_dir,
    shadow:
        "shallow"
    shell:
        """
        rinerator {input.pdb} {params.dir}/{wildcards.prot} > {log} 2>&1
        """


rule distance_based:
    input:
        pdbs=expand(parsed_structs, prot=sh.prot_ids),
    output:
        pickle=parsed_graphs,
    params:
        threshold=config["prots"]["features"]["distance"]["threshold"],
        node_feats=config["prots"]["features"]["node_feats"],
    script:
        "../scripts/distance_based.py"


rule parse_rinerator:
    input:
        rins=expand(rinerator_output, prot=sh.prot_ids),
    output:
        pickle=parsed_graphs,
    params:
        node_feats=config["prots"]["features"]["node_feats"],
        edge_feats=config["prots"]["features"]["edge_feats"],
    script:
        "../scripts/parse_rinerator.py"
