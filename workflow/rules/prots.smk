include: "structs.smk"


prot_data = sh._target("prot_data", sh.namer(config["prots"]) + ".pkl")
rinerator_dir = sh._target("rinerator", sh.namer(config["prots"]))
rinerator_output = os.path.join(rinerator_dir, "{prot}/{prot}_h.sif")

if config["prots"]["features"]["method"] == "rinerator":

    ruleorder: parse_rinerator > distance_based > esm


elif config["prots"]["features"]["method"] == "distance":

    ruleorder: distance_based > parse_rinerator > esm


elif config["prots"]["features"]["method"] == "esm":

    ruleorder: esm > parse_rinerator > distance_based


else:
    raise ValueError("Unknown method: {}".format(config["prots"]["features"]["method"]))


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


rule parse_rinerator:
    input:
        rins=expand(rinerator_output, prot=sh.prot_ids),
    output:
        pickle=prot_data,
    params:
        node_feats=config["prots"]["features"]["node_feats"],
        edge_feats=config["prots"]["features"]["edge_feats"],
    script:
        "../scripts/parse_rinerator.py"


rule distance_based:
    input:
        pdbs=expand(parsed_structs, prot=sh.prot_ids),
    output:
        pickle=prot_data,
    params:
        threshold=config["prots"]["features"]["distance"]["threshold"],
        node_feats=config["prots"]["features"]["node_feats"],
    script:
        "../scripts/distance_based.py"


rule esm:
    input:
        prot=sh.tables["prot"],
    output:
        pickle=prot_data,
    script:
        "../scripts/prot_esm.py"
