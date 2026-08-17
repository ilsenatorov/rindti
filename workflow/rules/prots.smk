include: "structs.smk"


prot_data = sh._target("prot_data", sh.namer(config["prots"]) + ".pkl")

if config["prots"]["features"]["method"] == "distance":

    ruleorder: distance_based > esm


elif config["prots"]["features"]["method"] == "esm":

    ruleorder: esm > distance_based


else:
    raise ValueError("Unknown method: {}".format(config["prots"]["features"]["method"]))


rule distance_based:
    input:
        pdbs=expand(parsed_structs, prot=sh.prot_ids),
    output:
        pickle=prot_data,
    params:
        threshold=config["prots"]["features"]["distance"]["threshold"],
        node_feats=config["prots"]["features"]["node_feats"],
        edge_feats=config["prots"]["features"]["edge_feats"],
    script:
        "../scripts/distance_based.py"


rule esm:
    input:
        seqs=sh.tables["prot"],
    output:
        pickle=prot_data,
    script:
        "../scripts/prot_esm.py"
