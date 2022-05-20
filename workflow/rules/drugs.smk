drug_output = sh._target("prepare_drugs", sh.namer(config["drugs"]) + ".pkl")


rule prepare_drugs:
    input:
        lig=sh.tables["lig"],
    output:
        drug_pickle=drug_output,
    params:
        node_feats=config["drugs"]["node_feats"],
        edge_feats=config["drugs"]["edge_feats"],
        max_num_atoms=config["drugs"]["max_num_atoms"],
    script:
        "../scripts/prepare_drugs.py"
