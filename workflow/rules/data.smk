final_output = sh._target("prepare_all", sh.namer(config) + ".pkl")
cluster_name = sh.namer(config["split_data"]["cluster"])


rule parse_dataset:
    input:
        inter=sh.tables["inter"],
    output:
        inter=sh._target("parse_dataset", sh.namer(config["parse_dataset"]) + ".tsv"),
    script:
        "../scripts/parse_dataset.py"


# Clustering is expensive and seed-independent, splitting is cheap and reseeded, so
# they are separate rules: a re-split does not re-cluster.
rule prot_fasta:
    input:
        seqs=sh.tables["prot"],
    output:
        fasta=sh._target("cluster_prots", cluster_name + ".fasta"),
        mapping=sh._target("cluster_prots", cluster_name + "_ids.tsv"),
    script:
        "../scripts/prot_fasta.py"


rule mmseqs_cluster:
    input:
        fasta=rules.prot_fasta.output.fasta,
    output:
        clusters=sh._target("cluster_prots", cluster_name + "_mmseqs_cluster.tsv"),
    params:
        prefix=lambda w, output: output.clusters[: -len("_cluster.tsv")],
        tmp=lambda w, output: output.clusters + "_tmp",
        min_seq_id=config["split_data"]["cluster"]["prot_identity"],
    log:
        sh._target("cluster_prots", cluster_name + "_mmseqs.log"),
    conda:
        "../envs/mmseqs.yaml"
    shell:
        # --threads 1 and an explicit cluster mode: mmseqs is otherwise free to vary
        # its output with the thread count, and the split seed would then no longer
        # describe the experiment.
        "mmseqs easy-cluster {input.fasta} {params.prefix} {params.tmp} "
        "--min-seq-id {params.min_seq_id} -c 0.8 --cov-mode 0 --cluster-mode 0 "
        "--threads 1 -v 1 > {log} 2>&1"


rule cluster_prots:
    input:
        clusters=rules.mmseqs_cluster.output.clusters,
        mapping=rules.prot_fasta.output.mapping,
    output:
        clusters=sh._target("cluster_prots", cluster_name + ".tsv"),
    script:
        "../scripts/cluster_prots.py"


rule cluster_drugs:
    input:
        smiles=sh.tables["lig"],
    output:
        clusters=sh._target("cluster_drugs", cluster_name + ".tsv"),
    params:
        cutoff=config["split_data"]["cluster"]["drug_similarity"],
    script:
        "../scripts/cluster_drugs.py"


def split_inputs() -> dict:
    """Inputs for `split_data`, which depend on the split method.

    Declared conditionally so that a random split does not pull the MMseqs2 conda
    environment into the DAG and build it for nothing.
    """
    inputs = {"inter": rules.parse_dataset.output.inter}
    method = config["split_data"]["method"]
    if method == "cluster_target":
        inputs["clusters"] = rules.cluster_prots.output.clusters
    elif method == "cluster_drug":
        inputs["clusters"] = rules.cluster_drugs.output.clusters
    elif method == "target":
        # Sequences, for exact-duplicate deduplication. They are not in `inter`:
        # parse_dataset leaves only Drug_ID, Target_ID and Y.
        inputs["seqs"] = sh.tables["prot"]
    return inputs


rule split_data:
    input:
        **split_inputs(),
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
