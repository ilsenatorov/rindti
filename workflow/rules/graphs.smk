rule rinerator:
    input:
        pdb=structures,
    output:
        sif=rinerator_output,
    log:
        target + "/logs/rinerator/{protein}.log",
    params:
        dir="{results}/rinerator_{structures}".format(results=target, structures=config["structures"]),
    message:
        "Running RINerator for {wildcards.protein}, logs are in {log}"
    shadow:
        "shallow"
    shell:
        """
        rinerator {input.pdb} {params.dir}/{wildcards.protein} > {log} 2>&1
        """


rule distance_based:
    input:
        pdbs=expand(structures, protein=targets),
    output:
        pickle=distance_protein_output,
    log:
        expand("{results}/logs/distance_based.log", results=target),
    params:
        threshold=config["distance"]["threshold"],
    message:
        "Running distance based network calculation, logs are in {log}"
    script:
        "scripts/distance_based.py"
