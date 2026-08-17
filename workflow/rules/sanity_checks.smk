for path in sh.tables.values():
    assert os.path.isfile(path), f"Missing the file {path}"

if config["prots"]["structs"]["method"] in ["template", "bsite"]:
    assert os.path.isdir(sh._source("templates")), "Missing the templates directory"
    assert os.listdir(sh._source("templates")), "Templates directory is empty"

# The feature rules (distance_based / esm) all declare the same output, and
# `ruleorder` only expresses a preference: if the preferred rule's inputs cannot be
# produced, snakemake silently runs a different one and the protein featurisation
# method changes without warning. Fail here instead.
if config["prots"]["features"]["method"] != "esm":
    missing = [p for p in sh.raw_structs if not os.path.isfile(p)]
    assert not missing, (
        f"{len(missing)} of {len(sh.raw_structs)} structure files are missing, "
        f"e.g. {missing[:3]}. Without them the '{config['prots']['features']['method']}' "
        "rule cannot run and snakemake would fall back to a different featurisation."
    )
