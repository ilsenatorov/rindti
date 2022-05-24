if not config["only_prots"]:
    for path in sh.tables.values():
        assert os.path.isfile(path), f"Missing the file {path}"

if config["prots"]["structs"]["method"] in ["template", "bsite"]:
    assert os.path.isdir(sh._source("templates")), "Missing the templates directory"
    assert os.listdir(sh._source("templates")), "Templates directory is empty"
