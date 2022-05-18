### CHECK IF templates ARE PRESENT ###
templates_dir = sh._source("templates")
if osp.isdir(templates_dir):
    templates = [os.path.join(templates_dir, x) for x in os.listdir(templates_dir) if x.endswith(".pdb")]
else:
    if not config["only_prots"] and config["structures"] not in ["whole", "plddt"]:
        raise ValueError("No templates available")
    templates = []
if not config["only_prots"] and not osp.isdir(sh._source("drugs")):
    raise ValueError("No drug interaction data available, can't calculate final data!")
### CHECK IF gnomad is available ###
# if osp.isfile(sh._source("gnomad.csv")):
#     gnomad = sh.source("gnomad.csv")
# else:
#     gnomad = []
### CHECK IF drug data is available ###
