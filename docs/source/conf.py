import datetime
import doctest
import os
import sys

sys.path.append(os.path.abspath("../.."))

import rindti

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "_templates"]

source_suffix = ".rst"
master_doc = "index"

author = "Ilya Senatorov"
project = "RINDTI"
copyright = f"{datetime.datetime.now().year}, {author}"

version = rindti.__version__
release = rindti.__version__

html_theme = "furo"

doctest_default_flags = doctest.NORMALIZE_WHITESPACE
autodoc_member_order = "bysource"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "torch_geometric": ("https://pytorch-geometric.readthedocs.io/en/latest/", None),
    "lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
}

rst_context = {"rindti": rindti}

add_module_names = False
fail_on_warning = True
