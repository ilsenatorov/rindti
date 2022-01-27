import datetime
import doctest

import sphinx_rtd_theme

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

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

doctest_default_flags = doctest.NORMALIZE_WHITESPACE
autodoc_member_order = "bysource"
intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/dev", None),
    "torch": ("https://pytorch.org/docs/master", None),
    "torch_geometric": ("https://pytorch-geometric.readthedocs.io/en/latest/", None),
    "pytorch_lightning": ("https://pytorch-lightning.readthedocs.io/en/latest/", None),
}

html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": True,
    "navigation_depth": 2,
}

rst_context = {"rindti": rindti}

add_module_names = False
fail_on_warning = True
