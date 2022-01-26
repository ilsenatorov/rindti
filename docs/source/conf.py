import datetime
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import rindti

# -- Project information -----------------------------------------------------
autosummary_generate = True
templates_path = ["_templates"]

source_suffix = ".rst"
master_doc = "index"


project = "RINDTI"
author = "Ilya Senatorov"
copyright = f"{datetime.datetime.now().year}, {author}"

ersion = rindti.__version__
release = rindti.__version__

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


html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": True,
    "navigation_depth": 2,
}

autodoc_member_order = "bysource"
intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/dev", None),
    "torch": ("https://pytorch.org/docs/master", None),
    "torch_geometric": ("https://pytorch-geometric.readthedocs.io/en/latest/", None),
}

rst_context = {"rindti": rindti}
add_module_names = False


def setup(app):
    """Docs setup."""

    def skip(app, what, name, obj, skip, options):
        """Ignore methods"""
        members = [
            "__init__",
            "__repr__",
            "__weakref__",
            "__dict__",
            "__module__",
        ]
        return True if name in members else skip

    def rst_jinja_render(app, docname, source):
        """Render Jinja templates."""
        src = source[0]
        rendered = app.builder.templates.render_string(src, rst_context)
        source[0] = rendered

    app.connect("autodoc-skip-member", skip)
    app.connect("source-read", rst_jinja_render)
