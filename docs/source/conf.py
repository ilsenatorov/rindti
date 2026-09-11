import datetime
import doctest
import inspect
import os
import re
import sys

sys.path.append(os.path.abspath("../.."))

# Read the version out of the source rather than importing the package: the docs
# are built without the training stack installed (see autodoc_mock_imports).
_version_file = os.path.join(os.path.dirname(__file__), "..", "..", "rindti", "version.py")
with open(_version_file) as _f:
    _version = re.search(r'version = "([^"]+)"', _f.read()).group(1)

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

version = _version
release = _version

html_theme = "furo"

doctest_default_flags = doctest.NORMALIZE_WHITESPACE
autodoc_member_order = "bysource"

# torch and friends are several GB of wheels and are not needed to read the
# signatures. Modules that build type unions out of these names at import time
# need ``from __future__ import annotations``, or the union is attempted against
# a mock object and raises.
autodoc_mock_imports = [
    "git",
    "lightning",
    "matplotlib",
    "numpy",
    "pandas",
    "rdkit",
    "seaborn",
    "sklearn",
    "tensorboard",
    "torch",
    "torch_geometric",
    "torchmetrics",
    "yaml",
]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "torch_geometric": ("https://pytorch-geometric.readthedocs.io/en/latest/", None),
    "lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
}

add_module_names = False
fail_on_warning = True


def _fix_mocked_class_signature(app, what, name, obj, options, signature, return_annotation):
    """Recover ``__init__`` signatures for classes with a mocked base.

    Sphinx's mock objects define ``__new__``, and autodoc prefers ``__new__``
    over ``__init__``, so every class inheriting from a mocked ``nn.Module`` or
    ``Data`` would otherwise document as ``(*args: Any, **kwargs: Any)``.
    """
    if what != "class" or not inspect.isclass(obj) or "__init__" not in obj.__dict__:
        return None
    try:
        parameters = list(inspect.signature(obj.__init__).parameters.values())[1:]
    except (TypeError, ValueError):
        return None
    return str(inspect.Signature(parameters)), return_annotation


def setup(app):
    app.connect("autodoc-process-signature", _fix_mocked_class_signature)
