Installation
============

RINDTI requires **Python 3.11 or newer**.

Installing the package
----------------------

.. code:: console

    git clone https://github.com/ilsenatorov/rindti
    cd rindti
    uv pip install -e ".[dev,workflow]"

On a machine without a GPU, add ``--torch-backend=cpu`` to pull much smaller wheels.

``pip`` works just as well if you prefer it:

.. code:: console

    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev,workflow]"

Optional dependency groups:

  - ``workflow`` - snakemake and the data-preparation pipeline
  - ``esm`` - protein language model features (``prots.features.method: esm``)
  - ``data`` - dataset download helpers (PyTDC, gdown)
  - ``baseline`` - XGBoost baselines
  - ``dev`` - pytest, ruff, pre-commit
  - ``docs`` - sphinx

External tools
--------------

One pipeline option relies on a program that cannot be installed from PyPI.

**PyMOL** is needed for ``prots.structs.method`` set to ``bsite``, ``template`` or
``plddt``. It is declared in ``workflow/envs/pymol.yaml``, so snakemake will build
the environment for you provided you pass:

.. code:: console

    snakemake --software-deployment-method conda ...

The default configuration (``structs.method: whole``, ``features.method: distance``)
does not require PyMOL.

Testing
-------

To check the installation:

.. code:: console

    pytest -m "not gpu and not snakemake"   # fast unit tests
    pytest -m "not gpu"                     # also runs the full pipeline

The ``snakemake``-marked tests build a conda environment for PyMOL the first time
they run, so expect the first invocation to be slow.
