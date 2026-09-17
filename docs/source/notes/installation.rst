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

**MMseqs2** is needed for ``split_data.method: cluster_target``, which groups proteins
by sequence identity so that no test target is homologous to a training one. It is
declared in ``workflow/envs/mmseqs.yaml``, so snakemake will build the environment for
you provided you pass:

.. code:: console

    snakemake --software-deployment-method conda ...

The default configuration (``split_data.method: random``) does not require it, and
neither do the structure-parsing methods: ``bsite``, ``template`` and ``plddt`` are
implemented with ``biotite``, which comes from the ``workflow`` extra.

Testing
-------

To check the installation:

.. code:: console

    pytest -m "not gpu and not snakemake"   # fast unit tests
    pytest -m "not gpu"                     # also runs the full pipeline

The ``snakemake``-marked tests build a conda environment for MMseqs2 the first time
they reach the ``cluster_target`` split, so expect that one to be slow initially.
