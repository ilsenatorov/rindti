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

Docker installation
-------------------

A pre-built Docker image containing RINDTI and its Python dependencies is available from Docker Hub:

`RINDTI Docker images <https://hub.docker.com/r/atsh00001/rindti/tags>`_

The ``cuda128`` tag is intended for GPU systems whose NVIDIA driver supports CUDA 12.8. Before using the image, verify that Docker is installed:

.. code:: console

    docker --version

Pull the image from Docker Hub:

.. code:: console

    docker pull atsh00001/rindti:cuda128

The ``latest`` tag may also be used:

.. code:: console

    docker pull atsh00001/rindti:latest

However, explicitly selecting ``cuda128`` is recommended for reproducible GPU environments.

Testing the Docker image
~~~~~~~~~~~~~~~~~~~~~~~~

Verify that PyTorch and PyTorch Lightning can be imported:

.. code:: console

    docker run --rm -it atsh00001/rindti:cuda128 \
        python -c "import torch; import pytorch_lightning; print('ok')"

The expected output is:

.. code:: text

    ok

On a machine with the NVIDIA Container Toolkit and a compatible GPU, verify that CUDA is available inside the container:

.. code:: console

    docker run --rm --gpus all -it atsh00001/rindti:cuda128 \
        python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

For the ``cuda128`` image, the expected CUDA version and availability are:

.. code:: text

    12.8
    True

Opening an interactive shell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To inspect the image or run commands interactively:

.. code:: console

    docker run --rm -it atsh00001/rindti:cuda128 /bin/bash

The RINDTI source code is installed under:

.. code:: text

    /opt/rindti

Inside the container, the training entry point can be inspected with:

.. code:: console

    cd /opt/rindti
    python train.py --help
