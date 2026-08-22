Installation
============


Conda installation
-------------------

One can install the packages necessary through conda (mamba) using the following commands:

.. code:: console

    conda install -c conda-forge mamba # mamba is much faster than conda
    mamba env create --quiet --name rindti --file workflow/envs/main.yaml


Then one can run optionally run `pip install .` in the root directory of the repository to install rindti as a package.

Manual installation
-------------------

In order to use this module, you must first install the following packages (preferably in the order listed here):

  - pytorch
  - torch_geometric
  - pytorch_lightning
  - snakemake
  - rdkit
  - seaborn
  - plotly

Then one can run optionally ``pip install .`` in the root directory of the repository to install rindti as a package.


Testing
-------

In order to asses whether the installation of the packages was succesfull, please run ``pytest`` in the root directory.
If the packages were not installed into path using pip, please use ``python -m pytest`` instead.
Furthermore, if your device has no GPU support, please use ``pytest -m "not gpu"``


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



