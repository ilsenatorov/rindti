Installation
============


Conda installation
-------------------

One can install the packages necessary through conda (mamba) using the following commands:

.. code:: console

        conda install -c conda-forge mamba # mamba is much faster than conda
        mamba env create --quiet --name rindti --file workflow/envs/main.yml

Alternatively one can install only the packages for the neural network (without the snakemake pipeline) through

.. code:: console

        conda install -c conda-forge mamba # mamba is much faster than conda
        mamba env create --quiet --name rindti --file workflow/envs/torch.yml

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

Rinerator
---------

RINerator is used to calculate the RINs of proteins. Currently it is not publicly available for installation.
