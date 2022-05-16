Installation
============


Conda installation
-------------------

One can install the packages necessary through conda (mamba) using the following commands:

.. code:: console

        conda install -c conda-forge mamba # mamba is much faster than conda
        mamba env create --quiet --name rindti --file workflow/envs/main.yml


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

Rinerator
---------

RINerator is used to calculate the RINs of proteins. Currently it is not publicly available for installation.

Testing
-------

In order to asses whether the installation of the packages was succesfull, please run ``pytest`` in the root directory.
If the packages were not installed into path using pip, please use ``python -m pytest`` instead.
