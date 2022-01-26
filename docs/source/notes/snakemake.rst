Snakemake workflows
===================

Introduction
------------

A workflow is a collection of rules that define the construction both the DTI and the protein pretraining datasets.


.. code:: console

        snakemake -j 1 --use-conda

Confuguration
-------------

The config file for the snakemake workflow is located in the ``config/snakemake.yaml`` file.
For most of the parameters changing the value and rerunning the pipeline from command line should be enough to generate a new file.

DTI dataset creation
--------------------

In order to create the DTI dataset, the following requirements have to be met:

- PDB structures, located in the ``resources/structures`` directory
- Drug interaction data, two tsv files labelled ``resources/drugs/inter.tsv`` and ``resources/drugs/ligs.tsv``. The interactions has to contain *Drug_ID*, *Target_ID* and *Y* columns, while the ligand one has to contain *Drug_ID* and *Drug* columns, where *Drug* contains SMILES representation of the drug.
- ``only_proteins`` entry in the ``config/snakemake.yaml`` has to be *false*

After running the pipeline with ``snakemake -j 16 --use-conda``, the pickle file should be created in ``results/prepare_all/`` folder.

Protein dataset creation
------------------------


In order to create the protein-only dataset (for pretraining), the following requirements have to be met:

- PDB structures, located in the ``resources/structures`` directory
- ``only_proteins`` entry in the ``config/snakemake.yaml`` has to be *true*

After running the pipeline with ``snakemake -j 16 --use-conda``, the pickle file should be created in ``results/prepare_proteins/`` folder.
