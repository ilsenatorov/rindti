Snakemake workflows
===================

Introduction
------------

A workflow is a collection of rules that define the construction both the DTI and the protein pretraining datasets.
It should be run with the following command:


.. code:: console

        snakemake -j 1 --use-conda --configfile your_config_file.yaml

Confuguration
-------------

The config files for the snakemake workflow are located in the ``config/snakemake/`` file.
`source` and `target` fields are required to set the correct working directory.
Examples of files for most standard DTI datasets are provided in the aforementioned directory.
For most of the parameters changing the value and rerunning the pipeline from command line should be enough to generate a new file.

File structure
--------------

It is recommended to organise your datasets folder as following::

        datasets
        ├── dataset1
        │   └── resources
        │       ├── drugs
        │       ├── structures
        │       └── templates[optional]
        └── dataset2
        └── resources
                ├── drugs
                ├── structures
                └── templates[optional]

After running the snakemake workflow for dataset1 and dataset2, the following files and directories will be generated (the actual directories might differ, depending on your config)::

        datasets
        ├── dataset1
        │   ├── resources
        │   │   ├── drugs
        │   │   ├── structures
        │   │   └── templates
        │   └── results
        │       ├── distance_based
        │       ├── logs
        │       ├── parse_dataset
        │       ├── parsed_structures_template
        │       ├── prepare_drugs
        │       ├── pymol_template
        │       ├── split_data
        │       └── structure_info
        └── dataset2
        ├── resources
        │   ├── drugs
        │   ├── structures
        │   └── templates
        └── results
                ├── distance_based
                ├── logs
                ├── parse_dataset
                ├── parsed_structures_template
                ├── prepare_drugs
                ├── pymol_template
                ├── split_data
                └── structure_info


DTI dataset creation
--------------------

In order to create the DTI dataset, the following requirements have to be met:

- PDB structures, located in the ``<source>/resources/structures`` directory
- Drug interaction data, two tsv files labelled ``<source>/resources/drugs/inter.tsv`` and ``<source>/resources/drugs/ligs.tsv``. The interactions has to contain *Drug_ID*, *Target_ID* and *Y* columns, while the ligand one has to contain *Drug_ID* and *Drug* columns, where *Drug* contains SMILES representation of the drug.
- ``only_proteins`` entry in the snakemake config has to be *false*

After running the pipeline with ``snakemake -j 16 --use-conda --configfile your_config_file.yaml``, the pickle file should be created in ``<target>/results/prepare_all/`` folder.

Protein dataset creation
------------------------


In order to create the protein-only dataset (for pretraining), the following requirements have to be met:

- PDB structures, located in the ``<source>/resources/structures`` directory
- ``only_proteins`` entry in the snakemake config has to be *true*

After running the pipeline with ``snakemake -j 16 --use-conda``, the pickle file should be created in ``results/prepare_proteins/`` folder.
