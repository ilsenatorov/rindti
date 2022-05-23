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
`source` field is required to set the correct working directory, the results of the pipeline will be put in the ``results`` directory in the same folder.
Examples of files for most standard DTI datasets are provided in the aforementioned directory.

File naming
-----------

The resulting files are named according the the string entries in the config file, followed by a string representing a hashed config dictionary.
While this decreases human readability, it is necessary to ensure that the once the config changes, the results are not overwritten.

For example, given the following config::
  prots:
    structs:
      method: whole
    features:
      method: distance
      node_feats: onehot
      edge_feats: none
  drugs:
    max_num_atoms: 150
    node_feats: label
    edge_feats: none
  split_data:
    method: random
  parse_dataset:
    filtering: all
    sampling: none
    task: class

The resulting file will be ``<target>/results/prepare_all/wdonlnranc_0f6b0ac6.pkl``.
In this file, the first part (``wdonlnranc``) is human-readable compression of the config (``w`` for ``whole``, ``d`` for ``distance``, etc), while the second part (``0f6b0ac6``) is a hashed version of the config.

File structure
--------------

It is recommended to organise your datasets folder as following::

  dataset1
  └── resources
  ├── structures
  │   ├── struct1.pdb
  │   ├── struct2.pdb
  │   ├── struct3.pdb
  ├── tables
  │   ├── inter.tsv
  │   ├── lig.tsv
  │   └── prot.tsv
  └── templates
          └── template1.pdb

After running the snakemake workflow for dataset1 and dataset2, the following files and directories will be generated (the actual directories might differ, depending on your config)::

  test_data
  ├── resources
  │   ├── structures
  │   ├── tables
  │   └── templates
  └── results
  ├── parse_dataset
  ├── parsed_structs
  ├── prepare_all
  ├── prepare_drugs
  ├── prot_data
  ├── pymol_logs
  ├── pymol_scripts
  ├── rinerator
  └── split_data


DTI dataset creation
--------------------

In order to create a DTI dataset, the following requirements have to be met:

- PDB structures, located in the ``<source>/resources/structures`` directory
- Necessary tsv tables tsv files located in the  ``<source>/resources/tables`` directory:
  - ``<source>/resources/tables/inter.tsv`` -  The interactions data, has to contain *Drug_ID*, *Target_ID* and *Y* columns,
  - ``<source>/resources/tables/lig.tsv`` -  The ligand data, has to contain *Drug_ID* and *Drug* columns, where *Drug* contains SMILES representation of the drug.
  - ``<source>/resources/tables/prot.tsv`` -  The protein data, has to contain *Target_ID* and *Target* columns, where *Target* contains the protein sequence.
- ``only_proteins`` entry in the snakemake config has to be *false*

After running the pipeline with ``snakemake -j 16 --use-conda --configfile your_config_file.yaml``, the pickle file should be created in ``<target>/results/prepare_all/`` folder.

Protein dataset creation
------------------------


In order to create the protein-only dataset (for pretraining), the following requirements have to be met:

- PDB structures, located in the ``<source>/resources/structures`` directory
- ``only_proteins`` entry in the snakemake config has to be *true*

After running the pipeline with ``snakemake -j 16 --use-conda``, the pickle file should be created in ``<target>/results/prot_data/`` folder.
