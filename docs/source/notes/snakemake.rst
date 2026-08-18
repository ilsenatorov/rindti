Snakemake workflows
===================

Introduction
------------

A workflow is a collection of rules that define the construction of the DTI dataset.
It should be run with the following command:


.. code:: console

  snakemake -j 1 --software-deployment-method conda --configfile your_config_file.yaml

Confuguration
-------------

The config files for the snakemake workflow are located in the ``config/snakemake/`` file.
`source` field is required to set the correct working directory, the results of the pipeline will be put in the ``results`` directory in the same folder.
Examples of files for most standard DTI datasets are provided in the aforementioned directory.

Affinity units and thresholds
-----------------------------

``parse_dataset.threshold`` is interpreted on the scale named by
``parse_dataset.unit``, and the two units point in opposite directions:

.. list-table::
   :header-rows: 1

   * - Dataset
     - ``unit``
     - ``threshold``
     - Positive when
   * - Davis, GLASS, BindingDB
     - ``nM``
     - 100
     - ``Y < threshold`` (lower Kd/Ki is a stronger interaction)
   * - KIBA
     - ``score``
     - 12.1
     - ``Y >= threshold`` (higher KIBA score is stronger)

.. Warning::
    Getting this wrong labels every interaction the same way. KIBA scores run
    0-17.2, so under the ``nM`` default every one of its 117,492 interactions was
    positive, ``posneg`` filtering then removed all of them, and the pipeline wrote
    an empty file without complaining. ``parse_dataset`` now refuses to emit an
    empty or single-class dataset, at each of binarization, filtering and sampling.

For ``nM`` datasets, affinities outside 1e-3 to 1e6 nM are dropped as measurement
artefacts, and the count is printed. GLASS contains 772 such rows out of 275,519.

Filtering and sampling
----------------------

``parse_dataset.filtering`` selects which interactions survive:

- ``all`` - keep everything.
- ``posneg`` - keep only drugs having at least one positive *and* one negative.
  This is aggressive on broad datasets: BindingDB Kd drops from 9,130 drugs to
  709, because most compounds there are measured against a single target.
- ``balanced`` - globally downsample the majority class.

``parse_dataset.sampling`` then rebalances per target: ``none``, ``under``, or
``over``.

.. Warning::
    ``over`` does not resample existing negatives. It takes interactions belonging
    to *other* targets, reassigns them to this one and relabels them negative -
    decoy generation, which injects false negatives whenever the drug does in fact
    bind. Prefer ``under`` or ``none``.

Features
--------

Proteins (``prots.features``):

- ``node_feats``: ``label`` or ``onehot`` residue identity.
- ``edge_feats``: ``none``, or ``distance`` to keep the Cα-Cα distance as a
  continuous edge attribute. Only the ``transformer`` node module consumes it -
  GIN, GAT and Cheb ignore edges, and FiLM expects discrete relation types.

Drugs (``drugs.node_feats``):

- ``label`` / ``onehot`` - element only, 14 features per atom.
- ``rich`` - adds degree, formal charge, hydrogen count, hybridisation,
  aromaticity, ring membership and chirality, for 47 features per atom.

Sweeps and ablations
--------------------

Any list value is expanded by :class:`rindti.utils.IterDict` into the full cross
product, one pipeline run per combination:

.. code:: console

  python run_snakemake.py config/snakemake/ablation.yaml --threads 8

``config/snakemake/ablation.yaml`` sweeps filtering, sampling, splitting and the
feature options above. Mind the cross product - it is 216 runs as shipped, each
building a complete dataset. For a one-factor-at-a-time ablation, vary a single
list and re-run per axis.

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

The resulting file will be ``<target>/results/prepare_all/wdonlnrancn_d1c1a34c.pkl``.
In this file, the first part (``wdonlnrancn``) is a human-readable compression of the config (``w`` for ``whole``, ``d`` for ``distance``, etc), while the second part (``d1c1a34c``) is a hash of the config.

.. Warning::
    Because the hash covers the whole config, changing any value produces a *new*
    file rather than rebuilding the old one. Stale outputs accumulate silently, and
    paths cannot be hardcoded - locate them by globbing the directory.

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
  └── split_data


DTI dataset creation
--------------------

In order to create a DTI dataset, the following requirements have to be met:

- PDB structures, located in the ``<source>/resources/structures`` directory
- Necessary tsv tables located in the  ``<source>/resources/tables`` directory:
  - ``<source>/resources/tables/inter.tsv`` -  The interactions data, has to contain *Drug_ID*, *Target_ID* and *Y* columns,
  - ``<source>/resources/tables/lig.tsv`` -  The ligand data, has to contain *Drug_ID* and *Drug* columns, where *Drug* contains SMILES representation of the drug.
  - ``<source>/resources/tables/prot.tsv`` -  The protein data, has to contain *Target_ID* and *Target* columns, where *Target* contains the protein sequence.

After running the pipeline with ``snakemake -j 16 --software-deployment-method conda --configfile your_config_file.yaml``, the pickle file should be created in ``<target>/results/prepare_all/`` folder.

File validation
^^^^^^^^^^^^^^^

The following code can be used to validate the configuration file:

.. code:: python

  from snakemake.utils import update_config, validate
  from rindti.utils import read_config

  default_config = read_config('config/snakemake/default.yaml')
  your_config = read_config('config/snakemake/your_config.yaml')
  update_config(default_config, your_config)  # recursive; dict.update is not
  validate(default_config, 'workflow/schemas/config.schema.yaml')

.. Warning::
    Use ``snakemake.utils.update_config``, not ``dict.update``. Dataset configs are
    overlays on ``default.yaml``, and a shallow update replaces whole subtrees - an
    overlay setting only ``prots.structs.method`` would silently delete
    ``prots.features``.

The following code can be used to validate the tables:

.. code:: python

  from snakemake.utils import validate
  import pandas as pd

  for i in ['inter', 'lig', 'prot']:
    df = pd.read_csv(f'test/test_data/resources/tables/{i}.tsv', sep='\t')
    validate(df, f'workflow/schemas/{i}.schema.yaml'.format(i))
