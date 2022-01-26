Snakemake workflow
==================


A workflow is a collection of rules that define the construction both the DTI and the protein pretraining datasets.

In the very basic form, after formatting the config file in `config/snakemake.yaml`, you can run the pipeline by executing the following command:

.. code:: bash

        snakemake -j 1 --use-conda
