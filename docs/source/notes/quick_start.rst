Quick start guide
=================


Once you have succesfully installed the package, you can start using it.


Downloading the dataset
------------------------

For this tutorial we will work with the `GLASS dataset <https://zhanggroup.org/GLASS/>`_

To download it, we will use the following code:

.. code:: console

    uv run workflow/scripts/get_datasets.py glass --min_num_aa 250 --download_structures true


Since this is a dataset of GPCRs, we do not accept protein structures with less than 250 aminoacids.

This will create a new folder in ``datasets/glass/resources``, which will contain all the necessary information.

.. Note::
    ``get_datasets.py`` is run with ``uv run`` rather than ``python``. It needs
    PyTDC, which pins ``numpy<2`` and so cannot share an environment with the
    training stack, so the script declares its own dependencies inline (PEP 723)
    and runs isolated.

Running the Snakemake pipeline
------------------------------

Once our dataset is downloaded, we can run the snakemake pipline with the following command:

.. code:: console

    snakemake -j 10 --software-deployment-method conda --configfile config/snakemake/glass.yaml


This will create the final pickle file for the GLASS dataset, which will be located in the ``datasets/glass/results/prepare_all`` folder.

.. Note::
    Some parts of the config rely on more than just the base file downloaded in the previous step.


Running the training
---------------------

We can run the training script with the following code:

.. code:: console

    rindti-train config/dti/glass.yaml



This will start the training process, which will take a while.

You can monitor the progress of the training by running the following command:

.. code:: console

    tensorboard --logdir=tb_logs
