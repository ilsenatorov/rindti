Quick start guide
=================


Once you have succesfully installed the package, you can start using it.


Downloading the dataset
------------------------

For this tutorial we will work with the `GLASS dataset <https://zhanggroup.org/GLASS/>`_

To download it, we will use the following code:

.. code:: console
  python workflow/scripts/get_datasets.py glass --min_num_aa 250

Since this is a dataset of GPCRs, we do not accept protein structures with less than 250 aminoacids.

This will create a new folder in ``datasets/glass/resources``, which will contain all the necessary information.

Running the Snakemake pipeline
------------------------------

Once our dataset is downloaded, we can run the snakemake pipline with the following command:

.. code:: console
  snakemake -j 10 --use-conda --configfile config/snakemake/glass.yaml

This will create the final pickle file for the GLASS dataset, which will be located in ``datasets/glass/resources/prepare_all`` folder.

.. Note::
  Some parts of the config rely on more than just the base file downloaded in the previous step.


Running the training
---------------------

We can run the training script with the following code:

.. code:: console
  python train.py config/dti/glass.yaml


This will start the training process, which will take a while.

You can monitor the progress of the training by running the following command:

.. code:: console
  tensorboard --logdir=tb_logs
