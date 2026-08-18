Training and experiments
========================

Running a single model
----------------------

``config/dti/base.yaml`` holds the shared training settings. Rather than copying it
per experiment, override values on the command line:

.. code:: console

    rindti-train config/dti/base.yaml \
        --set datamodule.filename=datasets/davis/results/prepare_all/<hash>.pkl \
        --set datamodule.exp_name=davis_random

``--set`` is repeatable and takes a dotted path. Values are parsed as YAML, so
``0.001``, ``true``, ``null`` and ``[a, b]`` all arrive with the right type. Only
keys that already exist can be set, so a typo fails immediately rather than being
silently ignored.

Because the pipeline names its outputs by config hash, the pickle path is not
predictable - glob the directory rather than hardcoding it.

Metrics
-------

Classification reports accuracy, AUROC, average precision (AUPRC) and Matthews
correlation coefficient. Regression reports MAE, MSE, explained variance, Pearson
and Spearman.

.. Note::
    AUPRC is the one to read on these datasets. Davis is roughly 7% positive, and
    AUROC flatters heavily at that imbalance. Spearman is the rank-based analogue
    of the concordance index that affinity-regression papers quote.

Sweeps and ablations
--------------------

Any list value in the training config is expanded into one run per combination,
each repeated over ``runs`` seeds:

.. code:: yaml

    model:
      feat_method: [concat, element_l1]
      prot:
        node:
          module: [ginconv, gatconv]

That is four configurations. ``config/dti/ablation.yaml`` sweeps the merge method,
convolution and pooling; as shipped it is 40 configurations over 3 seeds.

Each configuration logs to its own directory tagged with what it changed, for
example ``node.module=gatconv-pool.module=diffpool``. Tags use the shortest
unambiguous suffix of the config path, so ``node.module`` and ``pool.module`` stay
distinguishable.

Collecting results
------------------

Lightning writes a ``hparams.yaml`` beside every run. The collector walks the log
tree, joins metrics to those settings, and keeps only the hyperparameters that
actually vary - the columns an ablation is grouped by:

.. code:: console

    python -m rindti.utils.results --logdir tb_logs --output results.csv --summary true

.. code::

    hp:model,feat_method hp:model,prot,node,module     mean      std  n
                  concat                   ginconv 0.416667 0.117851  2
              element_l1                   gatconv 0.750000 0.353553  2

``--reduction`` chooses how each metric's epoch series collapses to one number:
``last`` (default), ``best_max`` for metrics where higher is better, or
``best_min`` for losses. The same functions are importable:

.. code:: python

    from rindti.utils import collect, summarise

    results = collect("tb_logs", reduction="best_max")
    table = summarise(results)

Baselines
---------

Frequency-prior baselines use no features at all, and expose how much of a score
comes from popularity bias rather than from the drug and protein graphs:

.. code:: console

    python -m rindti.models.dti.baseline.run \
        prot_drug_max datasets/davis/results/split_data/<hash>.tsv

They report on the held-out ``test`` split and return mean/std over ``n_runs``
repeats. On Davis this prints something like::

    Results (test, n=1)   ACC: 0.928   AUC: 0.843   AUPRC: 0.363   MCC: 0.066

which is worth internalising before reading any model score: knowing nothing but
how often each drug and target appears already gives 93% accuracy and 0.84 AUROC
on a dataset that is 7% positive. AUPRC and MCC are the metrics that show it.
