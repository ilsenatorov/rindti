rindti.models
=============

.. contents:: Contents
    :local:

Base Model
----------

The base model defines a lot of common methods that are identical for all models - logging, saving, etc.

.. autoclass:: rindti.models.base_model.BaseModel
   :members:
   :undoc-members:
   :exclude-members: training
   :show-inheritance:


DTI models
--------------------

Drug-target interaction prediction models.
Calculate embeddings for drugs and proteins, then use an MLP to predict the final result.

.. currentmodule:: rindti.models.dti
.. autosummary::
   :nosignatures:
   :recursive:


.. automodule:: rindti.models.dti
   :members:
   :undoc-members:
   :exclude-members: training, shared_step
   :show-inheritance:


Baseline models
----------------

These models predict the baseline values for datasets.
They have no access to the features and operate solely on the labels of drugs and proteins,
so they measure how much of a benchmark is solvable from label frequencies alone.
The pipeline runs them on every split it builds and writes
``results/baseline/<split name>.json``.

.. currentmodule:: rindti.models.dti.baseline
.. autosummary::
   :nosignatures:
   :recursive:


.. automodule:: rindti.models.dti.baseline
   :members:
   :undoc-members:
   :show-inheritance:


Metrics
----------------

The two metrics the affinity literature reports that torchmetrics does not provide,
plus the skill score that makes any of them interpretable against a baseline.

.. currentmodule:: rindti.models.metrics
.. autosummary::
   :nosignatures:
   :recursive:


.. automodule:: rindti.models.metrics
   :members:
   :undoc-members:
   :exclude-members: training
   :show-inheritance:
