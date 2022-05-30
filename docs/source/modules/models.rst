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

These models are used to predict the baseline values for datasets.
They do not have access to the actual features and operate solely on the labels of drugs and proteins.

.. currentmodule:: rindti.models.dti.baseline
.. autosummary::
   :nosignatures:
   :recursive:


.. automodule:: rindti.models.dti.baseline
   :members:
   :undoc-members:
   :show-inheritance:
