rindti.models
=============

.. contents:: Contents
    :local:

Base Model
----------

.. autoclass:: rindti.models.base_model.BaseModel
   :members:
   :undoc-members:
   :exclude-members: training
   :show-inheritance:

Encoder
----------

.. autoclass:: rindti.models.encoder.Encoder
   :members:
   :undoc-members:
   :exclude-members: training
   :show-inheritance:

DTI models
--------------------

.. currentmodule:: rindti.models.dti
.. autosummary::
   :nosignatures:

   ClassificationModel
   RegressionModel


.. automodule:: rindti.models.dti
   :members:
   :undoc-members:
   :exclude-members: training, shared_step
   :show-inheritance:

Pretraining models
------------------

.. currentmodule:: rindti.models.pretrain
.. autosummary::
   :nosignatures:


   BGRLModel
   GraphLogModel
   InfoGraphModel
   PfamModel

.. automodule:: rindti.models.pretrain
   :members:
   :undoc-members:
   :exclude-members: training, shared_step, training_step, training_epoch_end
   :show-inheritance:
