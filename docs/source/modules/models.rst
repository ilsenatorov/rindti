rindti.models
=============

DTI models
--------------------

.. currentmodule:: rindti.models.dti
.. autosummary::


.. automodule:: rindti.models.dti
   :members:
   :undoc-members:
   :exclude-members: training, shared_step

Pretraining models
------------------

.. currentmodule:: rindti.models.pretrain
.. autosummary::

   {% for cls in rindti.models.pretrain.classes %}
     {{ rindti.cls }}
   {% endfor %}

.. automodule:: rindti.models.pretrain
   :members:
   :undoc-members:
   :exclude-members: training, shared_step
