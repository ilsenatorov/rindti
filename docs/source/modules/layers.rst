rindti.layers
=============


.. contents:: Contents
    :local:


Base layer
----------


.. autoclass:: rindti.layers.base_layer.BaseLayer
   :members:
   :undoc-members:
   :exclude-members: training
   :show-inheritance:

Node layers
--------------------

.. currentmodule:: rindti.layers.graphconv
.. autosummary::
    :nosignatures:

    ChebConvNet
    FilmConvNet
    GatConvNet
    GINConvNet
    PNAConvNet
    TransformerNet


.. automodule:: rindti.layers.graphconv
   :members:
   :undoc-members:
   :exclude-members: training
   :show-inheritance:

Pooling layers
------------------

.. currentmodule:: rindti.layers.graphpool
.. autosummary::
   :nosignatures:

    DiffPoolNet
    GMTNet
    MeanPool


.. automodule:: rindti.layers.graphpool
   :members:
   :undoc-members:
   :exclude-members: training
   :show-inheritance:


Other layers
------------------

.. currentmodule:: rindti.layers.other
.. autosummary::
   :nosignatures:

    MLP
    MutualInformation
    SequenceEmbedding


.. automodule:: rindti.layers.other
   :members:
   :undoc-members:
   :exclude-members: training
   :show-inheritance:
