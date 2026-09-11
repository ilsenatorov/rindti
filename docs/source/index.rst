Welcome to RINDTI's documentation!
====================================

:github_url: https://github.com/ilsenatorov/rindti

RINDTI Documentation
=====================

Welcome to RINDTI's documentation!
This is a collection of various models and utilities that are focused on using residue-level
protein contact graphs - one node per residue, edges between residues whose C-alpha atoms lie
within a distance cutoff - for predicting Drug-Target Interactions.



.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Notes

   notes/installation
   notes/quick_start
   notes/snakemake
   notes/data
   notes/training


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package Reference

   modules/models
   modules/layers
   modules/data
   modules/utils

.. autosummary::
   :toctree: _autosummary
   :recursive:

Indices and tables
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
