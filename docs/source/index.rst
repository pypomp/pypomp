.. pypomp documentation master file, created by
   sphinx-quickstart on Thu Dec  4 12:36:01 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pypomp's documentation!
==================================

**pypomp** is a Python package for modeling and inference using partially observed Markov process (POMP) models, also called state-space models (SSM) or hidden Markov models (HMM).

Key Features
------------

* Estimation, filtering, and inference for highly nonlinear, non-Gaussian state space models via the particle filter
* GPU support and just-in-time compilation via JAX, enabling significant speedups
* New algorithms for model-fitting with gradient descent using improved gradient estimates

Getting Started
---------------

The main classes in pypomp are:

* :class:`pypomp.Pomp` - Core POMP model class
* :class:`pypomp.PanelPomp` - Panel POMP models for multiple units

For more information, see the `tutorials <https://pypomp.github.io/tutorials>`_ and `quantitative tests <https://pypomp.github.io/quant>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
