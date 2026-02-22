   
pypomp documentation
====================

**pypomp** is a Python package for modeling and inference using partially observed Markov process (POMP) models, also called state-space models (SSM) or hidden Markov models (HMM). Key features include:

* Estimation, filtering, and inference for nonlinear, non-Gaussian POMP models via the particle filter
* GPU support and just-in-time compilation via JAX, enabling significant speedups
* New algorithms for model-fitting with gradient descent using improved gradient estimates

  
Getting started
---------------

* The `tutorials <https://pypomp.github.io/tutorials>`_ provide introductory explanations of POMP models and methods in pypomp.
* The `quantitative tests <https://pypomp.github.io/quant>`_ provide sample code used for benchmarking package performance.
* The `pypomp organization GitHub site <https://github.com/pypomp>`_ hosts source code and related projects.

The main classes in pypomp are:

* :class:`pypomp.Pomp` - Core POMP model class
* :class:`pypomp.PanelPomp` - Panel POMP models for multiple units


Contents
--------

.. toctree::
   :maxdepth: 2

   api/index

   
Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
