pypomp documentation
====================

**Version:** |release| **Date:** |today|

**pypomp** is a Python package for modeling and inference using partially observed Markov process (POMP) models, also called state-space models (SSM) or hidden Markov models (HMM). Key features include:

* Estimation, filtering, and inference for nonlinear, non-Gaussian POMP models via the particle filter
* GPU support and just-in-time compilation via JAX, enabling significant speedups
* New algorithms for model-fitting with gradient descent using improved gradient estimates
 
 
Installation
------------
 
You can install **pypomp** from PyPI:
 
.. code-block:: bash
 
    pip install pypomp
 
**pypomp** depends on **JAX**. 
To take full advantage of GPU acceleration, we highly recommend installing the GPU-enabled version of JAX. 
Please refer to the `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_ for detailed instructions specific to your system.


  
Getting started
---------------

* The `tutorials <https://pypomp.github.io/tutorials>`_ provide introductory explanations of POMP models and methods in pypomp.
* The `quantitative tests <https://pypomp.github.io/quant>`_ provide sample code used for benchmarking package performance.
* The `pypomp organization GitHub site <https://github.com/pypomp>`_ hosts source code and related projects.

The main classes in pypomp are:

* :class:`~pypomp.core.pomp.Pomp` - Core POMP model class
* :class:`~pypomp.panel.panel.PanelPomp` - Panel POMP models for multiple units


Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   Tutorials <https://pypomp.github.io/tutorials>
   model_components
   best_practices

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Resources

   Quantitative Tests <https://pypomp.github.io/quant>
   GitHub <https://github.com/pypomp>

   
Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
