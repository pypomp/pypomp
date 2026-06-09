pypomp documentation
====================

**Version:** |release| **Date:** |today|

**pypomp** is a Python package for modeling and inference using partially observed Markov process (POMP) models, also called state-space models (SSM) or hidden Markov models (HMM). Key features include:

* Estimation, filtering, and inference for nonlinear, non-Gaussian POMP models via the particle filter
* GPU support and just-in-time compilation via JAX, enabling significant speedups
* New algorithms for model-fitting with gradient descent using improved gradient estimates
* Support for both standard POMP models and panel POMP models
 
 
Installation
------------
 
You can install **pypomp** from PyPI:
 
.. code-block:: bash
 
    pip install pypomp              # install with core dependencies
    pip install pypomp[benchmarks]  # install with packages for benchmarking
    pip install pypomp[viz]         # install with plot dependencies

To install the latest development branch:

.. code-block:: bash

    pip install git+https://github.com/pypomp/pypomp.git
 
**pypomp** depends on **JAX**. 
To take full advantage of GPU acceleration, we highly recommend installing the GPU-enabled version of JAX. 
Please refer to the `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_ for detailed instructions specific to your system.

  
Getting started
---------------

* The `tutorials <https://pypomp.github.io/tutorials>`_ provide introductory explanations of POMP models and methods in pypomp.
* The `quantitative tests <https://pypomp.github.io/quant>`_ provide sample code used for benchmarking package performance.
* The `pypomp organization GitHub site <https://github.com/pypomp>`_ hosts source code and related projects.

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
