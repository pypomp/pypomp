Functional API
==============

The ``pypomp.functional`` module provides a collection of pure, stateless JAX functions for model simulation and inference.
While the object-oriented :class:`~pypomp.core.pomp.Pomp` class is recommended for most users, the functional API is intended for advanced users who need to compose algorithms within custom JAX loops, scan, or higher-order functions.

Data Structs
------------

To use the functional API, you must first export your model's structural data and compiled functions into a :class:`~pypomp.functional.structs.PompStruct` or :class:`~pypomp.functional.structs.PanelPompStruct`.
This can be done using the :meth:`~pypomp.core.pomp.Pomp.to_struct` or :meth:`~pypomp.panel.panel.PanelPomp.to_struct` methods.

.. currentmodule:: pypomp.functional

.. autosummary::
   :toctree: generated/

   PompStruct
   PanelPompStruct

Core Algorithms
---------------

.. autosummary::
   :toctree: generated/

   pfilter
   panel_pfilter
   mif
   panel_mif
   simulate

Differentiable Particle Filtering
---------------------------------

These functions are primarily used for gradient-based parameter estimation.
``mop`` is designed to be fully differentiable with respect to the model parameters.
``pop`` (experimental) is ``mop`` with the score of the process log-density added to the particle weights, for process models that are not differentiable in the parameters.

.. autosummary::
   :toctree: generated/

   train
   panel_train
   mop
   pop

Bayesian Inference
------------------

Particle MCMC (PMMH) and ABC-MCMC samplers. Proposals come from :doc:`proposals`.

.. autosummary::
   :toctree: generated/

   pmcmc
   abc

Utilities
---------

.. autosummary::
   :toctree: generated/

   align_params
