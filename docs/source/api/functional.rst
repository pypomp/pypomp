Functional API
==============

The ``pypomp.functional`` module provides a collection of pure, stateless JAX functions for model simulation and inference. 
While the object-oriented :class:`~pypomp.core.pomp.Pomp` class is recommended for most users, the functional API is intended for advanced users who need to:

1.  **Compose algorithms** within custom JAX loops, scan, or higher-order functions.
2.  **Perform end-to-end differentiation** of the entire algorithm (e.g., using ``jax.grad`` on ``mop``).

PompStruct
----------

To use the functional API, you must first export your model's structural data and compiled functions into a :class:`~pypomp.functional.structs.PompStruct`. 
This can be done using the :meth:`pypomp.core.pomp.Pomp.to_struct` method.

.. currentmodule:: pypomp.functional

.. autosummary::
   :toctree: generated/

   PompStruct

Core Algorithms
---------------

.. autosummary::
   :toctree: generated/

   pfilter
   mif
   simulate

Differentiable Particle Filtering
---------------------------------

These functions are primarily used for gradient-based parameter estimation. 
``mop`` and ``dpop`` are designed to be fully differentiable with respect to the model parameters (``thetas_array``).

.. autosummary::
   :toctree: generated/

   train
   mop
   dpop
