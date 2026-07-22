Proposal Distributions
======================

.. currentmodule:: pypomp

Proposal objects are used by PMCMC and ABC-MCMC methods to generate parameter
updates inside JAX-compiled Metropolis-Hastings loops.

.. rubric:: Protocol Interface
.. autosummary::
   :toctree: generated/
   :template: autosummary/proposal_class.rst

   Proposal

.. rubric:: Proposal Classes
.. autosummary::
   :toctree: generated/
   :template: autosummary/proposal_class.rst

   MVNDiagRW
   MVNRWFull
   MVNRWAdaptive
