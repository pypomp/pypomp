Proposal Distributions
======================

.. currentmodule:: pypomp.proposals

Proposal objects are used by PMCMC and ABC-MCMC methods to generate parameter
updates inside JAX-compiled Metropolis-Hastings loops.

.. rubric:: Constructors
.. autosummary::
   :toctree: generated/

   mvn_diag_rw
   mvn_rw
   mvn_rw_adaptive

.. rubric:: Classes
.. autosummary::
   :toctree: generated/

   MVNDiagRW
   MVNRWFull
   MVNRWAdaptive
