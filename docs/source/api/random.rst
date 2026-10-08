Fast Random Number Generation on GPUs
======================================

The simulators included in the JAX package often suffer from warp divergence due to using rejection sampling.
This problem is especially pronounced when running particle filtering methods with hundreds of thousands of particles being run in parallel across method replications.
To address this, Pypomp includes replacement functions that use Inverse Transform Sampling to generate random variables.
These functions use JAX under the hood, so they can be used in a JIT-compiled context.
While the following functions include some branching in order to handle edge cases, the performance loss from warp divergence is minimal.


Random Variate Generators
-------------------------

.. currentmodule:: pypomp.random

.. autosummary::
   :toctree: generated/

   fast_poisson
   fast_binomial
   fast_multinomial
   fast_gamma
   fast_nbinomial

Inverse Cumulative Distribution Functions (CDFs)
------------------------------------------------

.. autosummary::
   :toctree: generated/

   poissoninv
   binominv
   gammainv

DPOP Log-Weights
----------------

Log-weight increments for a DPOP process model to add to its ``_logw`` state (see :func:`pypomp.functional.pop`).
Each has the gradient of the draw's log-probability in the distribution's parameters, with the draw held fixed.

.. autosummary::
   :toctree: generated/

   poisson_logw
   binomial_logw
   multinomial_logw
   euler_multinomial_logw
