Model Mechanics
===============

The following components define the core behavior of a POMP model.
Instead of interacting with internal wrapper classes, users provide functions to a :class:`~pypomp.core.pomp.Pomp` object following the specifications below.
The :class:`~pypomp.core.pomp.Pomp` object will fail to initialize if these functions do not strictly
adhere to the specifications.
This ensures that the arguments are internally mapped to the correct names in the function definition.

.. _rinit-tutorial:

State Initialization (rinit)
----------------------------

The ``rinit`` function defines the initialization process for the state variables at time :math:`t_0`.
It receives parameters, a PRNG key, covariates, and the initial time, and must return
a dictionary mapping state names to their initial values.

**Argument Binding:**
You can define the function arguments in two ways:

1. **By Name:** Use the exact names ``theta_``, ``key``, ``covars``, and ``t0``, in that order.
2. **By Type:** Label arguments with the types :data:`~pypomp.types.ParamDict`, :data:`~pypomp.types.RNGKey`, :data:`~pypomp.types.CovarDict`, and :data:`~pypomp.types.InitialTimeFloat`, in any order.

**Template:**

.. code-block:: python

    from pypomp.types import ParamDict, RNGKey, CovarDict, InitialTimeFloat

    def rinit(
        params: ParamDict,
        key: RNGKey,
        covars: CovarDict,
        t0: InitialTimeFloat
    ) -> dict:
        """
        Returns initial state dictionary.
        """
        # Access parameters by name
        S_0 = params['S_0']

        # Return dict with ALL state variables
        return {'S': S_0, 'I': 1.0, 'R': 0.0}

.. _rproc-tutorial:

State Transition (rproc)
------------------------

The ``rproc`` function defines the process model (state transitions).
It performs a single Euler step, receiving the current state, parameters, PRNG key, covariates, current time, and step size.

**Argument Binding:**
You can define the function arguments in two ways:

1. **By Name:** Use the exact parameter names ``X_``, ``theta_``, ``key``, ``covars``, ``t``, ``dt``, in that order.
2. **By Type:** Label arguments with the types :data:`~pypomp.types.StateDict`, :data:`~pypomp.types.ParamDict`, :data:`~pypomp.types.RNGKey`, :data:`~pypomp.types.CovarDict`, :data:`~pypomp.types.TimeFloat`, and :data:`~pypomp.types.StepSizeFloat`, in any order.

**Template:**

.. code-block:: python

    from pypomp.random import fast_poisson
    from pypomp.types import StateDict, ParamDict, RNGKey, CovarDict, TimeFloat, StepSizeFloat

    def rproc(
        state: StateDict,
        params: ParamDict,
        key: RNGKey,
        covars: CovarDict,
        t: TimeFloat,
        dt: StepSizeFloat
    ) -> dict:
        """
        Returns the new state after time step `dt`.
        """
        rate = params['beta'] * state['I']
        n_events = fast_poisson(key, rate * dt)

        new_S = state['S'] - n_events
        new_I = state['I'] + n_events

        return {'S': new_S, 'I': new_I}

.. _vectorized-rproc:

Manually Vectorized rproc (CPU optimization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, pypomp applies :func:`jax.vmap` to ``rproc``, so the function above is written for a *single* particle.
Decorating ``rproc`` with :func:`~pypomp.vectorized` disables that vmap: the function is then called once per Euler step with every state entry as a ``(J,)`` array, and is responsible for all ``J`` particles itself.

This sometimes improves performance on CPU, and it is entirely optional.
On GPU, the decorator is not expected to make a difference.

**Contract:**

- Each value in the state dict is a ``(J,)`` array, and the returned dict must
  hold ``(J,)`` arrays (scalars are broadcast).
- Parameters and covariates are shared across particles and stay scalar --
  except under :meth:`~pypomp.Pomp.mif`, where parameters are perturbed per
  particle and arrive as ``(J,)`` arrays. Writing elementwise code (rather than
  e.g. ``jnp.dot`` over parameter vectors) handles both cases.
- ``key`` is a **single** PRNG key for the whole step, not one key per particle.
  Draw for all particles at once with an explicit shape.
- Infer ``J`` from the state rather than hardcoding it, so the function still
  composes when pypomp maps over replicates.

**Template:**

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from pypomp import vectorized
    from pypomp.types import StateDict, ParamDict, RNGKey, CovarDict, TimeFloat, StepSizeFloat

    @vectorized
    def rproc(
        state: StateDict,
        params: ParamDict,
        key: RNGKey,
        covars: CovarDict,
        t: TimeFloat,
        dt: StepSizeFloat
    ) -> dict:
        """
        Returns the new state after time step `dt`, for all J particles.
        """
        S, I = state['S'], state['I']
        J = S.shape[0]

        # One batched draw for every particle, rather than J scalar draws.
        dw = jax.random.normal(key, (J,)) * jnp.sqrt(dt)
        infections = params['beta'] * S * I * dt + dw

        return {'S': S - infections, 'I': I + infections}

.. note::

   A vectorized ``rproc`` consumes randomness differently from its scalar
   counterpart, so log-likelihoods will not match a scalar implementation
   draw-for-draw even with the same seed. They agree in distribution, and
   deterministic dynamics agree exactly.

.. _dpop-rproc:

Process Models for DPOP (experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gradient-based training with :meth:`~pypomp.Pomp.train` differentiates through ``rproc``.
When the process draws discrete counts, such as Poisson, binomial or multinomial transitions, its sample paths are not differentiable in the parameters, and the gradient with respect to the transition parameters is zero.
DPOP corrects this by adding the score of the transitions' log-density to the particle weights.
It is enabled with ``dpop=True`` in :meth:`~pypomp.Pomp.train` and :meth:`~pypomp.PanelPomp.train`, or by calling :func:`~pypomp.functional.pop` in place of :func:`~pypomp.functional.mop`.

**Contract:**

- The model has a state named ``_logw``, which ``rinit`` sets to ``0.0``.
  It is reset to zero at every observation time, so it need not be listed in ``accumvars``.
- ``rproc`` adds to ``_logw`` the log-weight of every random draw whose distribution depends on the parameters and that has no pathwise gradient.
  :func:`~pypomp.random.poisson_logw`, :func:`~pypomp.random.binomial_logw`, :func:`~pypomp.random.nbinomial_logw`, :func:`~pypomp.random.multinomial_logw` and :func:`~pypomp.random.euler_multinomial_logw` compute these.
  Each returns a surrogate with the gradient of the draw's log-probability, holding the draw fixed.
- The draws themselves carry no gradient.
  The Poisson, binomial, negative binomial and multinomial samplers in :mod:`pypomp.random` already satisfy this; wrap draws from other samplers in :func:`jax.lax.stop_gradient`, or their gradient is counted twice.
- Draws with a pathwise gradient, such as :func:`~pypomp.random.fast_gamma` noise, are differentiated directly and need no log-weight.
- The model's gradients are finite. NaN gradients are not masked.

**Template:**

.. code-block:: python

    from pypomp.random import fast_poisson, poisson_logw
    from pypomp.types import StateDict, ParamDict, RNGKey, CovarDict, TimeFloat, StepSizeFloat

    def rproc(
        state: StateDict,
        params: ParamDict,
        key: RNGKey,
        covars: CovarDict,
        t: TimeFloat,
        dt: StepSizeFloat
    ) -> dict:
        rate = params['beta'] * state['I']
        n_events = fast_poisson(key, rate * dt)

        return {
            'S': state['S'] - n_events,
            'I': state['I'] + n_events,
            '_logw': state['_logw'] + poisson_logw(n_events, rate * dt),
        }

The helpers broadcast, so they also work in a :ref:`vectorized rproc <vectorized-rproc>`.
The multinomial helpers take categories along the last axis, as the samplers return them; :func:`~pypomp.random.euler_multinomial_logw` also takes tuples of per-event arrays.
:func:`pypomp.models.sir` and the ``"001d"`` and ``"002d"`` variants of :class:`~pypomp.models.UKMeasles` are complete examples.

.. _dmeas-tutorial:

Measurement Density (dmeas)
---------------------------

The ``dmeas`` function calculates the log-likelihood of the data given the state.
It must return a **scalar** (float or 0-d JAX array).

**Argument Binding:**
You can define the function arguments in two ways:

1. **By Name:** Use the exact parameter names ``Y_``, ``X_``, ``theta_``, ``covars``, ``t``, in that order.
2. **By Type:** Label arguments with the types :data:`~pypomp.types.ObservationDict`, :data:`~pypomp.types.StateDict`, :data:`~pypomp.types.ParamDict`, :data:`~pypomp.types.CovarDict`, and :data:`~pypomp.types.TimeFloat`, in any order.

**Template:**

.. code-block:: python

    import jax.scipy.stats as stats
    from pypomp.types import ObservationDict, StateDict, ParamDict, CovarDict, TimeFloat

    def dmeas(
        data: ObservationDict,
        state: StateDict,
        params: ParamDict,
        covars: CovarDict,
        t: TimeFloat
    ) -> float:
        """
        Returns scalar log-likelihood.
        """
        # Expected cases based on state
        mu = state['I'] * params['rho']

        # Log-likelihood of observed data
        lik = stats.poisson.logpmf(data['cases'], mu)

        return lik

.. _rmeas-tutorial:

Measurement Simulator (rmeas)
-----------------------------

The ``rmeas`` function simulates a single observation from the current state.
It must return a **dictionary** mapping observation names to their simulated values.

**Argument Binding:**
You can define the function arguments in two ways:

1. **By Name:** Use the exact parameter names ``X_``, ``theta_``, ``key``, ``covars``, ``t``, in that order.
2. **By Type:** Label arguments with the types :data:`~pypomp.types.StateDict`, :data:`~pypomp.types.ParamDict`, :data:`~pypomp.types.RNGKey`, :data:`~pypomp.types.CovarDict`, and :data:`~pypomp.types.TimeFloat`, in any order.

**Template:**

.. code-block:: python

    import jax.numpy as jnp
    from pypomp.random import fast_poisson
    from pypomp.types import StateDict, ParamDict, RNGKey, CovarDict, TimeFloat

    def rmeas(
        state: StateDict,
        params: ParamDict,
        key: RNGKey,
        covars: CovarDict,
        t: TimeFloat
    ) -> dict:
        """
        Returns simulated observation dictionary.
        """
        mu = state['I'] * params['rho']
        sim_cases = fast_poisson(key, mu)

        # Return dict mapping observation names to simulated values
        return {'cases': sim_cases}

.. _dprior-tutorial:

Prior Log-Density (dprior)
--------------------------

The ``dprior`` function evaluates the log-prior density given parameter values.
It must return a **scalar** (float or 0-d JAX array).

**Argument Binding:**
You can define the function arguments in two ways:

1. **By Name:** Use the exact parameter name ``theta_``.
2. **By Type:** Label the argument with the type :data:`~pypomp.types.ParamDict`.

**Template:**

.. code-block:: python

    import jax.scipy.stats as stats
    from pypomp.types import ParamDict

    def dprior(params: ParamDict) -> float:
        """
        Returns scalar log-prior density.
        """
        return stats.norm.logpdf(params['beta'], loc=1.0, scale=0.5)
