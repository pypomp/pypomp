Model Structure Components
==========================

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
2. **By Type:** Label arguments with the types ``ParamDict``, ``RNGKey``, ``CovarDict``, ``InitialTimeFloat``, in any order.

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

The ``rproc`` function defines the process model (state transitions). It performs a 
**single Euler step**, receiving the current state, parameters, PRNG key, covariates, 
current time, and step size.

**Argument Binding:**
You can define the function arguments in two ways:

1. **By Name:** Use the exact parameter names ``X_``, ``theta_``, ``key``, ``covars``, ``t``, ``dt``, in that order.
2. **By Type:** Label arguments with the types ``StateDict``, ``ParamDict``, ``RNGKey``, ``CovarDict``, ``TimeFloat``, ``StepSizeFloat``, in any order.

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

.. _dmeas-tutorial:

Measurement Density (dmeas)
---------------------------

The ``dmeas`` function calculates the log-likelihood of the data given the state. 
It must return a **scalar** (float or 0-d JAX array).

**Argument Binding:**
You can define the function arguments in two ways:

1. **By Name:** Use the exact parameter names ``Y_``, ``X_``, ``theta_``, ``covars``, ``t``, in that order.
2. **By Type:** Label arguments with the types ``ObservationDict``, ``StateDict``, ``ParamDict``, ``CovarDict``, ``TimeFloat``, in any order.

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

The ``rmeas`` function simulates a single observation vector from the current state. 
It must return a 1D **JAX Array** (not a dictionary).

**Argument Binding:**
You can define the function arguments in two ways:

1. **By Name:** Use the exact parameter names ``X_``, ``theta_``, ``key``, ``covars``, ``t``, in that order.
2. **By Type:** Label arguments with the types ``StateDict``, ``ParamDict``, ``RNGKey``, ``CovarDict``, ``TimeFloat``, in any order.

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
    ) -> jax.Array:
        """
        Returns simulated data array of shape (ydim,).
        """
        mu = state['I'] * params['rho']
        sim_cases = fast_poisson(key, mu)

        # Return array, e.g., [cases, deaths]
        return jnp.array([sim_cases])
