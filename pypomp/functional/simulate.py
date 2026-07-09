import jax
import jax.numpy as jnp
from .structs import PompStruct
from ..core.algorithms.simulate import _jv_simulate_internal


def simulate(
    struct: PompStruct,
    thetas_array: jax.Array,
    nsim: int,
    keys: jax.Array,
    times: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Simulate latent states and observations from a POMP model struct.

    Pure-functional implementation intended for users who need to compose
    the simulation within custom JAX loops or higher-order functions.
    For the standard interface, see :meth:`pypomp.Pomp.simulate`.

    JAX vectorises the computation across parameter sets and simulation
    replicates simultaneously.

    Parameters
    ----------
    struct : PompStruct
        Compiled structural representation of the POMP model.  Obtain via
        :meth:`~pypomp.Pomp.to_struct`.
    thetas_array : jax.Array
        Parameter array of shape ``(n_reps, n_params)`` on the natural
        scale.  Must be aligned with ``struct.param_names``.
    nsim : int
        Number of independent simulation replicates.
    keys : jax.Array
        Random keys of shape ``(n_reps, ...)``.
    times : jax.Array or None, optional
        Custom observation times.  Defaults to ``struct.times``.

    Returns
    -------
    tuple of (jax.Array, jax.Array)
        - ``X_sims``: simulated states of shape
          ``(n_reps, nsim, len(times), n_states)``.
        - ``Y_sims``: simulated observations of shape
          ``(n_reps, nsim, len(times), n_obs)``.

    Notes
    -----
    To align and stack input parameter arrays into the correct
    canonical ordering, use :func:`pypomp.functional.align_params`.

    See Also
    --------
    pypomp.Pomp.simulate : Object-oriented interface.
    align_params : Parameter alignment utility.
    """

    _times = struct.times if times is None else times
    ydim = struct.ys.shape[1] if struct.ys is not None else 1

    X, Y = _jv_simulate_internal(
        struct.rinit_pf,
        struct.rproc_pf,
        struct.rmeas_pf,
        thetas_array,
        struct.t0,
        _times,
        struct.dt_array_extended,
        struct.nstep_array,
        ydim,
        struct.covars_extended,
        struct.accumvars,
        nsim,
        keys,
    )
    # Transpose from (n_reps, time, dim, nsim) to (n_reps, nsim, time, dim)
    return jnp.transpose(X, (0, 3, 1, 2)), jnp.transpose(Y, (0, 3, 1, 2))
