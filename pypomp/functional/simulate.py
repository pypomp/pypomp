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
    """
    This is a pure functional implementation of the simulation algorithm, intended
    for users who need to compose it within custom JAX loops or higher-order
    functions. For a more user-friendly (but impurely-functional) interface, see
    :meth:`pypomp.core.pomp.Pomp.simulate`.

    This function propagates the system's latent state through time according to the
    process model (`rproc`) and generates corresponding simulated observations from
    the measurement model (`rmeas`).

    This implementation leverages JAX to efficiently vectorize the simulations across
    multiple parameter sets and simulation replicates simultaneously.

    Args:
        struct (PompStruct): The compiled structural representation of the POMP model.
        thetas_array (jax.Array): Array of initial parameters. Shape (n_reps, n_params).
        nsim (int): Number of simulations.
        keys (jax.Array): Random keys. Shape (n_reps, ...).
        times (jax.Array | None): Custom observation times. Defaults to struct.times.

    Returns:
        tuple[jax.Array, jax.Array]:
            X_sims: simulated states. Shape (n_reps, nsim, len(times), n_states)
            Y_sims: simulated observations. Shape (n_reps, nsim, len(times), n_obs)
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
