"""
This file contains the internal simulation functions for POMP models.
"""

from functools import partial
import jax
import jax.numpy as jnp
from typing import Callable
from .internal_functions import _keys_helper


@partial(jax.jit, static_argnums=(0, 1, 2, 7, 10))
def _simulate_internal(
    rinitializer: Callable,  # static
    rprocess_interp: Callable,  # static
    rmeasure: Callable,  # static
    theta: jax.Array,
    t0: float,
    times: jax.Array,
    dt_array_extended: jax.Array,
    ydim: int,  # static
    covars_extended: jax.Array | None,
    accumvars: tuple[int, ...] | None,
    nsim: int,  # static
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    times = times.astype(float)
    times0 = jnp.concatenate([jnp.array([t0]), times])

    covars0 = None if covars_extended is None else covars_extended[0]
    key, keys = _keys_helper(key=key, J=nsim, covars=covars0)
    X_sims = rinitializer(theta, keys, covars0, t0)

    n_obs = times.shape[0]
    X_array = jnp.zeros((n_obs + 1, X_sims.shape[1], nsim))
    X_array = X_array.at[0].set(X_sims.T)
    Y_array = jnp.zeros((n_obs, ydim, nsim))

    _simulate_helper2 = partial(
        _simulate_helper,
        rprocess_interp=rprocess_interp,
        rmeasure=rmeasure,
        theta=theta,
        times0=times0,
        dt_array_extended=dt_array_extended,
        covars_extended=covars_extended,
        nsim=nsim,
        accumvars=accumvars,
    )

    t, t_idx, X_sims, X_array, Y_array, key = jax.lax.fori_loop(
        lower=0,
        upper=n_obs,
        body_fun=_simulate_helper2,
        init_val=(t0, 0, X_sims, X_array, Y_array, key),
    )

    return X_array, Y_array


def _simulate_helper(
    i: int,
    inputs: tuple[jax.Array, int, jax.Array, jax.Array, jax.Array, jax.Array],
    times0: jax.Array,
    dt_array_extended: jax.Array,
    rprocess_interp: Callable,
    rmeasure: Callable,
    theta: jax.Array,
    covars_extended: jax.Array | None,
    accumvars: tuple[int, ...] | None,
    nsim: int,
) -> tuple[jax.Array, int, jax.Array, jax.Array, jax.Array, jax.Array]:
    (t, t_idx, X_sims, X_array, Y_array, key) = inputs

    key, keys = _keys_helper(key=key, J=nsim, covars=covars_extended)

    tol = jnp.sqrt(jnp.finfo(float).eps)
    nstep_dynamic = jnp.ceil(
        (times0[i + 1] - times0[i]) / dt_array_extended[t_idx] / (1 + tol)
    ).astype(int)

    X_sims, t_idx = rprocess_interp(
        X_sims,
        theta,
        keys,
        covars_extended,
        dt_array_extended,
        t,
        t_idx,
        nstep_dynamic,
        accumvars,
    )
    t = times0[i + 1]

    covars_t = None if covars_extended is None else covars_extended[t_idx]
    key, *keys = jax.random.split(key, num=nsim + 1)
    keys = jnp.array(keys)
    Y_sims = rmeasure(X_sims, theta, keys, covars_t, t)

    X_array = X_array.at[i + 1].set(X_sims.T)
    Y_array = Y_array.at[i].set(Y_sims.T)

    return t, t_idx, X_sims, X_array, Y_array, key
