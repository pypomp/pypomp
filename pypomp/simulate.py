"""
This file contains the internal simulation functions for POMP models.
"""

from functools import partial
import jax
import jax.numpy as jnp
from typing import Callable
from .internal_functions import _keys_helper


@partial(jax.jit, static_argnums=(0, 1, 2, 6, 9, 12))
def _simulate_internal(
    rinitializer: Callable,
    rprocess: Callable,
    rmeasure: Callable,
    theta: jax.Array,
    t0: float,
    times: jax.Array,
    ylen: int,
    ys_observed: jax.Array,
    dt_array_extended: jax.Array,
    ydim: int,
    covars_extended: jax.Array | None,
    accumvars: jax.Array | None,
    nsim: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    times = times.astype(float)
    covars_t = None if covars_extended is None else covars_extended[0]
    key, keys = _keys_helper(key=key, J=nsim, covars=covars_t)
    X_sims = rinitializer(theta, keys, covars_t, t0)

    X_array = jnp.zeros((ylen + 1, X_sims.shape[1], nsim))
    X_array = X_array.at[0].set(X_sims.T)
    Y_array = jnp.zeros((ylen, ydim, nsim))
    _simulate_helper2 = partial(
        _simulate_helper,
        rprocess=rprocess,
        rmeasure=rmeasure,
        theta=theta,
        times=times,
        ys_observed=ys_observed,
        dt_array_extended=dt_array_extended,
        covars_extended=covars_extended,
        nsim=nsim,
        accumvars=accumvars,
    )
    t, X_sims, X_array, Y_array, key, _ = jax.lax.fori_loop(
        lower=0,
        upper=ys_observed.shape[0],
        body_fun=_simulate_helper2,
        init_val=(t0, X_sims, X_array, Y_array, key, 0),
    )

    return X_array, Y_array


def _simulate_helper(
    i: int,
    inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, int],
    times: jax.Array,
    ys_observed: jax.Array,
    dt_array_extended: jax.Array,
    rprocess: Callable,
    rmeasure: Callable,
    theta: jax.Array,
    covars_extended: jax.Array | None,
    accumvars: jax.Array | None,
    nsim: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, int]:
    (t, X_sims, X_array, Y_array, key, obs_idx) = inputs

    key, *keys = jax.random.split(key, num=nsim + 1)
    keys = jnp.array(keys)
    covars_t = None if covars_extended is None else covars_extended[i]
    X_sims = rprocess(X_sims, theta, keys, covars_t, t, dt_array_extended[i])
    t = t + dt_array_extended[i]

    def with_observation(X_sims, X_array, Y_array, key, obs_idx, t):
        t = times[obs_idx]
        covars_t = None if covars_extended is None else covars_extended[i + 1]
        key, *keys = jax.random.split(key, num=nsim + 1)
        keys = jnp.array(keys)
        Y_sims = rmeasure(X_sims, theta, keys, covars_t, t)
        X_array = X_array.at[obs_idx + 1].set(X_sims.T)
        Y_array = Y_array.at[obs_idx].set(Y_sims.T)
        X_sims = jnp.where(
            accumvars is not None, X_sims.at[:, accumvars].set(0.0), X_sims
        )
        obs_idx = obs_idx + 1
        return X_sims, X_array, Y_array, key, obs_idx, t

    def without_observation(X_sims, X_array, Y_array, key, obs_idx, t):
        return X_sims, X_array, Y_array, key, obs_idx, t

    X_sims, X_array, Y_array, key, obs_idx, t = jax.lax.cond(
        ys_observed[i],
        with_observation,
        without_observation,
        *(X_sims, X_array, Y_array, key, obs_idx, t),
    )

    return t, X_sims, X_array, Y_array, key, obs_idx
