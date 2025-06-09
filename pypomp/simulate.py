"""
This file contains the internal simulation functions for POMP models.
"""

from functools import partial
import jax
import jax.numpy as jnp
from typing import Callable
from .internal_functions import _keys_helper
from .internal_functions import _interp_covars


@partial(jax.jit, static_argnums=(0, 1, 2, 6, 9))
def _simulate_internal(
    rinitializer: Callable,
    rprocess: Callable,
    rmeasure: Callable,
    theta: jax.Array,
    t0: float,
    times: jax.Array,
    ydim: int,
    covars: jax.Array | None,
    ctimes: jax.Array | None,
    nsim: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    ylen = len(times)
    key, keys = _keys_helper(key=key, J=nsim, covars=covars)
    covars_t = _interp_covars(t0, ctimes=ctimes, covars=covars)
    X_sims = rinitializer(theta, keys, covars_t, t0)

    X_array = jnp.zeros((ylen + 1, X_sims.shape[1], nsim))
    X_array = X_array.at[0].set(X_sims.T)
    Y_array = jnp.zeros((ylen, ydim, nsim))
    _simulate_helper2 = partial(
        _simulate_helper,
        times0=jnp.concatenate([jnp.array([t0]), times]),
        rprocess=rprocess,
        rmeasure=rmeasure,
        theta=theta,
        covars=covars,
        ctimes=ctimes,
        nsim=nsim,
    )
    X_sims, X_array, Y_array, key = jax.lax.fori_loop(
        lower=0,
        upper=ylen,
        body_fun=_simulate_helper2,
        init_val=(X_sims, X_array, Y_array, key),
    )

    return X_array, Y_array


def _simulate_helper(
    i: int,
    inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    times0: jax.Array,
    rprocess: Callable,
    rmeasure: Callable,
    theta: jax.Array,
    covars: jax.Array | None,
    ctimes: jax.Array | None,
    nsim: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    (X_sims, X_array, Y_array, key) = inputs
    t1 = times0[i]
    t2 = times0[i + 1]

    key, *keys = jax.random.split(key, num=nsim + 1)
    keys = jnp.array(keys)
    X_sims = rprocess(X_sims, theta, keys, ctimes, covars, t1, t2)

    covars_t = _interp_covars(t2, ctimes=ctimes, covars=covars)
    key, *keys = jax.random.split(key, num=nsim + 1)
    keys = jnp.array(keys)
    Y_sims = rmeasure(X_sims, theta, keys, covars_t, t2)

    X_array = X_array.at[i + 1].set(X_sims.T)
    Y_array = Y_array.at[i].set(Y_sims.T)
    return X_sims, X_array, Y_array, key
