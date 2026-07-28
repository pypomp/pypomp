"""
This file contains the internal simulation functions for POMP models.
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit

SHOULD_TRANS = False  # Should transformations be applied to the parameters?


def _simulate_internal(
    rinitializer: Callable,  # static
    rprocess_interp: Callable,  # static
    rmeasure: Callable,  # static
    theta: jax.Array,
    t0: float,
    times: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    ydim: int,  # static
    covars_extended: jax.Array | None,
    accumvars: tuple[int, ...] | None,
    nsim: int,  # static
    key: jax.Array,
    should_trans: bool = SHOULD_TRANS,  # argument allows abc, pmcmc to transform
) -> tuple[jax.Array, jax.Array]:
    times = times.astype(float)
    times0 = jnp.concatenate([jnp.array([t0]), times])

    covars0 = None if covars_extended is None else covars_extended[0]
    split_keys = jax.random.split(key, num=nsim + 1)
    key = split_keys[0]
    keys = split_keys[1:]
    X_sims = rinitializer(theta, keys, covars0, t0, should_trans)

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
        nstep_array=nstep_array,
        covars_extended=covars_extended,
        nsim=nsim,
        accumvars=accumvars,
        should_trans=should_trans,
    )

    t, t_idx, X_sims, X_array, Y_array, key = jax.lax.fori_loop(
        lower=0,
        upper=n_obs,
        body_fun=_simulate_helper2,
        init_val=(t0, 0, X_sims, X_array, Y_array, key),
    )

    return X_array, Y_array


_vmapped_simulate_internal = jax.vmap(
    _simulate_internal,
    in_axes=(None,) * 3 + (0,) + (None,) * 8 + (0,),
)

_jit_simulate_internal = jit(
    _simulate_internal,
    static_argnames=(
        "rinitializer",
        "rprocess_interp",
        "rmeasure",
        "ydim",
        "nsim",
        "should_trans",
    ),
)

_jv_simulate_internal = jit(
    _vmapped_simulate_internal,
    static_argnames=(
        "rinitializer",
        "rprocess_interp",
        "rmeasure",
        "ydim",
        "nsim",
        "should_trans",
    ),
)


def _simulate_helper(
    i: int,
    inputs: tuple[jax.Array, int, jax.Array, jax.Array, jax.Array, jax.Array],
    times0: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    rprocess_interp: Callable,
    rmeasure: Callable,
    theta: jax.Array,
    covars_extended: jax.Array | None,
    accumvars: tuple[int, ...] | None,
    nsim: int,
    should_trans: bool = SHOULD_TRANS,
) -> tuple[jax.Array, int, jax.Array, jax.Array, jax.Array, jax.Array]:
    (t, t_idx, X_sims, X_array, Y_array, key) = inputs

    split_keys = jax.random.split(key, num=nsim + 1)
    key = split_keys[0]
    keys = split_keys[1:]

    nstep = nstep_array[i].astype(int)

    X_sims, t_idx = rprocess_interp(
        X_sims,
        theta,
        keys,
        covars_extended,
        dt_array_extended,
        t,
        t_idx,
        nstep,
        accumvars,
        should_trans,
    )
    t = times0[i]

    covars_t = None if covars_extended is None else covars_extended[t_idx]
    split_keys = jax.random.split(key, num=nsim + 1)
    key = split_keys[0]
    keys = split_keys[1:]
    Y_sims = rmeasure(X_sims, theta, keys, covars_t, t, should_trans)

    X_array = X_array.at[i + 1].set(X_sims.T)
    Y_array = Y_array.at[i].set(Y_sims.T)

    return t, t_idx, X_sims, X_array, Y_array, key
