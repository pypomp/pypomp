from functools import partial
import jax
import jax.numpy as jnp
import pandas as pd
import xarray as xr
from typing import Callable
from .model_struct import RInit, RProc, RMeas
from .internal_functions import _keys_helper
from .internal_functions import _interp_covars


def simulate(
    rinit: RInit,
    rproc: RProc,
    rmeas: RMeas,
    theta: dict,
    times: jax.Array,
    key: jax.Array,
    covars: pd.DataFrame | None = None,
    Nsim: int = 1,
):
    """
    Simulates the evolution of a system over time using a Partially Observed
    Markov Process (POMP) model.

    Args:
        rinit (RInit): Simulator for the initial-state distribution.
        rproc (RProc): Simulator for the process model.
        rmeas (RMeas): Simulator for the measurement model.
        theta (array-like): Parameters involved in the POMP model.
        times (jax.Array): Times at which to generate observations.
        key (jax.Array): The random key for random number
            generation.
        covars (pandas.DataFrame, optional): Covariates for the process, or None if
            not applicable. Defaults to None.
        Nsim (int): The number of simulations to perform. Defaults to 1.


    Returns:
        dict: A dictionary of simulated values. 'X' contains the unobserved values
            whereas 'Y' contains the observed values as xarrays. In each case, the
            first dimension is the observation index, the second indexes the element of
            the observation vector, and the third is the simulation number.
    """
    if not isinstance(theta, dict):
        raise TypeError("theta must be a dictionary")
    if not all(isinstance(val, float) for val in theta.values()):
        raise TypeError("Each value of theta must be a float")

    X_sims, Y_sims = _simulate_internal(
        rinitializer=rinit.struct_pf,
        rprocess=rproc.struct_pf,
        rmeasure=rmeas.struct_pf,
        theta=jnp.array(list(theta.values())),
        t0=rinit.t0,
        times=jnp.array(times),
        ydim=rmeas.ydim,  # TODO: update simulate to not require ydim
        covars=jnp.array(covars) if covars is not None else None,
        ctimes=jnp.array(covars.index) if covars is not None else None,
        Nsim=Nsim,
        key=key,
    )
    # TODO: Add state names as coords
    X_sims = xr.DataArray(
        X_sims,
        dims=["time", "element", "sim"],
        coords={"time": jnp.concatenate([jnp.array([rinit.t0]), jnp.array(times)])},
    )
    Y_sims = xr.DataArray(
        Y_sims, dims=["time", "element", "sim"], coords={"time": times}
    )
    return {"X_sims": X_sims, "Y_sims": Y_sims}


@partial(jax.jit, static_argnums=(0, 1, 2, 6, 9))
def _simulate_internal(
    rinitializer: Callable,
    rprocess: Callable,
    rmeasure: Callable,
    theta: jax.Array,
    t0: float,
    times: jax.Array,
    ydim: int,
    covars: jax.Array,
    ctimes: jax.Array,
    Nsim: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    ylen = len(times)
    key, keys = _keys_helper(key=key, J=Nsim, covars=covars)
    covars_t = _interp_covars(t0, ctimes=ctimes, covars=covars)
    X_sims = rinitializer(theta, keys, covars_t, t0)

    X_array = jnp.zeros((ylen + 1, X_sims.shape[1], Nsim))
    X_array = X_array.at[0].set(X_sims.T)
    Y_array = jnp.zeros((ylen, ydim, Nsim))
    _simulate_helper2 = partial(
        _simulate_helper,
        times0=jnp.concatenate([jnp.array([t0]), times]),
        rprocess=rprocess,
        rmeasure=rmeasure,
        theta=theta,
        covars=covars,
        ctimes=ctimes,
        Nsim=Nsim,
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
    covars: jax.Array,
    ctimes: jax.Array,
    Nsim: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    (X_sims, X_array, Y_array, key) = inputs
    t1 = times0[i]
    t2 = times0[i + 1]

    key, *keys = jax.random.split(key, num=Nsim + 1)
    keys = jnp.array(keys)
    X_sims = rprocess(X_sims, theta, keys, ctimes, covars, t1, t2)

    covars_t = _interp_covars(t2, ctimes=ctimes, covars=covars)
    key, *keys = jax.random.split(key, num=Nsim + 1)
    keys = jnp.array(keys)
    Y_sims = rmeasure(X_sims, theta, keys, covars_t, t2)

    X_array = X_array.at[i + 1].set(X_sims.T)
    Y_array = Y_array.at[i].set(Y_sims.T)
    return X_sims, X_array, Y_array, key
