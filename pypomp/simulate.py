from functools import partial
import jax
import jax.numpy as jnp
import xarray as xr
from .internal_functions import _keys_helper


def simulate(
    rinit=None,
    rproc=None,
    rmeas=None,
    theta=None,
    ylen=None,
    covars=None,
    Nsim=1,
    key=None,
):
    """
    Simulates the evolution of a system over time using a Partially Observed
    Markov Process (POMP) model.

    Args:
        rinit (RInit, optional): Simulator for the initial-state distribution. Defaults
            to None.
        rproc (RProc, optional): Simulator for the process model. Defaults to None.
        rmeas (RMeas, optional): Simulator for the measurement model. Defaults to None.
        theta (array-like, optional): Parameters involved in the POMP model.
            Defaults to None.
        ylen (int, optional): Number of observations to generate in one time series.
        covars (array-like, optional): Covariates for the process, or None if
            not applicable. Defaults to None.
        Nsim (int, optional): The number of simulations to perform. Defaults to 1.
        key (jax.random.PRNGKey, optional): The random key for random number
            generation.

    Returns:
        dict: A dictionary of simulated values. 'X' contains the unobserved values
            whereas 'Y' contains the observed values as JAX arrays. In each case, the
            first dimension is the observation index, the second indexes the element of
            the observation vector, and the third is the simulation number.
    """
    if rinit is None or rproc is None or rmeas is None or theta is None or ylen is None:
        raise ValueError("Invalid arguments given to simulate")

    X_sims, Y_sims = _simulate_internal(
        rinitializer=rinit.struct_pf,
        rprocess=rproc.struct_pf,
        rmeasure=rmeas.struct_pf,
        theta=theta,
        ylen=ylen,
        covars=covars,
        Nsim=Nsim,
        key=key,
    )
    X_sims = xr.DataArray(X_sims, dims=["t", "i", "sim"])
    Y_sims = xr.DataArray(Y_sims, dims=["t", "i", "sim"])
    return {"X_sims": X_sims, "Y_sims": Y_sims}


def _simulate_internal(
    rinitializer, rprocess, rmeasure, theta, ylen, covars, Nsim, key
):
    key, keys = _keys_helper(key=key, J=Nsim, covars=covars)
    x_sims = rinitializer(theta, keys, covars)

    x_list = [None] * (ylen + 1)
    x_list[0] = x_sims
    y_list = [None] * ylen
    for i in range(ylen):
        key, *keys = jax.random.split(key, num=Nsim + 1)
        keys = jnp.array(keys)
        x_sims = rprocess(x_sims, theta, keys, covars)

        key, *keys = jax.random.split(key, num=Nsim + 1)
        keys = jnp.array(keys)
        y_sims = rmeasure(x_sims, theta, keys, covars)

        x_list[i + 1] = x_sims
        y_list[i] = y_sims
    X = jnp.swapaxes(jnp.stack(x_list, axis=0), 1, 2)
    Y = jnp.swapaxes(jnp.stack(y_list, axis=0), 1, 2)
    return X, Y
