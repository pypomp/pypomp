import jax
import jax.numpy as jnp
import xarray as xr
from .internal_functions import _keys_helper


def simulate(
    rinit,
    rproc,
    rmeas,
    theta,
    ylen,
    covars=None,
    Nsim=1,
    key=None,
):
    """
    Simulates the evolution of a system over time using a Partially Observed
    Markov Process (POMP) model.

    Args:
        rinit (RInit): Simulator for the initial-state distribution.
        rproc (RProc): Simulator for the process model.
        rmeas (RMeas): Simulator for the measurement model.
        theta (array-like): Parameters involved in the POMP model.
        ylen (int): Number of observations to generate in one time series.
        covars (array-like, optional): Covariates for the process, or None if
            not applicable. Defaults to None.
        Nsim (int, optional): The number of simulations to perform. Defaults to 1.
        key (jax.random.PRNGKey, optional): The random key for random number
            generation.

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
        ylen=ylen,
        covars=covars,
        Nsim=Nsim,
        key=key,
    )
    X_sims = xr.DataArray(X_sims, dims=["time", "element", "sim"])
    Y_sims = xr.DataArray(Y_sims, dims=["time", "element", "sim"])
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
