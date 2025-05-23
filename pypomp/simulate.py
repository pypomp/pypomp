from functools import partial
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
        covars=jnp.array(covars) if covars is not None else None,
        Nsim=Nsim,
        key=key,
    )
    # TODO: Add state names as coords
    X_sims = xr.DataArray(X_sims, dims=["time", "element", "sim"])
    Y_sims = xr.DataArray(Y_sims, dims=["time", "element", "sim"])
    return {"X_sims": X_sims, "Y_sims": Y_sims}


@partial(jax.jit, static_argnums=(0, 1, 2, 4, 6))
def _simulate_internal(
    rinitializer, rprocess, rmeasure, theta, ylen, covars, Nsim, key
):
    key, keys = _keys_helper(key=key, J=Nsim, covars=covars)
    x_sims = rinitializer(theta, keys, covars)

    x_list = jnp.zeros((ylen + 1, *x_sims.shape))
    x_list = x_list.at[0].set(x_sims)
    y_list = jnp.zeros((ylen, *rmeasure(x_sims, theta, keys, covars).shape))
    _simulate_helper2 = partial(
        _simulate_helper,
        rprocess=rprocess,
        rmeasure=rmeasure,
        theta=theta,
        covars=covars,
        Nsim=Nsim,
    )
    x_sims, x_list, y_list, key = jax.lax.fori_loop(
        lower=0,
        upper=ylen,
        body_fun=_simulate_helper2,
        init_val=(x_sims, x_list, y_list, key),
    )

    X = jnp.swapaxes(jnp.stack(x_list, axis=0), 1, 2)
    Y = jnp.swapaxes(jnp.stack(y_list, axis=0), 1, 2)
    return X, Y


def _simulate_helper(i, inputs, rprocess, rmeasure, theta, covars, Nsim):
    (x_sims, x_list, y_list, key) = inputs
    key, *keys = jax.random.split(key, num=Nsim + 1)
    keys = jnp.array(keys)
    x_sims = rprocess(x_sims, theta, keys, covars)

    key, *keys = jax.random.split(key, num=Nsim + 1)
    keys = jnp.array(keys)
    y_sims = rmeasure(x_sims, theta, keys, covars)
    x_list = x_list.at[i + 1].set(x_sims)
    y_list = y_list.at[i].set(y_sims)
    return x_sims, x_list, y_list, key
