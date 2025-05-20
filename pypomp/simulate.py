import jax
import jax.numpy as jnp
from .internal_functions import _keys_helper


def simulate(
    pomp_obj=None,
    rinit=None,
    rproc=None,
    rmeas=None,
    ys=None,
    theta=None,
    covars=None,
    Nsim=1,
    key=None,
):
    """
    Simulates the evolution of a system over time using a Partially Observed
    Markov Process (POMP) model. This function can either execute on a POMP
    object or utilize the specified parameters directly to perform the
    simulation.

    Args:
        pomp_obj (Pomp, optional): An instance of the POMP class. If provided,
            the function will execute on this object to perform the simulation.
            If not provided, the necessary model components must be provided
            separately. Defaults to None.
        rinit (RInit, optional): Simulator for the initial-state distribution. Defaults
            to None.
        rproc (RProc, optional): Simulator for the process model. Defaults to None.
        rmeas (RMeas, optional): Simulator for the measurement model. Defaults to None.
        ys (array-like, optional): The measurement array. Defaults to None.
        theta (array-like, optional): Parameters involved in the POMP model.
            Defaults to None.
        covars (array-like, optional): Covariates for the process, or None if
            not applicable. Defaults to None.
        Nsim (int, optional): The number of simulations to perform. Defaults to
            100.
        key (jax.random.PRNGKey, optional): The random key for random number
            generation.

    Returns:
        tuple: Depending on the 'format' argument, returns either:
            - A tuple containing arrays of the simulated states, lower
              confidence intervals, and upper confidence intervals.
            - DataFrames of the simulated states, lower confidence intervals,
              and upper confidence intervals.
    """
    if pomp_obj is not None:
        X, Y = pomp_obj.simulate(Nsim=Nsim, key=key)
    elif (
        rinit is not None and rproc is not None and theta is not None and ys is not None
    ):
        X, Y = _simulate_internal(
            rinitializer=rinit.struct_pf,
            rprocess=rproc.struct_pf,
            rmeasure=rmeas.struct_pf,
            ys=ys,
            theta=theta,
            covars=covars,
            Nsim=Nsim,
            key=key,
        )
    return {"X": X, "Y": Y}


def _simulate_internal(rinitializer, rprocess, rmeasure, ys, theta, covars, Nsim, key):
    ylen = len(ys)
    key, keys = _keys_helper(key=key, J=Nsim, covars=covars)
    x_sims = rinitializer(theta, keys, covars)

    x_list = [None] * (ylen + 1)
    x_list[0] = x_sims
    y_list = [None] * ylen
    for i in range(ylen):
        key, *keys = jax.random.split(key, num=Nsim + 1)
        keys = jnp.array(keys)
        x_sims = rprocess(x_sims, theta, keys, covars)
        y_sims = rmeasure(x_sims, theta, keys, covars)
        x_list[i + 1] = x_sims
        y_list[i] = y_sims
    X = jnp.stack(x_list, axis=0)
    Y = jnp.stack(y_list, axis=0)
    return X, Y
