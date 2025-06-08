"""
This module implements the MOP algorithm for POMP models.
"""

from functools import partial
import jax
import jax.numpy as jnp
from jax import jit
import pandas as pd
from typing import Callable
from .model_struct import RInit, RProc, DMeas
from .internal_functions import _keys_helper
from .internal_functions import _normalize_weights
from .internal_functions import _resampler
from .internal_functions import _interp_covars


def mop(
    J: int,
    key: jax.Array,
    rinit: RInit,
    rproc: RProc,
    dmeas: DMeas,
    theta: dict,
    ys: pd.DataFrame,
    covars: pd.DataFrame | None = None,
    alpha: float = 0.97,
) -> float:
    """
    Computes the mean negative log-likelihood using particle filtering.

    Args:
        J (int): Number of particles to use in the filter.
        key (jax.Array): Random key for reproducibility.
        rinit (RInit): Simulator for the initial-state distribution.
        rproc (RProc): Simulator for the process model.
        dmeas (DMeas): Density evaluator for the measurement model.
        theta (dict): Parameters involved in the POMP model. Each value must be a float.
        ys (pd.DataFrame): Observed measurements with time index.
        covars (pd.DataFrame, optional): Covariates for the process. Defaults to None.
        alpha (float, optional): Systematic resampling threshold. Defaults to 0.97.

    Returns:
        float: The log-likelihood estimate
    """
    if J < 1:
        raise ValueError("J should be greater than 0")

    if not isinstance(theta, dict):
        raise TypeError("theta must be a dictionary")
    if not all(isinstance(val, float) for val in theta.values()):
        raise TypeError("Each value of theta must be a float")

    return -_mop_internal(
        theta=jnp.array(list(theta.values())),
        t0=rinit.t0,
        times=jnp.array(ys.index),
        ys=jnp.array(ys),
        J=J,
        rinitializer=rinit.struct_pf,
        rprocess=rproc.struct_pf,
        dmeasure=dmeas.struct_pf,
        ctimes=jnp.array(covars.index) if covars is not None else None,
        covars=jnp.array(covars) if covars is not None else None,
        alpha=alpha,
        key=key,
    )


@partial(jit, static_argnums=(4, 5, 6, 7))
def _mop_internal(
    theta: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
    dmeasure: Callable,  # static
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    alpha: float,
    key: jax.Array,
) -> float:
    """
    Internal function for the MOP algorithm, which calls function 'mop_helper'
    iteratively.
    """
    key, keys = _keys_helper(key=key, J=J, covars=covars)
    covars_t = _interp_covars(t0, ctimes=ctimes, covars=covars)
    particlesF = rinitializer(theta, keys, covars_t, t0)
    weightsF = jnp.log(jnp.ones(J) / J)
    counts = jnp.ones(J).astype(int)
    loglik = 0

    mop_helper_2 = partial(
        _mop_helper,
        times0=jnp.concatenate([jnp.array([t0]), times]),
        ys=ys,
        theta=theta,
        rprocess=rprocess,
        dmeasure=dmeasure,
        ctimes=ctimes,
        covars=covars,
        alpha=alpha,
    )

    particlesF, loglik, weightsF, counts, key = jax.lax.fori_loop(
        lower=0,
        upper=len(ys),
        body_fun=mop_helper_2,
        init_val=(particlesF, loglik, weightsF, counts, key),
    )

    return -loglik


@partial(jit, static_argnums=(4, 5, 6, 7))
def _mop_internal_mean(
    theta: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
    dmeasure: Callable,  # static
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    alpha: float,
    key: jax.Array,
) -> float:
    """
    Internal function for calculating the mean result using MOP algorithm
    across the measurements. This is used in internal pypomp.train functions.
    """
    return _mop_internal(
        theta=theta,
        t0=t0,
        times=times,
        ys=ys,
        J=J,
        rinitializer=rinitializer,
        rprocess=rprocess,
        dmeasure=dmeasure,
        ctimes=ctimes,
        covars=covars,
        alpha=alpha,
        key=key,
    ) / len(ys)


def _mop_helper(
    i: int,
    inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    times0: jax.Array,
    ys: jax.Array,
    theta: jax.Array,
    rprocess: Callable,
    dmeasure: Callable,
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    alpha: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Helper functions for MOP algorithm, which conducts a single iteration of
    filtering and is called in function 'mop_internal'.
    """
    particlesF, loglik, weightsF, counts, key = inputs
    J = len(particlesF)
    t1 = times0[i]
    t2 = times0[i + 1]

    weightsP = alpha * weightsF

    key, keys = _keys_helper(key=key, J=J, covars=covars)
    particlesP = rprocess(particlesF, theta, keys, ctimes, covars, t1, t2)

    covars_t = _interp_covars(t2, ctimes=ctimes, covars=covars)
    measurements = dmeasure(ys[i], particlesP, theta, covars_t, t2)
    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)

    loglik = (
        loglik
        + jax.scipy.special.logsumexp(weightsP + measurements)
        - jax.scipy.special.logsumexp(weightsP)
    )
    # test different, logsumexp - source code (floating point arithmetic issue)
    # make a little note in the code, discuss it in the quant test about the small difference
    # logsumexp source code

    (norm_weights, loglik_phi_t) = _normalize_weights(
        jax.lax.stop_gradient(measurements)
    )

    key, subkey = jax.random.split(key)
    (counts, particlesF, norm_weightsF) = _resampler(
        counts, particlesP, norm_weights, subkey=subkey
    )

    weightsF = (weightsP + measurements - jax.lax.stop_gradient(measurements))[counts]

    return (particlesF, loglik, weightsF, counts, key)
