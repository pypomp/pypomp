from functools import partial
import jax
import jax.numpy as jnp
import xarray as xr
import pandas as pd
from jax import jit
from tqdm import tqdm
from typing import Callable
from .model_struct import RInit, RProc, DMeas
from .pfilter import _pfilter_internal
from .internal_functions import _normalize_weights
from .internal_functions import _keys_helper
from .internal_functions import _resampler_thetas
from .internal_functions import _no_resampler_thetas
from .internal_functions import _interp_covars

MONITORS = 1  # TODO: figure out what this is for and remove it if possible


def mif(
    rinit: RInit,
    rproc: RProc,
    dmeas: DMeas,
    ys: pd.DataFrame,
    theta: dict,
    sigmas: float | jax.Array,
    sigmas_init: float | jax.Array,
    M: int,
    a: float,
    J: int,
    key: jax.Array,
    covars: pd.DataFrame | None = None,
    thresh: float = 0.0,
    monitor: bool = False,
    verbose: bool = False,
):
    """
    Perform the iterated filtering (IF2) algorithm for a partially observed
    Markov process (POMP) model to estimate model parameters by maximizing
    the likelihood.

    Args:
        rinit (RInit): Simulator for the initial-state distribution.
        rproc (RProc): Simulator for the process model.
        dmeas (DMeas): Density evaluation for the measurement model.
        ys (array-like): The measurement array.
        theta (dict): Initial parameters for the POMP model. Each value must be a float.
        sigmas (float): Perturbation factor for parameters.
        sigmas_init (float): Initial perturbation factor for parameters.
        covars (array-like): Covariates or None if not applicable.
        M (int): Number of algorithm iterations.
        a (float): Decay factor for sigmas.
        J (int): Number of particles.
        thresh (float): Resampling threshold.
        monitor (bool): Flag to monitor log-likelihood values.
        verbose (bool): Flag to print log-likelihood and parameter information.
        key (jax.random.PRNGKey): Random key for reproducibility.

    Raises:
        ValueError: If J is less than 1 or any required arguments are missing.

    Returns:
        dict: a dictionary containing:
            - xarray of log-likelihood values through iterations.
            - xarray of parameters through iterations.
    """
    if J < 1:
        raise ValueError("J should be greater than 0.")

    if not isinstance(theta, dict):
        raise TypeError("theta must be a dictionary")
    if not all(isinstance(val, float) for val in theta.values()):
        raise TypeError("Each value of theta must be a float")

    nLLs, theta_ests = _mif_internal(
        theta=jnp.array(list(theta.values())),
        t0=rinit.t0,
        times=jnp.array(ys.index),
        ys=jnp.array(ys),
        rinitializer=rinit.struct_pf,
        rprocess=rproc.struct_pf,
        dmeasure=dmeas.struct_pf,
        rinitializers=rinit.struct_per,
        rprocesses=rproc.struct_per,
        dmeasures=dmeas.struct_per,
        sigmas=sigmas,
        sigmas_init=sigmas_init,
        ctimes=jnp.array(covars.index) if covars is not None else None,
        covars=jnp.array(covars) if covars is not None else None,
        M=M,
        a=a,
        J=J,
        thresh=thresh,
        monitor=monitor,
        verbose=verbose,
        key=key,
    )

    return {
        "logLik": xr.DataArray(-nLLs, dims=["iteration"]),
        "thetas": xr.DataArray(
            theta_ests,
            dims=["iteration", "particle", "theta"],
            coords={
                "iteration": range(0, M + 1),
                "particle": range(1, J + 1),
                "theta": list(theta.keys()),
            },
        ),
    }


def _mif_internal(
    theta: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    rinitializer: Callable,
    rprocess: Callable,
    dmeasure: Callable,
    rinitializers: Callable,
    rprocesses: Callable,
    dmeasures: Callable,
    sigmas: float | jax.Array,
    sigmas_init: float | jax.Array,
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    M: int,
    a: float,
    J: int,
    thresh: float,
    monitor: bool,
    verbose: bool,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    logliks = []
    params = []

    ndim = theta.ndim
    thetas = jnp.tile(theta, (J,) + (1,) * ndim)
    params.append(thetas)

    if monitor:
        key, subkey = jax.random.split(key=key)
        loglik = jnp.mean(
            jnp.array(
                [
                    _pfilter_internal(
                        theta=thetas.mean(0),
                        t0=t0,
                        times=times,
                        ys=ys,
                        J=J,
                        rinitializer=rinitializer,
                        rprocess=rprocess,
                        dmeasure=dmeasure,
                        ctimes=ctimes,
                        covars=covars,
                        thresh=thresh,
                        key=subkey,
                    )
                    for i in range(MONITORS)
                ]
            )
        )
        logliks.append(loglik)

    for m in tqdm(range(M)):
        # TODO: Cool sigmas between time-iterations.
        key, *subkeys = jax.random.split(key=key, num=3)
        sigmas = a * sigmas
        sigmas_init = a * sigmas_init
        thetas = thetas + sigmas_init * jax.random.normal(
            shape=thetas.shape, key=subkeys[0]
        )
        loglik_ext, thetas = _perfilter_internal(
            thetas=thetas,
            t0=t0,
            times=times,
            ys=ys,
            J=J,
            sigmas=sigmas,
            rinitializers=rinitializers,
            rprocesses=rprocesses,
            dmeasures=dmeasures,
            ndim=ndim,
            ctimes=ctimes,
            covars=covars,
            thresh=thresh,
            key=subkeys[1],
        )
        params.append(thetas)

        if monitor:
            key, subkey = jax.random.split(key=key)
            loglik = jnp.mean(
                jnp.array(
                    [
                        _pfilter_internal(
                            theta=thetas.mean(0),
                            t0=t0,
                            times=times,
                            ys=ys,
                            J=J,
                            rinitializer=rinitializer,
                            rprocess=rprocess,
                            dmeasure=dmeasure,
                            ctimes=ctimes,
                            covars=covars,
                            thresh=thresh,
                            key=subkey,
                        )
                        for i in range(MONITORS)
                    ]
                )
            )
            logliks.append(loglik)

            if verbose:
                print(loglik)
                print(thetas.mean(0))

    return jnp.array(logliks), jnp.array(params)


@partial(jit, static_argnums=(4, 6, 7, 8, 9))
def _perfilter_internal(
    thetas: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    J: int,  # static
    sigmas: jax.Array,
    rinitializers: Callable,  # static
    rprocesses: Callable,  # static
    dmeasures: Callable,  # static
    ndim: int,  # static
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    thresh: float,
    key: jax.Array,
):
    """
    Internal function for the perturbed particle filtering algorithm, which calls
    function 'perfilter_helper' iteratively.
    """
    loglik = 0
    key, subkey = jax.random.split(key)
    thetas = thetas + sigmas * jax.random.normal(
        shape=(J,) + thetas.shape[-ndim:], key=subkey
    )

    key, keys = _keys_helper(key=key, J=J, covars=covars)
    covars_t = _interp_covars(t0, ctimes=ctimes, covars=covars)
    particlesF = rinitializers(thetas, keys, covars_t, t0)

    norm_weights = jnp.log(jnp.ones(J) / J)
    counts = jnp.ones(J).astype(int)

    perfilter_helper_2 = partial(
        _perfilter_helper,
        times0=jnp.concatenate([jnp.array([t0]), times]),
        ys=ys,
        rprocesses=rprocesses,
        dmeasures=dmeasures,
        sigmas=sigmas,
        ctimes=ctimes,
        covars=covars,
        thresh=thresh,
    )
    (particlesF, thetas, loglik, norm_weights, counts, key) = jax.lax.fori_loop(
        lower=0,
        upper=len(ys),
        body_fun=perfilter_helper_2,
        init_val=(particlesF, thetas, loglik, norm_weights, counts, key),
    )

    return -loglik, thetas


def _perfilter_helper(
    i: int,
    inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    times0: jax.Array,
    ys: jax.Array,
    rprocesses: Callable,
    dmeasures: Callable,
    sigmas: jax.Array,
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    thresh: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Helper functions for perturbed particle filtering algorithm, which conducts
    a single iteration of filtering and is called in function
    'perfilter_internal'.
    """
    (particlesF, thetas, loglik, norm_weights, counts, key) = inputs
    J = len(particlesF)
    t1 = times0[i]
    t2 = times0[i + 1]

    key, subkey = jax.random.split(key)
    thetas += sigmas * jnp.array(jax.random.normal(shape=thetas.shape, key=subkey))

    key, keys = _keys_helper(key=key, J=J, covars=covars)
    particlesP = rprocesses(particlesF, thetas, keys, ctimes, covars, t1, t2)

    covars_t = _interp_covars(t2, ctimes=ctimes, covars=covars)
    measurements = jnp.nan_to_num(
        dmeasures(ys[i], particlesP, thetas, covars_t, t2).squeeze(), nan=jnp.log(1e-18)
    )  # shape (Np,)

    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)

    weights = norm_weights + measurements
    norm_weights, loglik_t = _normalize_weights(weights)

    loglik += loglik_t
    oddr = jnp.exp(jnp.max(norm_weights)) / jnp.exp(jnp.min(norm_weights))
    key, subkey = jax.random.split(key)
    counts, particlesF, norm_weights, thetas = jax.lax.cond(
        oddr > thresh,
        _resampler_thetas,
        _no_resampler_thetas,
        *(counts, particlesP, norm_weights, thetas, subkey),
    )

    return (particlesF, thetas, loglik, norm_weights, counts, key)
