from functools import partial
import jax.numpy as jnp
import jax
from jax import jit
from typing import Optional
from .internal_functions import _resampler
from .internal_functions import _no_resampler
from .internal_functions import _keys_helper
from .internal_functions import _normalize_weights
from .internal_functions import _interp_covars


def pfilter(
    J: int,
    rinit: callable,
    rproc: callable,
    dmeas: callable,
    theta: dict,
    ys: jax.Array,
    key: jax.Array,
    covars: Optional[jax.Array] = None,
    thresh: float = 0,
) -> float:
    """
    An outside function for the particle filtering algorithm.

    Args:
        J (int): The number of particles.
        rinit (RInit): Simulator for the initial-state distribution.
        rproc (RProc): Simulator for the process model.
        dmeas (DMeas): Density evaluation for the measurement model.
        theta (dict): Parameters involved in the POMP model. Each value must be a float.
        ys (array-like): The measurement array.
        key (jax.random.PRNGKey): The random key.
        covars (array-like, optional): Covariates or None if not applicable.
        thresh (float, optional): Threshold value to determine whether to
            resample particles.

    Raises:
        ValueError: Missing the pomp class object and required arguments for
            calling 'pfilter_internal' directly

    Returns:
        float: The log-likelihood estimate
    """
    if J < 1:
        raise ValueError("J should be greater than 0.")
    if rinit is None or rproc is None or dmeas is None or ys is None:
        raise ValueError("Missing rinit, rproc, dmeas, theta, or ys.")

    if not isinstance(theta, dict):
        raise TypeError("theta must be a dictionary")
    if not all(isinstance(val, float) for val in theta.values()):
        raise TypeError("Each value of theta must be a float")

    return -_pfilter_internal(
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
        thresh=thresh,
        key=key,
    )


@partial(jit, static_argnums=(4, 5, 6, 7))
def _pfilter_internal(
    theta: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    J: int,  # static
    rinitializer: callable,  # static
    rprocess: callable,  # static
    dmeasure: callable,  # static
    ctimes: Optional[jax.Array],
    covars: Optional[jax.Array],
    thresh: float,
    key: jax.Array,
):
    """
    Internal function for particle the filtering algorithm, which calls function
    'pfilter_helper' iteratively.
    """
    key, keys = _keys_helper(key=key, J=J, covars=covars)
    covars_t = _interp_covars(t0, ctimes=ctimes, covars=covars)
    particlesF = rinitializer(theta, keys, covars_t, t0)
    norm_weights = jnp.log(jnp.ones(J) / J)
    counts = jnp.ones(J).astype(int)
    loglik = 0

    pfilter_helper_2 = partial(
        _pfilter_helper,
        times0=jnp.concatenate([jnp.array([t0]), times]),
        rprocess=rprocess,
        dmeasure=dmeasure,
        theta=theta,
        ctimes=ctimes,
        covars=covars,
        ys=ys,
        thresh=thresh,
    )
    particlesF, loglik, norm_weights, counts, key = jax.lax.fori_loop(
        lower=0,
        upper=len(ys),
        body_fun=pfilter_helper_2,
        init_val=[particlesF, loglik, norm_weights, counts, key],
    )

    return -loglik


@partial(jit, static_argnums=(2, 3, 4, 5))
def _pfilter_internal_mean(
    theta, ys, J, rinitializer, rprocess, dmeasure, covars, thresh, key
):
    return _pfilter_internal(
        theta, ys, J, rinitializer, rprocess, dmeasure, covars, thresh, key
    ) / len(ys)


def _pfilter_helper(
    i: int,
    inputs: tuple[jax.Array, float, jax.Array, jax.Array, jax.Array],
    times0: jax.Array,
    rprocess: callable,
    dmeasure: callable,
    theta: jax.Array,
    ctimes: jax.Array,
    covars: jax.Array,
    ys: jax.Array,
    thresh: float,
) -> tuple[jax.Array, float, jax.Array, jax.Array, jax.Array]:
    """
    Helper function for the particle filtering algorithm in POMP, which conducts
    filtering for one time-iteration.
    """
    (particlesF, loglik, norm_weights, counts, key) = inputs
    J = len(particlesF)
    t1 = times0[i]
    t2 = times0[i + 1]

    key, keys = _keys_helper(key=key, J=J, covars=covars)
    particlesP = rprocess(particlesF, theta, keys, ctimes, covars, t1, t2)

    covars_t = _interp_covars(t2, ctimes=ctimes, covars=covars)
    measurements = dmeasure(ys[i], particlesP, theta, covars_t, t2)
    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)

    weights = norm_weights + measurements
    norm_weights, loglik_t = _normalize_weights(weights)
    loglik += loglik_t

    oddr = jnp.exp(jnp.max(norm_weights)) / jnp.exp(jnp.min(norm_weights))
    key, subkey = jax.random.split(key)
    counts, particlesF, norm_weights = jax.lax.cond(
        oddr > thresh,
        _resampler,
        _no_resampler,
        *(counts, particlesP, norm_weights, subkey),
    )

    return [particlesF, loglik, norm_weights, counts, key]
