"""
Core routines for Bayesian iterated filtering (BIF).

BIF runs an IF2-like mutation-selection filter in estimation space and then
uses the fixed initial perturbation kernel to deconvolve the stationary
parameter cloud. The Stage 1 kernel keeps the initial perturbation fixed across
outer iterations while cooling only the within-trajectory random walk.
"""

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.special import logsumexp

from .helpers import _keys_helper
from .helpers import _normalize_weights
from .helpers import _no_resampler_thetas
from .helpers import _resampler_thetas
from .helpers import _geometric_cooling
from .pfilter import _vmapped_pfilter_internal


SHOULD_TRANS = True


def _bif_internal(
    theta_Jd: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    rinitializers: Callable,
    rprocesses_interp: Callable,
    dmeasures: Callable,
    rw_sigmas: jax.Array,
    perturb_sigmas: jax.Array,
    accumvars: jax.Array | None,
    covars_extended: jax.Array | None,
    M: int,
    a: float,
    J: int,
    thresh: float,
    key: jax.Array,
    dprior: Callable,
    rinitializer_pf: Callable,
    rprocess_pf: Callable,
    dmeasure_pf: Callable,
    n_monitors: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    times = times.astype(float)
    all_keys = jax.random.split(key, num=M + 1)
    m_keys = all_keys[1:]

    def scan_body(current_theta_Jd, scan_inputs):
        m, iter_key = scan_inputs

        next_theta_Jd, neg_loglik_per = _bif_perfilter_internal(
            m,
            current_theta_Jd,
            iter_key,
            dt_array_extended=dt_array_extended,
            nstep_array=nstep_array,
            t0=t0,
            times=times,
            ys=ys,
            J=J,
            rw_sigmas=rw_sigmas,
            perturb_sigmas=perturb_sigmas,
            rinitializers=rinitializers,
            rprocesses_interp=rprocesses_interp,
            dmeasures=dmeasures,
            dprior=dprior,
            accumvars=accumvars,
            covars_extended=covars_extended,
            thresh=thresh,
            a=a,
        )

        if n_monitors >= 1:
            current_theta_mean = jnp.mean(next_theta_Jd, axis=0)
            _, *subkeys = jax.random.split(iter_key, n_monitors + 1)
            neg_loglik_m = jnp.mean(
                _vmapped_pfilter_internal(
                    current_theta_mean,
                    dt_array_extended,
                    nstep_array,
                    t0,
                    times,
                    ys,
                    J,
                    rinitializer_pf,
                    rprocess_pf,
                    dmeasure_pf,
                    accumvars,
                    covars_extended,
                    thresh,
                    jnp.array(subkeys),
                    False,
                    False,
                    False,
                    False,
                    SHOULD_TRANS,
                )["neg_loglik"]
            )
        else:
            neg_loglik_m = neg_loglik_per

        return next_theta_Jd, (jnp.mean(next_theta_Jd, axis=0), neg_loglik_m)

    scan_xs = (jnp.arange(M), m_keys)
    final_theta_Jd, (theta_history_mean, neg_logliks_history) = jax.lax.scan(
        f=scan_body,
        init=theta_Jd,
        xs=scan_xs,
    )

    theta_traces_Md = jnp.concatenate(
        [jnp.mean(theta_Jd, axis=0)[None, :], theta_history_mean], axis=0
    )

    return neg_logliks_history, theta_traces_Md, final_theta_Jd


_vmapped_bif_internal = jax.vmap(
    _bif_internal,
    in_axes=(1,) + (None,) * 16 + (0,) + (None,) * 5,
)

_jv_bif_internal = jit(
    _vmapped_bif_internal,
    static_argnums=(6, 7, 8, 13, 15, 18, 19, 20, 21, 22),
)


def _bif_perfilter_internal(
    m: int,
    thetas_Jd: jax.Array,
    key: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    J: int,
    rw_sigmas: jax.Array,
    perturb_sigmas: jax.Array,
    rinitializers: Callable,
    rprocesses_interp: Callable,
    dmeasures: Callable,
    dprior: Callable,
    accumvars: jax.Array | None,
    covars_extended: jax.Array | None,
    thresh: float,
    a: float,
) -> tuple[jax.Array, jax.Array]:
    loglik = jnp.array(0.0)
    key, subkey = jax.random.split(key)
    thetas_Jd = thetas_Jd + perturb_sigmas * jax.random.normal(
        shape=thetas_Jd.shape, key=subkey
    )

    key, keys = _keys_helper(key=key, J=J, covars=covars_extended)
    covars0 = None if covars_extended is None else covars_extended[0]
    particlesF_Jx = rinitializers(thetas_Jd, keys, covars0, t0, SHOULD_TRANS)

    norm_weights = jnp.full(J, -jnp.log(J))
    counts = jnp.ones(J, dtype=int)

    time_indices = jnp.arange(len(ys))
    cooling_factors = jax.vmap(
        lambda i: _geometric_cooling(nt=i, m=m + 1, ntimes=len(times), a=a)
    )(time_indices)

    all_keys = jax.random.split(key, num=len(ys) + 1)
    step_keys_raw = all_keys[1:]

    perfilter_scan_xs = (
        time_indices,
        cooling_factors,
        step_keys_raw,
    )

    init_state = (
        t0,
        particlesF_Jx,
        thetas_Jd,
        loglik,
        norm_weights,
        counts,
        0,
    )

    def scan_body(carry, xs):
        return _bif_perfilter_helper(
            carry,
            xs,
            rprocesses_interp=rprocesses_interp,
            dmeasures=dmeasures,
            dprior=dprior,
            rw_sigmas=rw_sigmas,
            accumvars=accumvars,
            covars_extended=covars_extended,
            dt_array_extended=dt_array_extended,
            nstep_array=nstep_array,
            times=times,
            ys=ys,
            thresh=thresh,
            n_obs=len(ys),
        )

    (_, _, thetas_Jd, loglik, _, _, _), _ = jax.lax.scan(
        f=scan_body,
        init=init_state,
        xs=perfilter_scan_xs,
    )

    return thetas_Jd, -loglik


def _bif_perfilter_helper(
    carry: tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        int,
    ],
    xs: tuple[jax.Array, jax.Array, jax.Array],
    rprocesses_interp: Callable,
    dmeasures: Callable,
    dprior: Callable,
    rw_sigmas: jax.Array,
    accumvars: jax.Array | None,
    covars_extended: jax.Array | None,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    times: jax.Array,
    ys: jax.Array,
    thresh: float,
    n_obs: int,
) -> tuple[
    tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        int,
    ],
    None,
]:
    (t, particlesF_Jx, thetas_Jd, loglik, norm_weights, counts, t_idx) = carry
    (obs_idx, cooling_factor, step_key) = xs
    J = len(particlesF_Jx)

    key_perturb, key_process, key_resample = jax.random.split(step_key, 3)

    rw_sigmas_cooled = cooling_factor * rw_sigmas
    thetas_Jd = thetas_Jd + rw_sigmas_cooled * jax.random.normal(
        shape=thetas_Jd.shape, key=key_perturb
    )

    _, keys = _keys_helper(key=key_process, J=J, covars=covars_extended)

    nstep = nstep_array[obs_idx].astype(int)

    particlesP_Jx, t_idx = rprocesses_interp(
        particlesF_Jx,
        thetas_Jd,
        keys,
        covars_extended,
        dt_array_extended,
        t,
        t_idx,
        nstep,
        accumvars,
        SHOULD_TRANS,
    )
    t = times[obs_idx]

    covars_t = None if covars_extended is None else covars_extended[t_idx]

    measurements = jnp.nan_to_num(
        dmeasures(
            ys[obs_idx], particlesP_Jx, thetas_Jd, covars_t, t, SHOULD_TRANS
        ).squeeze(),
        nan=jnp.log(1e-18),
    )
    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)

    prior_terms = jax.vmap(dprior)(thetas_Jd) / n_obs
    measurements = measurements + prior_terms

    weights = norm_weights + measurements
    norm_weights, loglik_t = _normalize_weights(weights)
    loglik = loglik + loglik_t

    oddr = jnp.exp(jnp.max(norm_weights)) / jnp.exp(jnp.min(norm_weights))
    counts, particlesF_Jx, norm_weights, thetas_Jd = jax.lax.cond(
        oddr > thresh,
        _resampler_thetas,
        _no_resampler_thetas,
        *(counts, particlesP_Jx, norm_weights, thetas_Jd, key_resample),
    )

    return (t, particlesF_Jx, thetas_Jd, loglik, norm_weights, counts, t_idx), None


@jit
def _bif_deconvolution_diag(
    cloud_Nd: jax.Array,
    sd_d: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Leave-one-out Gaussian deconvolution weights for a diagonal kernel."""
    n = cloud_Nd.shape[0]
    d = cloud_Nd.shape[1]
    inv_var = 1.0 / (sd_d * sd_d)
    diffs = cloud_Nd[:, None, :] - cloud_Nd[None, :, :]
    quad = jnp.sum(diffs * diffs * inv_var, axis=-1)
    log_norm = -0.5 * d * jnp.log(2.0 * jnp.pi) - jnp.sum(jnp.log(sd_d))
    log_k = log_norm - 0.5 * quad
    log_k = log_k.at[jnp.diag_indices(n)].set(-jnp.inf)
    log_Hf = logsumexp(log_k, axis=1) - jnp.log(n - 1)
    log_w = -log_Hf
    log_w = log_w - logsumexp(log_w)
    weights = jnp.exp(log_w)
    ess = 1.0 / jnp.sum(weights * weights)
    return log_Hf, weights, ess
