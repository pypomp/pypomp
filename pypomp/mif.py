# pypomp/mif.py
from functools import partial
import jax
import jax.numpy as jnp
from jax import jit
from typing import Callable

from .internal_functions import _normalize_weights
from .internal_functions import _keys_helper
from .internal_functions import _resampler_thetas
from .internal_functions import _no_resampler_thetas
from .internal_functions import _geometric_cooling

# ---- Parameter transforms: minimal integration ----
# Import ParTrans and helpers; do not define a local identity here.
from .parameter_trans import ParTrans, _pt_forward, _pt_inverse


def _mif_internal(
    theta: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    rinitializers: Callable,         # static
    rprocesses_interp: Callable,     # static
    dmeasures: Callable,             # static
    sigmas: float | jax.Array,
    sigmas_init: float | jax.Array,
    accumvars: jax.Array | None,
    covars_extended: jax.Array | None,
    # NOTE: internal functions should not rely on defaults; pass partrans explicitly.
    partrans: ParTrans,              # optional parameter transform (identity chosen by outer wrapper)
    M: int,                          # static
    a: float,
    J: int,                          # static
    thresh: float,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    IF2 core for a single set of parameters (shape (J, n_theta)).
    Runs M iterations and returns the negative log-likelihoods per iteration
    and the updated particle parameters.
    """
    times = times.astype(float)
    n_theta = theta.shape[-1]
    logliks_M = jnp.zeros(M)
    thetas_MJd = jnp.zeros((M, J, n_theta))
    thetas_MJd = jnp.concatenate([theta.reshape((1, J, n_theta)), thetas_MJd], axis=0)

    _perfilter_internal_2 = partial(
        _perfilter_internal,
        dt_array_extended=dt_array_extended,
        nstep_array=nstep_array,
        t0=t0,
        times=times,
        ys=ys,
        J=J,
        sigmas=sigmas,
        sigmas_init=sigmas_init,
        rinitializers=rinitializers,
        rprocesses_interp=rprocesses_interp,
        dmeasures=dmeasures,
        accumvars=accumvars,
        covars_extended=covars_extended,
        thresh=thresh,
        a=a,
        partrans=partrans,
    )

    (thetas_Md, logliks_M, key) = jax.lax.fori_loop(
        lower=0,
        upper=M,
        body_fun=_perfilter_internal_2,
        init_val=(thetas_MJd, logliks_M, key),
    )
    return logliks_M, thetas_Md


# vmap: replicate along theta's 2nd axis (index 1); vmap over keys along axis 0
_vmapped_mif_internal = jax.vmap(
    _mif_internal,
    in_axes=(1,) + (None,) * 17 + (0,),  # matches the function signature (17 None entries)
)

# jit: static argument indices (0-based)
# 6=rinitializers, 7=rprocesses_interp, 8=dmeasures, 13=partrans, 14=M, 16=J
_jv_mif_internal = jit(
    _vmapped_mif_internal,
    static_argnums=(6, 7, 8, 13, 14, 16),
)


def _perfilter_internal(
    m: int,
    inputs: tuple[jax.Array, jax.Array, jax.Array],
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    J: int,
    sigmas: float | jax.Array,
    sigmas_init: float | jax.Array,
    rinitializers: Callable,
    rprocesses_interp: Callable,
    dmeasures: Callable,
    accumvars: jax.Array | None,
    covars_extended: jax.Array | None,
    thresh: float,
    a: float,
    # NOTE: internal functions should not rely on defaults; pass partrans explicitly.
    partrans: ParTrans,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    One IF2 iteration: run a perturbed particle filter once over the observation times.
    """
    (thetas_MJd, logliks_M, key) = inputs
    thetas_Jd = thetas_MJd[m]
    loglik = 0.0

    # ---- Initial perturbation on the estimation scale ----
    key, subkey = jax.random.split(key)
    sigmas_init_cooled = (
        _geometric_cooling(nt=0, m=m, ntimes=len(times), a=a) * sigmas_init
    )
    z0 = _pt_forward(thetas_Jd, partrans)
    z0 = z0 + sigmas_init_cooled * jax.random.normal(shape=z0.shape, key=subkey)
    thetas_Jd  = _pt_inverse(z0, partrans)  # map back to natural scale

    key, keys = _keys_helper(key=key, J=J, covars=covars_extended)
    covars0 = None if covars_extended is None else covars_extended[0]
    particlesF_Jx = rinitializers(thetas_Jd, keys, covars0, t0)

    norm_weights = jnp.log(jnp.ones(J) / J)
    counts = jnp.ones(J).astype(int)

    perfilter_helper_2 = partial(
        _perfilter_helper,
        dt_array_extended=dt_array_extended,
        nstep_array=nstep_array,
        times=times,
        ys=ys,
        rprocesses_interp=rprocesses_interp,
        dmeasures=dmeasures,
        sigmas=sigmas,
        accumvars=accumvars,
        covars_extended=covars_extended,
        thresh=thresh,
        m=m,
        a=a,
        partrans=partrans,
    )
    (t, particlesF_Jx, thetas_Jd, loglik, norm_weights, counts, key, t_idx) = (
        jax.lax.fori_loop(
            lower=0,
            upper=len(ys),
            body_fun=perfilter_helper_2,
            init_val=(
                t0,
                particlesF_Jx,
                thetas_Jd,
                loglik,
                norm_weights,
                counts,
                key,
                0,
            ),
        )
    )

    logliks_M = logliks_M.at[m].set(-loglik)
    thetas_MJd = thetas_MJd.at[m + 1].set(thetas_Jd)
    return thetas_MJd, logliks_M, key


def _perfilter_helper(
    i: int,
    inputs: tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        int,
    ],
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    times: jax.Array,
    ys: jax.Array,
    rprocesses_interp: Callable,
    dmeasures: Callable,
    sigmas: float | jax.Array,
    accumvars: jax.Array | None,
    covars_extended: jax.Array | None,
    thresh: float,
    m: int,
    a: float,
    # NOTE: internal functions should not rely on defaults; pass partrans explicitly.
    partrans: ParTrans,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    int,
]:
    """
    One predict–update step within the i-th observation interval, followed by
    optional resampling.
    """
    (t, particlesF_Jx, thetas_Jd, loglik, norm_weights, counts, key, t_idx) = inputs
    J = len(particlesF_Jx)

    # ---- At the start of the interval: perturb parameters on the estimation scale ----
    sigmas_cooled = _geometric_cooling(nt=i, m=m, ntimes=len(times), a=a) * sigmas
    key, subkey = jax.random.split(key)
    z = _pt_forward(thetas_Jd, partrans)
    z = z + sigmas_cooled * jnp.array(jax.random.normal(shape=z.shape, key=subkey))
    thetas_Jd = _pt_inverse(z, partrans)  # map back to natural scale for rproc/dmeas

    key, keys = _keys_helper(key=key, J=J, covars=covars_extended)

    nstep = nstep_array[i].astype(int)

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
    )
    t = times[i]

    covars_t = None if covars_extended is None else covars_extended[t_idx]

    measurements = jnp.nan_to_num(
        dmeasures(ys[i], particlesP_Jx, thetas_Jd, covars_t, t).squeeze(),
        nan=jnp.log(1e-18),
    )
    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)

    weights = norm_weights + measurements
    norm_weights, loglik_t = _normalize_weights(weights)
    loglik = loglik + loglik_t

    oddr = jnp.exp(jnp.max(norm_weights)) / jnp.exp(jnp.min(norm_weights))
    key, subkey = jax.random.split(key)
    counts, particlesF_Jx, norm_weights, thetas_Jd = jax.lax.cond(
        oddr > thresh,
        _resampler_thetas,
        _no_resampler_thetas,
        *(counts, particlesP_Jx, norm_weights, thetas_Jd, subkey),
    )

    return (t, particlesF_Jx, thetas_Jd, loglik, norm_weights, counts, key, t_idx)


# --------------------------------
# Panel IF2 (optional): integrate partrans with minimal changes as well
# --------------------------------
def _panel_mif_internal(
    shared_array: jax.Array,  # (n_shared, J)
    unit_array: jax.Array,    # (n_spec, J, U)
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    ys_per_unit: jax.Array,   # (U, T, ...)
    rinitializers: Callable,
    rprocesses_interp: Callable,
    dmeasures: Callable,
    sigmas: float | jax.Array,
    sigmas_init: float | jax.Array,
    accumvars: jax.Array | None,
    covars_per_unit: jax.Array | None,  # (U, ...) or None
    partrans: ParTrans,  
    unit_param_permutations: jax.Array,  # (U, n_params)
    M: int,
    a: float,
    J: int,
    U: int,
    thresh: float,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Fully JIT-compiled Panel IF2 across M iterations and U units.

    Returns
        shared_array_final: (n_shared, J)
        unit_array_final: (n_spec, J, U)
        shared_traces: (M+1, n_shared+1) where [:,0] is sum logLik per iter, [:,1:] are means
        unit_traces: (M+1, n_spec+1, U) where [:,0,:] is per-unit logLik per iter, [:,1:,:] are means
    """
    n_shared = shared_array.shape[0]
    n_spec = unit_array.shape[0]

    shared_traces = jnp.zeros((M + 1, n_shared + 1))
    unit_traces = jnp.zeros((M + 1, n_spec + 1, U))

    shared_means0 = jnp.where(
        n_shared > 0, jnp.mean(shared_array, axis=1), jnp.zeros((n_shared,))
    )
    unit_means0 = jnp.where(
        n_spec > 0, jnp.mean(unit_array, axis=1), jnp.zeros((n_spec, U))
    )
    shared_traces = shared_traces.at[0, 1:].set(shared_means0)
    shared_traces = shared_traces.at[0, 0].set(jnp.nan)
    unit_traces = unit_traces.at[0, 1:, :].set(unit_means0)
    unit_traces = unit_traces.at[0, 0, :].set(jnp.nan)

    def iter_body(m: int, carry):
        (
            shared_array_m,
            unit_array_m,
            shared_traces_m,
            unit_traces_m,
            key_m,
        ) = carry

        sum_loglik_iter = 0.0

        def unit_body(u: int, inner_carry):
            (
                shared_array_u,
                unit_array_u,
                sum_loglik_u,
                unit_traces_u,
                key_u,
            ) = inner_carry

            # Build per-unit thetas: (J, n_params) in unit's canonical order
            thetas_u_panel_order = (
                jnp.concatenate([shared_array_u.T, unit_array_u[:, :, u].T], axis=1)
                if (n_shared + n_spec) > 0
                else jnp.zeros((J, 0))
            )
            thetas_u = thetas_u_panel_order[:, unit_param_permutations[u]]

            key_u, subkey = jax.random.split(key_u)

            covars_u = None if covars_per_unit is None else covars_per_unit[u]

            sigmas_init_cooled = (
                _geometric_cooling(nt=0, m=m, ntimes=len(times), a=a) * sigmas_init
            )
            sigmas_cooled = (
                _geometric_cooling(nt=0, m=m, ntimes=len(times), a=a) * sigmas
            )

            nLL_u, updated_thetas_u = _mif_internal(
                thetas_u,
                dt_array_extended,
                nstep_array,
                t0,
                times,
                ys_per_unit[u],
                rinitializers,
                rprocesses_interp,
                dmeasures,
                sigmas_cooled,
                sigmas_init_cooled,
                accumvars,
                covars_u,
                partrans,
                1,
                a,
                J,
                thresh,
                subkey,
            )
            nLL_u = nLL_u[0]
            updated_thetas_u = updated_thetas_u[1]  # skip the initial parameters

            # Split back into shared and unit-specific blocks
            def update_shared(ppm, ut):
                return ut[:, :n_shared].T

            def keep_shared(ppm, ut):
                return ppm

            def update_spec(ppa, ut):
                return ut[:, n_shared:].T

            def keep_spec(ppa, ut):
                return ppa[:, :, u]

            new_shared_array = jax.lax.cond(
                n_shared > 0,
                update_shared,
                keep_shared,
                *(shared_array_u, updated_thetas_u),
            )
            updated_spec_u = jax.lax.cond(
                n_spec > 0,
                update_spec,
                keep_spec,
                *(unit_array_u, updated_thetas_u),
            )
            unit_array_u = unit_array_u.at[:, :, u].set(updated_spec_u)
            shared_array_u = new_shared_array

            loglik_u = -nLL_u
            sum_loglik_u = sum_loglik_u + loglik_u
            unit_traces_u = unit_traces_u.at[m + 1, 0, u].set(loglik_u)

            return (
                shared_array_u,
                unit_array_u,
                sum_loglik_u,
                unit_traces_u,
                key_u,
            )

        (
            shared_array_m,
            unit_array_m,
            sum_loglik_iter,
            unit_traces_m,
            key_m,
        ) = jax.lax.fori_loop(
            lower=0,
            upper=U,
            body_fun=unit_body,
            init_val=(
                shared_array_m,
                unit_array_m,
                sum_loglik_iter,
                unit_traces_m,
                key_m,
            ),
        )

        shared_means = jnp.where(
            n_shared > 0, jnp.mean(shared_array_m, axis=1), jnp.zeros((n_shared,))
        )
        unit_means = jnp.where(
            n_spec > 0, jnp.mean(unit_array_m, axis=1), jnp.zeros((n_spec, U))
        )
        shared_traces_m = shared_traces_m.at[m + 1, 1:].set(shared_means)
        shared_traces_m = shared_traces_m.at[m + 1, 0].set(sum_loglik_iter)
        unit_traces_m = unit_traces_m.at[m + 1, 1:, :].set(unit_means)

        return (
            shared_array_m,
            unit_array_m,
            shared_traces_m,
            unit_traces_m,
            key_m,
        )

    (
        shared_array,
        unit_array,
        shared_traces,
        unit_traces,
        key,
    ) = jax.lax.fori_loop(
        0,
        M,
        iter_body,
        (shared_array, unit_array, shared_traces, unit_traces, key),
    )

    return (shared_array, unit_array, shared_traces, unit_traces)


_vmapped_panel_mif_internal = jax.vmap(
    _panel_mif_internal, in_axes=((0, 0) + (None,) * 19 + (0,))
)

# static arg indices (0-based):
# 7=rinitializers, 8=rprocesses_interp, 9=dmeasures, 14=partrans, 15=M, 17=J, 18=U
_jv_panel_mif_internal = jit(
    _vmapped_panel_mif_internal,
    static_argnums=(7, 8, 9, 14, 16, 18, 19),
)