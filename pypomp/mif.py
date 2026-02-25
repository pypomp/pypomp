import jax
import jax.numpy as jnp
from jax import jit
from typing import Callable
from .internal_functions import _normalize_weights
from .internal_functions import _keys_helper
from .internal_functions import _resampler_thetas
from .internal_functions import _no_resampler_thetas
from .internal_functions import _geometric_cooling

SHOULD_TRANS = True  # Should transformations be applied to the parameters?


def _mif_internal(
    theta_Jd: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    rinitializers: Callable,  # static
    rprocesses_interp: Callable,  # static
    dmeasures: Callable,  # static
    sigmas: float | jax.Array,
    sigmas_init: float | jax.Array,
    accumvars: jax.Array | None,
    covars_extended: jax.Array | None,
    M: int,  # static
    a: float,
    J: int,  # static
    thresh: float,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    times = times.astype(float)
    all_keys = jax.random.split(key, num=M + 1)
    m_keys = all_keys[1:]

    def mif_scan_body(carry, scan_inputs):
        current_theta_Jd = carry
        m, iter_key = scan_inputs

        next_theta_Jd, loglik_m = _perfilter_internal(
            m,
            current_theta_Jd,
            iter_key,
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
        )
        return next_theta_Jd, (next_theta_Jd, loglik_m)

    init_carry = theta_Jd
    scan_xs = (jnp.arange(M), m_keys)

    final_state, (thetas_history, logliks_history) = jax.lax.scan(
        f=mif_scan_body,
        init=init_carry,
        xs=scan_xs,
    )

    # thetas_MJd: (M+1, J, n_theta)
    thetas_MJd = jnp.concatenate([theta_Jd[None, :, :], thetas_history], axis=0)
    # logliks_M: (M,)
    logliks_M = logliks_history

    return logliks_M, thetas_MJd


_vmapped_mif_internal = jax.vmap(
    _mif_internal,
    in_axes=(1,) + (None,) * 16 + (0,),
)

_jv_mif_internal = jit(_vmapped_mif_internal, static_argnums=(6, 7, 8, 13, 15))


def _perfilter_internal(
    m: int,
    thetas_Jd: jax.Array,
    key: jax.Array,
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
) -> tuple[jax.Array, jax.Array]:
    """
    Internal function for one iteration of the perturbed particle filtering algorithm.
    """
    loglik = jnp.array(0.0)
    key, subkey = jax.random.split(key)
    sigmas_init_cooled = (
        _geometric_cooling(nt=0, m=m, ntimes=len(times), a=a) * sigmas_init
    )
    thetas_Jd = thetas_Jd + sigmas_init_cooled * jax.random.normal(
        shape=thetas_Jd.shape, key=subkey
    )

    key, keys = _keys_helper(key=key, J=J, covars=covars_extended)
    covars0 = None if covars_extended is None else covars_extended[0]
    particlesF_Jx = rinitializers(thetas_Jd, keys, covars0, t0, SHOULD_TRANS)

    norm_weights = jnp.full(J, -jnp.log(J))
    counts = jnp.ones(J, dtype=int)

    time_indices = jnp.arange(len(ys))
    cooling_factors = jax.vmap(
        lambda i: _geometric_cooling(nt=i, m=m, ntimes=len(times), a=a)
    )(time_indices)

    all_keys = jax.random.split(key, num=len(ys) + 1)
    step_keys_raw = all_keys[1:]

    perfilter_scan_xs = (
        ys,
        times,
        nstep_array,
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
        return _perfilter_helper(
            carry,
            xs,
            rprocesses_interp=rprocesses_interp,
            dmeasures=dmeasures,
            sigmas=sigmas,
            accumvars=accumvars,
            covars_extended=covars_extended,
            dt_array_extended=dt_array_extended,
            thresh=thresh,
        )

    (t, particlesF_Jx, thetas_Jd, loglik, norm_weights, counts, t_idx), _ = (
        jax.lax.scan(
            f=scan_body,
            init=init_state,
            xs=perfilter_scan_xs,
        )
    )

    return thetas_Jd, -loglik


def _perfilter_helper(
    carry: tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        int,
    ],
    xs: tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ],
    rprocesses_interp: Callable,
    dmeasures: Callable,
    sigmas: float | jax.Array,
    accumvars: jax.Array | None,
    covars_extended: jax.Array | None,
    dt_array_extended: jax.Array,
    thresh: float,
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
    """
    Runs one iteration of the perturbed particle filtering algorithm.
    """
    (t, particlesF_Jx, thetas_Jd, loglik, norm_weights, counts, t_idx) = carry
    (y, time, nstep, cooling_factor, step_key) = xs
    J = len(particlesF_Jx)

    key_perturb, key_process, key_resample = jax.random.split(step_key, 3)

    sigmas_cooled = cooling_factor * sigmas
    thetas_Jd = thetas_Jd + sigmas_cooled * jax.random.normal(
        shape=thetas_Jd.shape, key=key_perturb
    )

    _, keys = _keys_helper(key=key_process, J=J, covars=covars_extended)

    nstep = nstep.astype(int)

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
    t = time

    covars_t = None if covars_extended is None else covars_extended[t_idx]

    measurements = jnp.nan_to_num(
        dmeasures(y, particlesP_Jx, thetas_Jd, covars_t, t, SHOULD_TRANS).squeeze(),
        nan=jnp.log(1e-18),
    )
    if len(measurements.shape) > 1:
        measurements = measurements.sum(axis=-1)

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


def _panel_mif_internal(
    shared_array: jax.Array,  # (n_shared, J)
    unit_array: jax.Array,  # (n_spec, J, U)
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    ys_per_unit: jax.Array,  # (U, T, ...)
    rinitializers: Callable,
    rprocesses_interp: Callable,
    dmeasures: Callable,
    sigmas: jax.Array,
    sigmas_init: jax.Array,
    accumvars: jax.Array | None,
    covars_per_unit: jax.Array | None,  # (U, ...) or None
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

    shared_means0 = jnp.mean(shared_array, axis=1) if n_shared > 0 else jnp.zeros((0,))
    unit_means0 = jnp.mean(unit_array, axis=1) if n_spec > 0 else jnp.zeros((0, U))

    shared_trace_0 = jnp.concatenate([jnp.array([jnp.nan]), shared_means0])[None, :]
    unit_trace_0 = jnp.concatenate(
        [jnp.array([jnp.nan] * U)[None, :], unit_means0], axis=0
    )[None, :, :]

    all_keys = jax.random.split(key, num=M + 1)
    m_keys = all_keys[1:]

    def iter_body(carry, scan_inputs):
        (
            shared_array_m,
            unit_array_m,
        ) = carry
        m, iter_key = scan_inputs

        sum_loglik_iter = 0.0

        def unit_scan_fn(inner_carry, unit_inputs):
            (
                shared_array_u,
                sum_loglik_u,
            ) = inner_carry

            (
                unit_array_u_m_single,  # (n_spec, J)
                unit_param_perm_u,
                ys_u,
                covars_u_dummy,
                u_idx,
                unit_key,
            ) = unit_inputs

            covars_u = None if covars_per_unit is None else covars_u_dummy

            # Build per-unit thetas: (J, n_params) in unit's canonical order
            if (n_shared + n_spec) > 0:
                thetas_u_panel_order = jnp.concatenate(
                    [shared_array_u.T, unit_array_u_m_single.T], axis=1
                )
            else:
                thetas_u_panel_order = jnp.zeros((J, 0))
            thetas_u = thetas_u_panel_order[:, unit_param_perm_u]

            subkey = unit_key

            sigmas_u = sigmas[unit_param_perm_u]
            sigmas_init_u = sigmas_init[unit_param_perm_u]

            sigmas_init_u_cooled = (
                _geometric_cooling(nt=0, m=m, ntimes=len(times), a=a) * sigmas_init_u
            )
            sigmas_u_cooled = (
                _geometric_cooling(nt=0, m=m, ntimes=len(times), a=a) * sigmas_u
            )

            nLL_u, updated_thetas_u = _mif_internal(
                thetas_u,
                dt_array_extended,
                nstep_array,
                t0,
                times,
                ys_u,
                rinitializers,
                rprocesses_interp,
                dmeasures,
                sigmas_u_cooled,
                sigmas_init_u_cooled,
                accumvars,
                covars_u,
                1,
                a,
                J,
                thresh,
                subkey,
            )
            nLL_u = nLL_u[0]
            # skips initial parameters from output:
            updated_thetas_u = updated_thetas_u[1]  # (J, n_params)

            inv_perm = jnp.argsort(unit_param_perm_u)
            updated_thetas_panel = updated_thetas_u[:, inv_perm]

            if n_shared > 0:
                new_shared_array = updated_thetas_panel[:, :n_shared].T
            else:
                new_shared_array = shared_array_u

            if n_spec > 0:
                updated_spec_u = updated_thetas_panel[:, n_shared:].T
            else:
                updated_spec_u = unit_array_u_m_single

            loglik_u = -nLL_u
            sum_loglik_u = sum_loglik_u + loglik_u

            if n_spec > 0:
                unit_traces_u_m_local = jnp.concatenate(
                    [jnp.array([loglik_u]), jnp.mean(updated_spec_u, axis=1)]
                )
            else:
                unit_traces_u_m_local = jnp.array([loglik_u])

            new_inner_carry = (
                new_shared_array,
                sum_loglik_u,
            )

            scan_outputs = (updated_spec_u, unit_traces_u_m_local)

            return new_inner_carry, scan_outputs

        unit_keys = jax.random.split(iter_key, num=U)

        # unit_array_m: (n_spec, J, U) -> we want to scan over U
        unit_scan_seq = (
            jnp.moveaxis(unit_array_m, 2, 0) if n_spec > 0 else jnp.zeros((U, 0, J)),
            unit_param_permutations,
            ys_per_unit,
            covars_per_unit
            if covars_per_unit is not None
            else jnp.zeros((U, 0)),  # dummy
            jnp.arange(U),
            unit_keys,
        )

        initial_inner_carry = (
            shared_array_m,
            sum_loglik_iter,
        )

        final_inner_carry, (unit_array_m_new_seq, unit_traces_m_new_seq) = jax.lax.scan(
            f=unit_scan_fn,
            init=initial_inner_carry,
            xs=unit_scan_seq,
        )

        (
            shared_array_m,
            sum_loglik_iter,
        ) = final_inner_carry

        # unit_array_m_new_seq: (U, n_spec, J) -> move back to (n_spec, J, U)
        if n_spec > 0:
            unit_array_m = jnp.moveaxis(unit_array_m_new_seq, 0, 2)

        # unit_traces_m_new_seq: (U, n_spec + 1) -> (n_spec + 1, U)
        unit_traces_m_row = jnp.moveaxis(unit_traces_m_new_seq, 0, 1)

        if n_shared > 0:
            shared_means = jnp.mean(shared_array_m, axis=1)
            shared_traces_m_row = jnp.concatenate(
                [jnp.array([sum_loglik_iter]), shared_means]
            )
        else:
            shared_traces_m_row = jnp.array([sum_loglik_iter])

        return (
            (shared_array_m, unit_array_m),
            (shared_traces_m_row, unit_traces_m_row),
        )

    initial_iter_carry = (shared_array, unit_array)
    iter_scan_xs = (jnp.arange(M), m_keys)

    (final_iter_state, (shared_traces_history, unit_traces_history)) = jax.lax.scan(
        f=iter_body,
        init=initial_iter_carry,
        xs=iter_scan_xs,
    )

    (shared_array, unit_array) = final_iter_state

    shared_traces = jnp.concatenate([shared_trace_0, shared_traces_history], axis=0)
    unit_traces = jnp.concatenate([unit_trace_0, unit_traces_history], axis=0)

    return (shared_array, unit_array, shared_traces, unit_traces)


_vmapped_panel_mif_internal = jax.vmap(
    _panel_mif_internal, in_axes=((0, 0) + (None,) * 18 + (0,))
)

_jv_panel_mif_internal = jit(
    _vmapped_panel_mif_internal,
    static_argnums=(
        7,  # rinitializers
        8,  # rprocesses_interp
        9,  # dmeasures
        15,  # M
        17,  # J
        18,  # U
    ),
)
