import jax
import jax.numpy as jnp
from jax import jit
from .helpers import _normalize_weights
from .helpers import _resampler_thetas
from .helpers import _no_resampler_thetas

from .pfilter import _vmapped_pfilter_internal
from .types import (
    MifConfig,
    MifInputs,
    PfilterConfig,
    PfilterInputs,
    PerfilterState,
    PerfilterStepInputs,
)

SHOULD_TRANS = True  # Should transformations be applied to the parameters?


def _mif_internal(
    theta_Jd: jax.Array,
    key: jax.Array,
    config: MifConfig,
    inputs: MifInputs,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # 1. Prepare keys.
    all_keys = jax.random.split(key, num=config.M + 1)
    m_keys = all_keys[1:]

    # 2. Prepare scan input
    config_pf = config.to_pfilter_config()
    inputs_pf = inputs.to_pfilter_inputs()
    init_carry = theta_Jd
    scan_xs = (jnp.arange(config.M), m_keys)
    mif_scan_body_fn = jax.tree_util.Partial(
        _mif_scan_body,
        config,
        inputs,
        config_pf,
        inputs_pf,
    )

    # 3. Run the perturbed particle filter + optional unperturbed particle filter.
    final_theta_Jd, (thetas_history_mean, neg_logliks_history) = jax.lax.scan(
        f=mif_scan_body_fn,
        init=init_carry,
        xs=scan_xs,
    )

    # 4. Collect results.
    # thetas_traces_Md: (M+1, n_theta)
    thetas_traces_Md = jnp.concatenate(
        [jnp.mean(theta_Jd, axis=0)[None, :], thetas_history_mean], axis=0
    )
    # neg_logliks_M: (M,)
    neg_logliks_M = neg_logliks_history

    return neg_logliks_M, thetas_traces_Md, final_theta_Jd


def _mif_scan_body(
    config: MifConfig,
    inputs: MifInputs,
    config_pf: PfilterConfig,
    inputs_pf: PfilterInputs,
    carry: jax.Array,
    scan_inputs: tuple[int | jax.Array, jax.Array],
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """Runs one iteration of IF2; optionally runs pfilter for unperturbed logLik."""
    current_theta_Jd = carry
    m, iter_key = scan_inputs

    # 1. Run the perturbed filter.
    next_theta_Jd, neg_loglik_per, _ = _perfilter_internal(
        m_current=m,
        thetas_Jd=current_theta_Jd,
        key=iter_key,
        config=config,
        inputs=inputs,
    )

    # 2. Optionally run pfilter for unperturbed logLik
    if config.n_monitors >= 1:
        current_theta_mean = jnp.mean(current_theta_Jd, axis=0)
        mon_keys = jax.random.split(iter_key, config.n_monitors)
        neg_loglik_m = jnp.mean(
            _vmapped_pfilter_internal(
                current_theta_mean,
                mon_keys,
                config_pf,
                inputs_pf,
            )["neg_loglik"]
        )
    else:
        neg_loglik_m = neg_loglik_per

    return next_theta_Jd, (jnp.mean(next_theta_Jd, axis=0), neg_loglik_m)


def _perfilter_internal(
    m_current: int | jax.Array,
    thetas_Jd: jax.Array,
    key: jax.Array,
    config: MifConfig,
    inputs: MifInputs,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Runs one iteration of the perturbed particle filtering algorithm."""
    # 1. Setup.
    J = config.J
    loglik = jnp.array(0.0)
    norm_weights = jnp.full(J, -jnp.log(J))
    counts = jnp.ones(J, dtype=int)
    time_indices = jnp.arange(len(inputs.ys))
    cooling_factors = jax.vmap(
        lambda i: config.cooling_fn(i, m_current, len(inputs.times))
    )(time_indices)
    # Ancestry tracking is for PIF (Panel.mif with block=False).
    if config.return_ancestry:
        ancestry = jnp.arange(J)
    else:
        ancestry = jnp.zeros((0,), dtype=jnp.int32)

    # 2. Perturb the parameters for t0
    key, subkey = jax.random.split(key)
    sigmas_init_cooled = (
        config.cooling_fn(0, m_current, len(inputs.times)) * inputs.sigmas_init
    )
    thetas_Jd = thetas_Jd + sigmas_init_cooled * jax.random.normal(
        shape=thetas_Jd.shape, key=subkey
    )

    # 3. Initialize particle states at t0
    split_keys = jax.random.split(key, num=J + 1)
    key = split_keys[0]
    keys = split_keys[1:]
    covars0 = None if inputs.covars_extended is None else inputs.covars_extended[0]
    particlesF_Jx = config.rinitializer(
        thetas_Jd, keys, covars0, inputs.t0, SHOULD_TRANS
    )

    # 4. Prepare scan inputs
    all_keys = jax.random.split(key, num=len(inputs.ys) + 1)
    step_keys_raw = all_keys[1:]
    perfilter_scan_xs = PerfilterStepInputs(
        y=inputs.ys,
        time=inputs.times,
        nstep=inputs.nstep_array,
        cooling_factor=cooling_factors,
        step_key=step_keys_raw,
    )
    init_state = PerfilterState(
        t=inputs.t0,
        particlesF=particlesF_Jx,
        thetas=thetas_Jd,
        loglik=loglik,
        norm_weights=norm_weights,
        counts=counts,
        t_idx=0,
        ancestry=ancestry,
    )
    scan_body_fn = jax.tree_util.Partial(
        _perfilter_scan_body,
        config,
        inputs,
    )

    # 5. Run the perturbed particle filter over the time series.
    final_state, _ = jax.lax.scan(
        f=scan_body_fn,
        init=init_state,
        xs=perfilter_scan_xs,
    )

    return final_state.thetas, -final_state.loglik, final_state.ancestry


def _perfilter_scan_body(
    config: MifConfig,
    inputs: MifInputs,
    carry: PerfilterState,
    xs: PerfilterStepInputs,
) -> tuple[PerfilterState, None]:
    """Runs the perturbed particle filter for one observation interval."""
    # 1. Setup.
    key_perturb, key_process, key_resample = jax.random.split(xs.step_key, 3)
    nstep_int = jnp.asarray(xs.nstep).astype(int)

    # 2. Perturb the parameters.
    sigmas_cooled = xs.cooling_factor * inputs.sigmas
    perturbed_thetas = carry.thetas + sigmas_cooled * jax.random.normal(
        shape=carry.thetas.shape, key=key_perturb
    )

    # 3. Propagate the particles for one observation interval.
    keys_step = jax.random.split(key_process, num=config.J + 1)[1:]
    particlesP_Jx, updated_t_idx = config.rprocess_interp(
        carry.particlesF,
        perturbed_thetas,
        keys_step,
        inputs.covars_extended,
        inputs.dt_array_extended,
        carry.t,
        carry.t_idx,
        nstep_int,
        config.accumvars,
        SHOULD_TRANS,
    )

    # 4. Update covars to current observation time.
    covars_t = (
        None
        if inputs.covars_extended is None
        else inputs.covars_extended[updated_t_idx]
    )

    # 5. Compute log-likelihood contribution of current observation.
    measurements = jnp.nan_to_num(
        config.dmeasure(
            xs.y, particlesP_Jx, perturbed_thetas, covars_t, xs.time, SHOULD_TRANS
        ).squeeze(),
        nan=jnp.log(1e-18),
    )

    # 6. Update the running log-likelihood and normalize particle weights.
    weights = carry.norm_weights + measurements
    norm_weights_updated, loglik_t = _normalize_weights(weights)
    loglik_updated = carry.loglik + loglik_t

    # 7. Resample if necessary.
    should_resample = jnp.max(norm_weights_updated) - jnp.min(
        norm_weights_updated
    ) > jnp.log(config.thresh)
    counts_resampled, particlesF_resampled, norm_weights_resampled, thetas_resampled = (
        jax.lax.cond(
            should_resample,
            _resampler_thetas,
            _no_resampler_thetas,
            *(
                carry.counts,
                particlesP_Jx,
                norm_weights_updated,
                perturbed_thetas,
                key_resample,
            ),
        )
    )

    # 8. Update the ancestry (used for PIF, i.e., Panel.mif with block=False).
    if config.return_ancestry:
        step_indices = jax.lax.cond(
            should_resample,
            lambda: counts_resampled,
            lambda: jnp.arange(config.J),
        )
        ancestry_updated = carry.ancestry[step_indices]
    else:
        ancestry_updated = carry.ancestry

    # 9. Return the updated state
    new_state = PerfilterState(
        t=xs.time,
        particlesF=particlesF_resampled,
        thetas=thetas_resampled,
        loglik=loglik_updated,
        norm_weights=norm_weights_resampled,
        counts=counts_resampled,
        t_idx=updated_t_idx,
        ancestry=ancestry_updated,
    )
    return new_state, None


_vmapped_mif_internal = jax.vmap(
    _mif_internal,
    in_axes=(
        0,  # theta_Jd
        0,  # key
        None,  # config
        None,  # inputs
    ),
)

_jv_mif_internal = jit(
    _vmapped_mif_internal,
    static_argnames=("config",),
)
