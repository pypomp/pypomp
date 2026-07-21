import jax
import jax.numpy as jnp
from jax import jit
from .pfilter import _vmapped_pfilter_internal
from .types import (
    PanelMifConfig,
    PanelMifInputs,
    PanelMifIterInputs,
    PanelMifState,
    PfilterConfig,
    PfilterInputs,
    UnitStepInputs,
)
from .mif import _perfilter_internal


def _panel_mif_internal(
    shared_array: jax.Array,
    unit_array: jax.Array,
    key: jax.Array,
    config: PanelMifConfig,
    inputs: PanelMifInputs,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Fully JIT-compatible panel iterated filtering across M iterations and U units.

    Returns
        shared_array_final: (J, n_shared)
        unit_array_final: (J, U, n_spec)
        shared_traces: (M+1, n_shared+1) where [:,0] is sum logLik per iter, [:,1:] are means
        unit_traces: (M+1, U, n_spec+1) where [:,:,0] is per-unit logLik per iter, [:,:,1:] are means
    """
    # 1. Setup metadata and initial traces.
    n_shared = shared_array.shape[1]
    n_spec = unit_array.shape[2]
    inv_perms = jax.vmap(jnp.argsort)(inputs.unit_param_permutations)

    shared_means0 = jnp.mean(shared_array, axis=0) if n_shared > 0 else jnp.zeros((0,))
    unit_means0 = (
        jnp.mean(unit_array, axis=0) if n_spec > 0 else jnp.zeros((config.U, 0))
    )

    shared_trace_0 = jnp.concatenate([jnp.array([jnp.nan]), shared_means0])[None, :]
    unit_trace_0 = jnp.concatenate(
        [jnp.full((config.U, 1), jnp.nan), unit_means0], axis=-1
    )[None, :, :]

    # 2. Prepare scan input.
    all_keys = jax.random.split(key, num=config.M + 1)
    m_keys = all_keys[1:]

    config_pf = config.to_pfilter_config()

    initial_iter_carry = PanelMifState(shared=shared_array, unit_specific=unit_array)
    iter_scan_xs = PanelMifIterInputs(m=jnp.arange(config.M), key=m_keys)

    iter_body_fn = jax.tree_util.Partial(
        _panel_mif_iter_body,
        config,
        inputs,
        config_pf,
        inv_perms,
    )

    # 3. Run the panel iterated filtering loop over M iterations.
    (final_iter_state, (shared_traces_history, unit_traces_history)) = jax.lax.scan(
        f=iter_body_fn,
        init=initial_iter_carry,
        xs=iter_scan_xs,
    )

    # 4. Collect results.
    shared_array = final_iter_state.shared
    unit_array = final_iter_state.unit_specific

    shared_traces = jnp.concatenate([shared_trace_0, shared_traces_history], axis=0)
    unit_traces = jnp.concatenate([unit_trace_0, unit_traces_history], axis=0)

    return (shared_array, unit_array, shared_traces, unit_traces)


def _panel_mif_iter_body(
    config: PanelMifConfig,
    inputs: PanelMifInputs,
    config_pf: PfilterConfig,
    inv_perms: jax.Array,
    carry: PanelMifState,
    scan_inputs: PanelMifIterInputs,
) -> tuple[PanelMifState, tuple[jax.Array, jax.Array]]:
    """Run one iteration of the panel iterated filtering loop over U units."""
    shared_array_m = carry.shared
    unit_array_m = carry.unit_specific
    m = scan_inputs.m
    iter_key = scan_inputs.key

    # 1. Prepare scan inputs for each unit.
    unit_keys = jax.random.split(iter_key, num=config.U)
    unit_scan_seq = UnitStepInputs(
        permutation=inputs.unit_param_permutations,
        ys=inputs.ys,
        covariates_dummy=inputs.covars_extended
        if inputs.covars_extended is not None
        else jnp.zeros((config.U, 0)),
        unit_idx=jnp.arange(config.U),
        key=unit_keys,
        inverse_permutation=inv_perms,
    )

    initial_inner_carry = PanelMifState(
        shared=shared_array_m,
        unit_specific=unit_array_m,
    )

    unit_scan_fn_partial = jax.tree_util.Partial(
        _panel_mif_unit_scan_fn,
        config,
        inputs,
        config_pf,
        m,
    )

    # 2. Run the scan across all units for this iteration.
    final_inner_carry, unit_traces_m_new_seq = jax.lax.scan(
        f=unit_scan_fn_partial,
        init=initial_inner_carry,
        xs=unit_scan_seq,
    )

    # 3. Extract parameters and compile iteration traces.
    shared_array_m = final_inner_carry.shared
    unit_array_m = final_inner_carry.unit_specific

    unit_traces_m_row = unit_traces_m_new_seq

    sum_loglik_iter = jnp.sum(unit_traces_m_new_seq[:, 0])

    n_shared = shared_array_m.shape[1]
    if n_shared > 0:
        shared_means = jnp.mean(shared_array_m, axis=0)
        shared_traces_m_row = jnp.concatenate(
            [jnp.array([sum_loglik_iter]), shared_means]
        )
    else:
        shared_traces_m_row = jnp.array([sum_loglik_iter])

    return PanelMifState(shared=shared_array_m, unit_specific=unit_array_m), (
        shared_traces_m_row,
        unit_traces_m_row,
    )


def _panel_mif_unit_scan_fn(
    config: PanelMifConfig,
    inputs: PanelMifInputs,
    config_pf: PfilterConfig,
    m: int | jax.Array,
    state: PanelMifState,
    unit: UnitStepInputs,
) -> tuple[PanelMifState, jax.Array]:
    """Run the panel iterated filtering step for a single unit."""
    n_shared = state.shared.shape[1]
    n_spec = state.unit_specific.shape[2]

    covariates = None if inputs.covars_extended is None else unit.covariates_dummy

    # 1. Reconstruct parameters for current unit into the canonical panel order
    if n_spec > 0:
        current_unit_thetas = state.unit_specific[:, unit.unit_idx, :]
    else:
        current_unit_thetas = jnp.zeros((config.J, 0))

    if (n_shared + n_spec) > 0:
        thetas_panel_order = jnp.concatenate(
            [state.shared, current_unit_thetas], axis=1
        )
    else:
        thetas_panel_order = jnp.zeros((config.J, 0))

    # 2. Permute parameters and sigmas to match the unit's local model order
    thetas_unit_order = thetas_panel_order[:, unit.permutation]
    rw_sigma_unit_order = inputs.rw_sigma._permuted(unit.permutation)

    # 3. Build sub-configurations and run the single-unit perfilter step
    mif_config = config.to_mif_config()

    mif_inputs = inputs.to_mif_inputs(
        ys_u=unit.ys,
        rw_sigma_u=rw_sigma_unit_order,
        covars_u=covariates,
    )

    updated_thetas_unit, neg_loglik_per, ancestry = _perfilter_internal(
        m_current=m,
        thetas_Jd=thetas_unit_order,
        key=unit.key,
        config=mif_config,
        inputs=mif_inputs,
    )

    # 4. Compute monitored log-likelihood (optional)
    if config.n_monitors >= 1:
        current_theta_mean = jnp.mean(thetas_unit_order, axis=0)
        mon_keys = jax.random.split(unit.key, config.n_monitors)
        inputs_pf = PfilterInputs(
            ys=unit.ys,
            dt_array_extended=inputs.dt_array_extended,
            nstep_array=inputs.nstep_array,
            t0=inputs.t0,
            times=inputs.times,
            covars_extended=covariates,
        )
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

    # 5. Map updated parameters back to the canonical panel order
    updated_thetas_panel = updated_thetas_unit[:, unit.inverse_permutation]

    # 6. Distribute updated parameters back into the global state
    if n_shared > 0:
        new_shared = updated_thetas_panel[:, :n_shared]
    else:
        new_shared = state.shared

    if n_spec > 0:
        updated_spec = updated_thetas_panel[:, n_shared:]

        # If not blocking, we must align the particle histories across units
        # based on the current unit's resampling ancestry.
        unit_specific_carry = state.unit_specific
        if not config.block:
            unit_specific_carry = unit_specific_carry[ancestry, :, :]

        new_unit_specific = unit_specific_carry.at[:, unit.unit_idx, :].set(
            updated_spec
        )
    else:
        updated_spec = current_unit_thetas
        new_unit_specific = state.unit_specific

    # 7. Package updated state and compute local metrics
    next_state = PanelMifState(shared=new_shared, unit_specific=new_unit_specific)

    log_likelihood = -neg_loglik_m
    unit_trace = (
        jnp.concatenate([jnp.array([log_likelihood]), jnp.mean(updated_spec, axis=0)])
        if n_spec > 0
        else jnp.array([log_likelihood])
    )

    return next_state, unit_trace


_vmapped_panel_mif_internal = jax.vmap(
    _panel_mif_internal,
    in_axes=(
        0,  # shared_array
        0,  # unit_array
        0,  # key
        None,  # config
        None,  # inputs
    ),
)

_jv_panel_mif_internal = jit(
    _vmapped_panel_mif_internal,
    static_argnames=("config",),
)
