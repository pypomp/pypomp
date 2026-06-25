from functools import partial
import jax
from jax import jit
import jax.numpy as jnp
from typing import Callable
from .dpop import _dpop_internal
from .helpers import _cosine_cooling
from ..optimizer import Optimizer, SGD, Adam


_panel_dpop_internal_vmap = jax.vmap(
    _dpop_internal,
    in_axes=(
        0,  # theta (per unit)
        0,  # ys (per unit)
        None,  # dt_array_extended
        None,  # nstep_array
        None,  # t0
        None,  # times
        None,  # J
        None,  # rinitializer
        None,  # rprocess_interp
        None,  # dmeasure
        None,  # accumvars
        0,  # covars_extended (per unit)
        None,  # alpha
        None,  # process_weight_index
        None,  # ntimes
        0,  # key (per unit)
    ),
)


@partial(
    jit,
    static_argnames=(
        "J",
        "rinitializer",
        "rprocess_interp",
        "dmeasure",
        "accumvars",
        "chunk_size",
        "process_weight_index",
        "ntimes",
    ),
)
def _chunked_panel_dpop_internal(
    shared_array: jax.Array,  # (n_shared,)
    unit_array: jax.Array,  # (n_spec, U)
    unit_param_permutations: jax.Array,  # (U, n_params)
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    covars_extended: jax.Array | None,
    keys: jax.Array,
    J: int,
    rinitializer: Callable,
    rprocess_interp: Callable,
    dmeasure: Callable,
    accumvars: tuple[int, ...] | None,
    chunk_size: int,
    alpha: float,
    process_weight_index: int | None,
    ntimes: int,
):
    U = unit_array.shape[1]
    n_params = unit_param_permutations.shape[1]
    n_chunks = U // chunk_size

    ys_c = ys.reshape((n_chunks, chunk_size) + ys.shape[1:])
    covars_c = (
        None
        if covars_extended is None
        else covars_extended.reshape((n_chunks, chunk_size) + covars_extended.shape[1:])
    )
    keys_c = keys.reshape((n_chunks, chunk_size) + keys.shape[1:])

    # unit_array: (n_spec, U) -> (n_chunks, chunk_size, n_spec)
    unit_array_c = unit_array.T.reshape((n_chunks, chunk_size, -1))

    # unit_param_permutations: (U, n_params) -> (n_chunks, chunk_size, n_params)
    unit_param_permutations_c = unit_param_permutations.reshape(
        (n_chunks, chunk_size, n_params)
    )

    shared_tiled = jnp.tile(shared_array, (chunk_size, 1))

    def scan_fn(carry, chunk_idx):
        unit_array_chunk = unit_array_c[chunk_idx]  # (chunk_size, n_spec)
        unit_param_perm_chunk = unit_param_permutations_c[
            chunk_idx
        ]  # (chunk_size, n_params)

        theta_chunk_unordered = jnp.concatenate(
            [shared_tiled, unit_array_chunk], axis=1
        )

        def apply_perm(theta, perm):
            return theta[perm]

        theta_chunk = jax.vmap(apply_perm)(theta_chunk_unordered, unit_param_perm_chunk)

        ys_chunk = ys_c[chunk_idx]
        covars_chunk = None if covars_c is None else covars_c[chunk_idx]
        key_chunk = keys_c[chunk_idx]

        res = _panel_dpop_internal_vmap(
            theta_chunk,
            ys_chunk,
            dt_array_extended,
            nstep_array,
            t0,
            times,
            J,
            rinitializer,
            rprocess_interp,
            dmeasure,
            accumvars,
            covars_chunk,
            alpha,
            process_weight_index,
            ntimes,
            key_chunk,
        )
        return carry + jnp.sum(res), None

    total_neg_loglik, _ = jax.lax.scan(scan_fn, 0.0, jnp.arange(n_chunks))

    return total_neg_loglik / (U * ntimes)


@partial(
    jit,
    static_argnames=(
        "J",
        "rinitializer",
        "rprocess_interp",
        "dmeasure",
        "accumvars",
        "chunk_size",
        "optimizer",
        "M",
        "n_obs",
        "U",
        "process_weight_index",
        "ntimes",
    ),
)
def _panel_dpop_train_internal(
    shared_array: jax.Array,  # (n_shared,)
    unit_array: jax.Array,  # (n_spec, U)
    unit_param_permutations: jax.Array,  # (U, n_params)
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    covars_extended: jax.Array | None,
    keys: jax.Array,  # (M, U, ...)
    J: int,
    rinitializer: Callable,
    rprocess_interp: Callable,
    dmeasure: Callable,
    accumvars: tuple[int, ...] | None,
    chunk_size: int,
    optimizer: Optimizer,
    M: int,
    eta_shared: jax.Array,  # (M, n_shared) per-iteration LR schedule
    eta_spec: jax.Array,  # (M, n_spec)   per-iteration LR schedule
    alpha: float,
    alpha_cooling: float,
    n_obs: int,  # ys.shape[1]
    U: int,  # ys.shape[0]
    process_weight_index: int | None,
    ntimes: int,
    decay: float,
):
    if not isinstance(optimizer, (SGD, Adam)):
        raise ValueError(
            f"Optimizer '{optimizer.__class__.__name__}' not supported for panel dpop_train"
        )

    times = times.astype(float)
    ylen = n_obs * U
    n_chunks = (U + chunk_size - 1) // chunk_size

    ys_c = ys.reshape((n_chunks, chunk_size, n_obs, -1))
    covars_c = (
        None
        if covars_extended is None
        else covars_extended.reshape((n_chunks, chunk_size) + covars_extended.shape[1:])
    )
    unit_array_c = unit_array.T.reshape((n_chunks, chunk_size, -1))
    unit_param_permutations_c = unit_param_permutations.reshape(
        (n_chunks, chunk_size, -1)
    )

    def _chunk_obj(
        s_ests, u_ests, perm_chunk, ys_chunk, covars_chunk, curr_alpha, keys_chunk
    ):
        shared_tiled = jnp.tile(s_ests, (chunk_size, 1))
        theta_unordered = jnp.concatenate([shared_tiled, u_ests], axis=1)
        theta_chunk = jax.vmap(lambda t, p: t[p])(theta_unordered, perm_chunk)
        res = _panel_dpop_internal_vmap(
            theta_chunk,
            ys_chunk,
            dt_array_extended,
            nstep_array,
            t0,
            times,
            J,
            rinitializer,
            rprocess_interp,
            dmeasure,
            accumvars,
            covars_chunk,
            curr_alpha,
            process_weight_index,
            ntimes,
            keys_chunk,
        )
        return jnp.sum(res) / (chunk_size * n_obs)

    def iteration_scan_step(carry, i):
        (shared_ests, unit_ests_c, opt_state_s, opt_state_u_c, global_step) = carry
        iter_keys_c = keys[i].reshape((n_chunks, chunk_size) + keys.shape[2:])

        # Learning rate decay
        i_f = i.astype(jnp.float32)
        lr_scale = 1.0 / (1.0 + decay * i_f)
        eta_shared_scaled = eta_shared[i] * lr_scale
        eta_spec_scaled = eta_spec[i] * lr_scale
        curr_alpha = 1.0 - (1.0 - alpha) * _cosine_cooling(i, M, alpha_cooling)

        def chunk_scan_step(chunk_carry, chunk_idx):
            c_s, c_opt_state_s, c_step = chunk_carry
            c_u = unit_ests_c[chunk_idx]

            c_opt_state_u = jax.tree.map(lambda x: x[chunk_idx], opt_state_u_c)

            covars_chunk = None if covars_c is None else covars_c[chunk_idx]
            loglik, (g_s, g_u) = jax.value_and_grad(_chunk_obj, argnums=(0, 1))(
                c_s,
                c_u,
                unit_param_permutations_c[chunk_idx],
                ys_c[chunk_idx],
                covars_chunk,
                curr_alpha,
                iter_keys_c[chunk_idx],
            )
            loglik *= ylen

            # Adjusts for jnp.sum(res) / (chunk_size * n_obs) in _chunk_obj()
            g_u = g_u * chunk_size

            dir_s, new_opt_state_s = optimizer.step(g_s, c_opt_state_s, c_step)
            dir_u, new_opt_state_u = optimizer.step(g_u, c_opt_state_u, i)

            c_s = c_s + (eta_shared_scaled / n_chunks) * dir_s
            c_u = c_u + eta_spec_scaled * dir_u
            return (c_s, new_opt_state_s, c_step + 1), (loglik, c_u, new_opt_state_u)

        (
            (final_s, final_opt_state_s, final_step),
            (chunk_lls, new_u_c, new_opt_state_u_c),
        ) = jax.lax.scan(
            chunk_scan_step,
            (shared_ests, opt_state_s, global_step),
            jnp.arange(n_chunks),
        )

        new_carry = (
            final_s,
            new_u_c,
            final_opt_state_s,
            new_opt_state_u_c,
            final_step,
        )
        unit_flat = new_u_c.reshape((-1, new_u_c.shape[-1])).T
        return new_carry, (jnp.mean(chunk_lls), final_s, unit_flat)

    initial_carry = (
        shared_array,
        unit_array_c,
        optimizer.init_state(shared_array),
        optimizer.init_state(unit_array_c),
        0,
    )

    _, (logliks, shared_copies, unit_copies) = jax.lax.scan(
        iteration_scan_step, initial_carry, jnp.arange(M)
    )

    loglik_init = (
        _chunked_panel_dpop_internal(
            shared_array,
            unit_array,
            unit_param_permutations,
            dt_array_extended,
            nstep_array,
            t0,
            times,
            ys,
            covars_extended,
            keys[0],
            J,
            rinitializer,
            rprocess_interp,
            dmeasure,
            accumvars,
            chunk_size,
            alpha,
            process_weight_index,
            ntimes,
        )
        * ylen
    )

    logliks = jnp.concatenate((jnp.array([loglik_init]), logliks))
    shared_copies = jnp.concatenate((shared_array[None, :], shared_copies), axis=0)
    unit_copies = jnp.concatenate((unit_array[None, :, :], unit_copies), axis=0)

    return logliks, shared_copies, unit_copies


_vmapped_panel_dpop_train_internal = jax.vmap(
    _panel_dpop_train_internal,
    in_axes=(0, 0) + (None,) * 7 + (0,) + (None,) * 17,
)
