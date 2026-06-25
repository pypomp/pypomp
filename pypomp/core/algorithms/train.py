from functools import partial
import jax
from jax import jit
import jax.numpy as jnp
from typing import Callable
from .pfilter import (
    _pfilter_internal,
    _vmapped_pfilter_internal,
)
from .mop import (
    _chunked_panel_mop_internal,
    _panel_mop_internal_vmap,
)
from .helpers import _cosine_cooling
from .ad_helpers import _jvg_mop, _jgrad_mop, _jhess_mop
from ..optimizer import (
    Optimizer,
    SGD,
    Adam,
    FullMatrixAdam,
    Newton,
    WeightedNewton,
    BFGS,
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
        "optimizer",
        "M",
        "n_obs",
        "U",
        "alpha_cooling",
    ),
)
def _panel_train_internal(
    shared_array: jax.Array,  # (n_shared,)
    unit_array: jax.Array,  # (U, n_spec)
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
    eta_shared: jax.Array,
    eta_spec: jax.Array,
    alpha: float,
    alpha_cooling: float,
    n_obs: int,  # ys.shape[1]
    U: int,  # ys.shape[0]
):
    if not isinstance(optimizer, (SGD, Adam, FullMatrixAdam)):
        raise ValueError(
            f"Optimizer '{optimizer.__class__.__name__}' not supported for panel train"
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
    unit_array_c = unit_array.reshape((n_chunks, chunk_size, -1))
    unit_param_permutations_c = unit_param_permutations.reshape(
        (n_chunks, chunk_size, -1)
    )

    def _chunk_obj(
        s_ests, u_ests, perm_chunk, ys_chunk, covars_chunk, keys_chunk, curr_alpha
    ):
        shared_tiled = jnp.tile(s_ests, (chunk_size, 1))
        theta_unordered = jnp.concatenate([shared_tiled, u_ests], axis=1)
        theta_chunk = jax.vmap(lambda t, p: t[p])(theta_unordered, perm_chunk)
        res = _panel_mop_internal_vmap(
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
            keys_chunk,
        )
        return jnp.sum(res) / (chunk_size * n_obs)

    def iteration_scan_step(carry, i):
        (shared_ests, unit_ests_c, opt_state_s, opt_state_u_c, global_step) = carry
        iter_keys_c = keys[i].reshape((n_chunks, chunk_size) + keys.shape[2:])

        def chunk_scan_step(chunk_carry, chunk_idx):
            c_s, c_opt_state_s, c_step = chunk_carry
            c_u = unit_ests_c[chunk_idx]

            c_opt_state_u = jax.tree.map(lambda x: x[chunk_idx], opt_state_u_c)

            curr_alpha = 1.0 - (1.0 - alpha) * _cosine_cooling(i, M, alpha_cooling)

            covars_chunk = None if covars_c is None else covars_c[chunk_idx]
            neg_loglik, (g_s, g_u) = jax.value_and_grad(_chunk_obj, argnums=(0, 1))(
                c_s,
                c_u,
                unit_param_permutations_c[chunk_idx],
                ys_c[chunk_idx],
                covars_chunk,
                iter_keys_c[chunk_idx],
                curr_alpha,
            )
            neg_loglik *= ylen

            # Adjusts for jnp.sum(res) / (chunk_size * n_obs) in _chunk_obj()
            g_u = g_u * chunk_size

            if optimizer.clip_norm is not None:
                g_s = jnp.clip(g_s, -optimizer.clip_norm, optimizer.clip_norm)
                g_u = jnp.clip(g_u, -optimizer.clip_norm, optimizer.clip_norm)

            dir_s, new_opt_state_s = optimizer.step(g_s, c_opt_state_s, c_step)
            dir_u, new_opt_state_u = optimizer.step(g_u, c_opt_state_u, i)

            if optimizer.scale:
                dir_s = dir_s / jnp.maximum(jnp.linalg.norm(dir_s), 1e-8)
                norm_u = jnp.linalg.norm(dir_u, axis=-1, keepdims=True)
                dir_u = dir_u / jnp.maximum(norm_u, 1e-8)

            c_s = c_s + (eta_shared[i] / n_chunks) * dir_s
            c_u = c_u + eta_spec[i] * dir_u
            return (c_s, new_opt_state_s, c_step + 1), (
                neg_loglik,
                c_u,
                new_opt_state_u,
            )

        (
            (final_s, final_opt_state_s, final_step),
            (chunk_neg_logliks, new_u_c, new_opt_state_u_c),
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
        unit_flat = new_u_c.reshape((-1, new_u_c.shape[-1]))
        return new_carry, (jnp.mean(chunk_neg_logliks), final_s, unit_flat)

    initial_carry = (
        shared_array,
        unit_array_c,
        optimizer.init_state(shared_array),
        optimizer.init_state(unit_array_c),
        0,
    )

    _, (neg_logliks, shared_copies, unit_copies) = jax.lax.scan(
        iteration_scan_step, initial_carry, jnp.arange(M)
    )

    neg_loglik_init = (
        _chunked_panel_mop_internal(
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
        )
        * ylen
    )

    neg_logliks = jnp.concatenate((jnp.array([neg_loglik_init]), neg_logliks))
    shared_copies = jnp.concatenate((shared_array[None, :], shared_copies), axis=0)
    unit_copies = jnp.concatenate((unit_array[None, :, :], unit_copies), axis=0)

    return neg_logliks, shared_copies, unit_copies


_vmapped_panel_train_internal = jax.vmap(
    _panel_train_internal,
    in_axes=(0, 0) + (None,) * 7 + (0,) + (None,) * 14,
)


@partial(
    jit,
    static_argnames=(
        "rinitializer",
        "rprocess_interp",
        "dmeasure",
        "J",
        "optimizer",
        "M",
        "thresh",
        "n_monitors",
    ),
)
def _train_internal(
    theta_ests: jax.Array,
    ys: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    rinitializer: Callable,  # static
    rprocess_interp: Callable,  # static
    dmeasure: Callable,  # static
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    J: int,  # static
    optimizer: Optimizer,  # static
    M: int,  # static
    eta: jax.Array,
    thresh: float,  # static
    alpha: float | jax.Array,
    key: jax.Array,
    alpha_cooling: float,
    n_monitors: int,  # static
):
    """
    Internal function for conducting the MOP gradient estimate method.
    """
    times = times.astype(float)
    ylen = ys.shape[0]
    if n_monitors < 1 and optimizer.ls:
        raise ValueError("Line search requires at least one monitor")

    if not isinstance(
        optimizer, (SGD, Adam, FullMatrixAdam, Newton, WeightedNewton, BFGS)
    ):
        raise ValueError(f"Optimizer '{optimizer.__class__.__name__}' not supported")

    def scan_step(carry, i):
        (
            theta_ests,
            key,
            opt_state,
        ) = carry

        curr_alpha = 1.0 - (1.0 - alpha) * _cosine_cooling(i, M, alpha_cooling)

        if n_monitors == 1:
            key, subkey = jax.random.split(key)
            neg_loglik, grad = _jvg_mop(
                theta_ests=theta_ests,
                ys=ys,
                dt_array_extended=dt_array_extended,
                nstep_array=nstep_array,
                t0=t0,
                times=times,
                J=J,
                rinitializer=rinitializer,
                rprocess=rprocess_interp,
                dmeasure=dmeasure,
                accumvars=accumvars,
                covars_extended=covars_extended,
                alpha=curr_alpha,
                key=subkey,
            )
            neg_loglik *= ylen
        else:
            key, subkey = jax.random.split(key)
            grad = _jgrad_mop(
                theta_ests=theta_ests,
                ys=ys,
                dt_array_extended=dt_array_extended,
                nstep_array=nstep_array,
                t0=t0,
                times=times,
                J=J,
                rinitializer=rinitializer,
                rprocess=rprocess_interp,
                dmeasure=dmeasure,
                accumvars=accumvars,
                covars_extended=covars_extended,
                alpha=curr_alpha,
                key=subkey,
            )
            if n_monitors > 0:
                key, *subkeys = jax.random.split(key, n_monitors + 1)
                neg_loglik = jnp.mean(
                    _vmapped_pfilter_internal(
                        theta_ests,
                        dt_array_extended,
                        nstep_array,
                        t0,
                        times,
                        ys,
                        J,
                        rinitializer,
                        rprocess_interp,
                        dmeasure,
                        accumvars,
                        covars_extended,
                        0,
                        jnp.array(subkeys),
                        False,
                        False,
                        False,
                        False,
                        True,
                    )["neg_loglik"]
                )
            else:
                neg_loglik = jnp.array(jnp.nan)

        if optimizer.clip_norm is not None:
            grad = jnp.clip(grad, -optimizer.clip_norm, optimizer.clip_norm)

        key, subkey_hess = jax.random.split(key)

        def compute_hessian():
            return _jhess_mop(
                theta_ests=theta_ests,
                ys=ys,
                dt_array_extended=dt_array_extended,
                nstep_array=nstep_array,
                t0=t0,
                times=times,
                J=J,
                rinitializer=rinitializer,
                rprocess=rprocess_interp,
                dmeasure=dmeasure,
                accumvars=accumvars,
                covars_extended=covars_extended,
                alpha=curr_alpha,
                key=subkey_hess,
            )

        direction, new_opt_state = optimizer.step(
            grad=grad,
            state=opt_state,
            step_num=i,
            compute_hessian_fn=compute_hessian,
            eta_i=eta[i],
        )

        if optimizer.scale:
            direction = direction / jnp.linalg.norm(direction)

        if optimizer.ls:

            def _obj_neg_loglik(theta):
                neg_loglik = _pfilter_internal(
                    theta,
                    dt_array_extended=dt_array_extended,
                    nstep_array=nstep_array,
                    t0=t0,
                    times=times,
                    ys=ys,
                    J=J,
                    rinitializer=rinitializer,
                    rprocess_interp=rprocess_interp,
                    dmeasure=dmeasure,
                    accumvars=accumvars,
                    covars_extended=covars_extended,
                    thresh=thresh,
                    key=subkey,
                    CLL=False,
                    ESS=False,
                    filter_mean=False,
                    prediction_mean=False,
                    should_trans=True,
                )["neg_loglik"]

                return jnp.squeeze(neg_loglik)

            eta_scalar = _line_search(
                _obj_neg_loglik,
                curr_obj=neg_loglik,
                pt=theta_ests,
                grad=grad,
                direction=direction,
                k=i + 1,
                eta=jnp.mean(eta[i]),
                xi=10,
                tau=optimizer.max_ls_itn,
                c=optimizer.c,
                frac=0.5,
                stoch=False,
            )
            theta_ests = theta_ests + (eta_scalar) * direction

        else:
            theta_ests = theta_ests + eta[i] * direction

        new_carry = (
            theta_ests,
            key,
            new_opt_state,
        )

        return new_carry, (neg_loglik, theta_ests)

    initial_carry = (
        theta_ests,
        key,
        optimizer.init_state(theta_ests),
    )

    _, (neg_logliks_history, Acopies_history) = jax.lax.scan(
        scan_step,
        initial_carry,
        jnp.arange(M),
    )

    neg_logliks = jnp.concatenate((jnp.array([jnp.nan]), neg_logliks_history))
    Acopies = jnp.concatenate((theta_ests[jnp.newaxis, ...], Acopies_history))

    return neg_logliks, Acopies


# Map over theta and key
_vmapped_train_internal = jax.vmap(
    _train_internal,
    in_axes=(0,) + (None,) * 16 + (0,) + (None,) * 2,
)


def _line_search(
    obj: Callable,
    curr_obj: jax.Array,
    pt: jax.Array,
    grad: jax.Array,
    direction: jax.Array,
    k: int,
    eta: jax.Array,
    xi: int,
    tau: int,
    c: float,
    frac: float,
    stoch: bool,
) -> jax.Array:
    """
    Conducts line search algorithm to determine the step size under stochastic
    Quasi-Newton methods. The implentation of the algorithm refers to
    https://arxiv.org/pdf/1909.01238.pdf.

    Args:
        obj (function): The objective function aiming to minimize
        curr_obj (jax.Array): The value of the objective function at the current
            point.
        pt (jax.Array): The array containing current parameter values.
        grad (jax.Array): The gradient of the objective function at the current
            point.
        direction (jax.Array): The direction to update the parameters.
        k (int, optional): Iteration index.
        eta (float, optional): Initial step size.
        xi (int, optional): Reduction limit.
        tau (int, optional): The maximum number of iterations.
        c (float, optional): The user-defined Armijo condition constant.
        frac (float, optional): The fraction of the step size to reduce by each
            iteration.
        stoch (bool, optional): Boolean argument controlling whether to adjust
            the initial step size.

    Returns:
        jax.Array: optimal step size
    """
    eta = jnp.where(stoch, jnp.minimum(eta, xi / k), eta)
    # check whether the new point(new_obj)satisfies the stochastic Armijo condition
    # if not, repeat until the condition is met
    # previous: grad.T @ direction

    def line_search_body(carry):
        eta_val, itn, should_continue = carry
        next_obj = obj(pt + eta_val * direction)
        should_continue = (
            next_obj > curr_obj + eta_val * c * jnp.sum(grad * direction)
        ) | jnp.isnan(next_obj)
        eta_new = jnp.where(should_continue & (itn < tau), eta_val * frac, eta_val)
        itn_new = itn + 1
        return eta_new, itn_new, should_continue & (itn < tau)

    eta_final, _, _ = jax.lax.while_loop(
        lambda carry: carry[2], line_search_body, (eta, 0, True)
    )
    return eta_final
