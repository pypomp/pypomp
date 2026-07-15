from functools import partial
from dataclasses import replace
import jax
from jax import jit
import jax.numpy as jnp
from typing import Callable, cast
from .pfilter import (
    _pfilter_internal,
    _vmapped_pfilter_internal,
)
from .mop import (
    _panel_mop_internal_vmap,
)
from .types import (
    PanelTrainConfig,
    PanelTrainInputs,
    PanelTrainState,
    ChunkState,
    ChunkMetrics,
    IterationMetrics,
    TrainConfig,
    TrainInputs,
    TrainState,
    TrainMetrics,
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
        "config",
        "optimizer",
    ),
)
def _train_internal(
    theta_ests: jax.Array,
    key: jax.Array,
    config: TrainConfig,
    inputs: TrainInputs,
    optimizer: Optimizer,
) -> tuple[jax.Array, jax.Array]:
    if config.n_monitors < 1 and optimizer.ls:
        raise ValueError("Line search requires at least one monitor")

    if not isinstance(
        optimizer, (SGD, Adam, FullMatrixAdam, Newton, WeightedNewton, BFGS)
    ):
        raise ValueError(f"Optimizer '{optimizer.__class__.__name__}' not supported")

    initial_carry = TrainState(
        theta_ests=theta_ests,
        key=key,
        opt_state=optimizer.init_state(theta_ests),
    )
    step_fn = jax.tree_util.Partial(
        _train_scan_step,
        config,
        optimizer,
        inputs,
    )

    _, history = jax.lax.scan(
        step_fn,
        initial_carry,
        jnp.arange(config.M),
    )

    neg_logliks = jnp.concatenate((jnp.array([jnp.nan]), history.neg_loglik))
    Acopies = jnp.concatenate((theta_ests[jnp.newaxis, ...], history.theta_ests))

    return neg_logliks, Acopies


def _train_scan_step(
    config: TrainConfig,
    optimizer: Optimizer,
    inputs: TrainInputs,
    carry: TrainState,
    m: int,
) -> tuple[TrainState, TrainMetrics]:
    theta_ests = carry.theta_ests
    key = carry.key
    opt_state = carry.opt_state

    alpha_m = 1.0 - (1.0 - inputs.alpha) * _cosine_cooling(
        m, config.M, config.alpha_cooling
    )
    inputs_m = replace(inputs, alpha=alpha_m)

    if config.n_monitors == 1:
        key, subkey = jax.random.split(key)
        neg_loglik, grad = _jvg_mop(
            theta_ests,
            subkey,
            config,
            inputs_m,
        )
        ylen = inputs.ys.shape[0]
        neg_loglik *= ylen
    else:
        key, subkey = jax.random.split(key)
        grad = _jgrad_mop(
            theta_ests,
            subkey,
            config,
            inputs_m,
        )
        if config.n_monitors > 0:
            key, *subkeys = jax.random.split(key, config.n_monitors + 1)
            neg_loglik = jnp.mean(
                _vmapped_pfilter_internal(
                    theta_ests,
                    jnp.array(subkeys),
                    config.to_pfilter_config(should_trans=True),
                    inputs.to_pfilter_inputs(),
                )["neg_loglik"]
            )
        else:
            neg_loglik = jnp.array(jnp.nan)

    if optimizer.clip_norm is not None:
        grad = jnp.clip(grad, -optimizer.clip_norm, optimizer.clip_norm)

    key, subkey_hess = jax.random.split(key)

    # this is only run if the optimizer.step code uses it
    def compute_hessian():
        return _jhess_mop(
            theta_ests,
            subkey_hess,
            config,
            inputs_m,
        )

    direction, new_opt_state = optimizer.step(
        grad=grad,
        state=opt_state,
        step_num=m,
        compute_hessian_fn=compute_hessian,
        eta_i=inputs.eta[m],
    )

    if optimizer.scale:
        direction = direction / jnp.linalg.norm(direction)

    if optimizer.ls:

        def _obj_neg_loglik(theta):
            neg_loglik_val = _pfilter_internal(
                theta,
                subkey,
                config.to_pfilter_config(should_trans=True),
                inputs.to_pfilter_inputs(),
            )["neg_loglik"]

            return jnp.squeeze(neg_loglik_val)

        eta_scalar = _line_search(
            _obj_neg_loglik,
            curr_obj=neg_loglik,
            pt=theta_ests,
            grad=grad,
            direction=direction,
            k=m + 1,
            eta=jnp.mean(inputs.eta[m]),
            xi=10,
            tau=optimizer.max_ls_itn,
            c=optimizer.c,
            frac=0.5,
            stoch=False,
        )
        theta_ests = theta_ests + (eta_scalar) * direction

    else:
        theta_ests = theta_ests + inputs.eta[m] * direction

    new_carry = TrainState(
        theta_ests=theta_ests,
        key=key,
        opt_state=new_opt_state,
    )

    metrics = TrainMetrics(
        neg_loglik=neg_loglik,
        theta_ests=theta_ests,
    )

    return new_carry, metrics


# Map over theta and key
_vmapped_train_internal = jax.vmap(
    _train_internal,
    in_axes=(0, 0, None, None, None),
)


@partial(
    jit,
    static_argnames=(
        "config",
        "optimizer",
    ),
)
def _panel_train_internal(
    shared_array: jax.Array,
    unit_array: jax.Array,
    config: PanelTrainConfig,
    inputs: PanelTrainInputs,
    optimizer: Optimizer,
):
    if not isinstance(optimizer, (SGD, Adam, FullMatrixAdam)):
        raise ValueError(
            f"Optimizer '{optimizer.__class__.__name__}' not supported for panel train"
        )

    n_chunks = (config.U + config.chunk_size - 1) // config.chunk_size

    # Reshape for chunk-wise processing, which vectorizes more
    ys_c = inputs.ys.reshape((n_chunks, config.chunk_size, config.n_obs, -1))
    covars_c = (
        None
        if inputs.covars_extended is None
        else inputs.covars_extended.reshape(
            (n_chunks, config.chunk_size) + inputs.covars_extended.shape[1:]
        )
    )
    unit_array_c = unit_array.reshape((n_chunks, config.chunk_size, -1))
    unit_param_permutations_c = inputs.unit_param_permutations.reshape(
        (n_chunks, config.chunk_size, -1)
    )

    initial_carry = PanelTrainState(
        shared_ests=shared_array,
        unit_ests_chunked=unit_array_c,
        opt_state_shared=optimizer.init_state(shared_array),
        opt_state_unit_chunked=optimizer.init_state(unit_array_c),
        global_step=0,
    )

    step_fn = jax.tree_util.Partial(
        _iteration_scan_step,
        config,
        optimizer,
        inputs,
        ys_c,
        covars_c,
        unit_param_permutations_c,
    )

    _, history = jax.lax.scan(
        step_fn,
        initial_carry,
        jnp.arange(config.M),
    )

    neg_loglik_init = jnp.nan

    neg_logliks = jnp.concatenate((jnp.array([neg_loglik_init]), history.neg_loglik))
    shared_copies = jnp.concatenate(
        (shared_array[None, :], history.shared_ests), axis=0
    )
    unit_copies = jnp.concatenate((unit_array[None, :, :], history.unit_ests), axis=0)

    return neg_logliks, shared_copies, unit_copies


def _iteration_scan_step(
    config: PanelTrainConfig,
    optimizer: Optimizer,
    inputs: PanelTrainInputs,
    ys_c: jax.Array,
    covars_c: jax.Array | None,
    unit_param_permutations_c: jax.Array,
    carry: PanelTrainState,
    m: int,
) -> tuple[PanelTrainState, IterationMetrics]:
    """Performs gradient descent step across chunks."""
    n_chunks = (config.U + config.chunk_size - 1) // config.chunk_size
    iter_keys_c = inputs.keys[m].reshape(
        (n_chunks, config.chunk_size) + inputs.keys.shape[2:]
    )

    chunk_step_fn = jax.tree_util.Partial(
        _chunk_scan_step,
        config,
        optimizer,
        inputs,
        ys_c,
        covars_c,
        unit_param_permutations_c,
        iter_keys_c,
        carry.unit_ests_chunked,
        carry.opt_state_unit_chunked,
        m,
    )
    initial_chunk_carry = ChunkState(
        shared_ests=carry.shared_ests,
        opt_state_shared=carry.opt_state_shared,
        global_step=carry.global_step,
    )

    final_chunk_carry, chunk_metrics = jax.lax.scan(
        chunk_step_fn,
        initial_chunk_carry,
        jnp.arange(n_chunks),
    )

    new_carry = PanelTrainState(
        shared_ests=final_chunk_carry.shared_ests,
        unit_ests_chunked=chunk_metrics.unit_ests_chunk,
        opt_state_shared=final_chunk_carry.opt_state_shared,
        opt_state_unit_chunked=chunk_metrics.opt_state_unit_chunk,
        global_step=final_chunk_carry.global_step,
    )
    unit_flat = chunk_metrics.unit_ests_chunk.reshape(
        (
            chunk_metrics.unit_ests_chunk.shape[0]
            * chunk_metrics.unit_ests_chunk.shape[1],
            -1,
        )
    )
    iter_metrics = IterationMetrics(
        neg_loglik=jnp.mean(chunk_metrics.neg_loglik),
        shared_ests=final_chunk_carry.shared_ests,
        unit_ests=unit_flat,
    )
    return new_carry, iter_metrics


def _chunk_scan_step(
    config: PanelTrainConfig,
    optimizer: Optimizer,
    inputs: PanelTrainInputs,
    ys_c: jax.Array,
    covars_c: jax.Array | None,
    unit_param_permutations_c: jax.Array,
    iter_keys_c: jax.Array,
    unit_ests_chunked: jax.Array,
    opt_state_unit_chunked: jax.Array,
    m: int,
    chunk_carry: ChunkState,
    chunk_idx: int,
) -> tuple[ChunkState, ChunkMetrics]:
    """Performs gradient descent step for one chunk."""
    curr_shared_ests = chunk_carry.shared_ests
    curr_opt_state_shared = chunk_carry.opt_state_shared
    curr_global_step = chunk_carry.global_step

    curr_unit_ests_chunk = unit_ests_chunked[chunk_idx]
    curr_opt_state_unit = jax.tree.map(lambda x: x[chunk_idx], opt_state_unit_chunked)

    alpha_m = 1.0 - (1.0 - inputs.alpha) * _cosine_cooling(
        m, config.M, config.alpha_cooling
    )

    covars_chunk = None if covars_c is None else covars_c[chunk_idx]

    neg_loglik, (grad_shared, grad_unit) = jax.value_and_grad(
        _compute_chunk_loss, argnums=(0, 1)
    )(
        curr_shared_ests,
        curr_unit_ests_chunk,
        unit_param_permutations_c[chunk_idx],
        ys_c[chunk_idx],
        covars_chunk,
        iter_keys_c[chunk_idx],
        alpha_m,
        config,
        inputs,
    )

    ylen = config.n_obs * config.U
    neg_loglik *= ylen
    grad_unit = grad_unit * config.chunk_size

    if optimizer.clip_norm is not None:
        grad_shared = jnp.clip(grad_shared, -optimizer.clip_norm, optimizer.clip_norm)
        grad_unit = jnp.clip(grad_unit, -optimizer.clip_norm, optimizer.clip_norm)

    dir_shared, new_opt_state_shared = optimizer.step(
        grad_shared, curr_opt_state_shared, curr_global_step
    )
    dir_unit, new_opt_state_unit = optimizer.step(grad_unit, curr_opt_state_unit, m)

    if optimizer.scale:
        dir_shared = dir_shared / jnp.maximum(jnp.linalg.norm(dir_shared), 1e-8)
        norm_unit = jnp.linalg.norm(dir_unit, axis=-1, keepdims=True)
        dir_unit = dir_unit / jnp.maximum(norm_unit, 1e-8)

    n_chunks = (config.U + config.chunk_size - 1) // config.chunk_size
    curr_shared_ests = curr_shared_ests + (inputs.eta_shared[m] / n_chunks) * dir_shared
    curr_unit_ests_chunk = curr_unit_ests_chunk + inputs.eta_spec[m] * dir_unit

    new_chunk_carry = ChunkState(
        shared_ests=curr_shared_ests,
        opt_state_shared=new_opt_state_shared,
        global_step=curr_global_step + 1,
    )
    chunk_metrics = ChunkMetrics(
        neg_loglik=neg_loglik,
        unit_ests_chunk=curr_unit_ests_chunk,
        opt_state_unit_chunk=new_opt_state_unit,
    )
    return new_chunk_carry, chunk_metrics


def _compute_chunk_loss(
    s_ests: jax.Array,
    u_ests: jax.Array,
    perm_chunk: jax.Array,
    ys_chunk: jax.Array,
    covars_chunk: jax.Array | None,
    keys_chunk: jax.Array,
    curr_alpha: float,
    config: PanelTrainConfig,
    inputs: PanelTrainInputs,
) -> jax.Array:
    shared_tiled = jnp.tile(s_ests, (config.chunk_size, 1))
    theta_unordered = jnp.concatenate([shared_tiled, u_ests], axis=1)
    theta_chunk = jax.vmap(lambda t, p: t[p])(theta_unordered, perm_chunk)
    mop_config = config.to_mop_config()
    mop_inputs = replace(
        inputs.to_mop_inputs(),
        ys=ys_chunk,
        covars_extended=covars_chunk,
        alpha=curr_alpha,
    )
    res = _panel_mop_internal_vmap(
        theta_chunk,
        keys_chunk,
        mop_config,
        mop_inputs,
    )
    return jnp.sum(res) / (config.chunk_size * config.n_obs)


# vmap axes spec: None = broadcast, 0 = map over first axis
inputs_in_axes = PanelTrainInputs(
    unit_param_permutations=cast(jax.Array, None),
    dt_array_extended=cast(jax.Array, None),
    nstep_array=cast(jax.Array, None),
    t0=cast(float, None),
    times=cast(jax.Array, None),
    ys=cast(jax.Array, None),
    covars_extended=cast(jax.Array, None),
    keys=cast(jax.Array, 0),
    eta_shared=cast(jax.Array, None),
    eta_spec=cast(jax.Array, None),
    alpha=cast(float, None),
)

_vmapped_panel_train_internal = jax.vmap(
    _panel_train_internal,
    in_axes=(0, 0, None, inputs_in_axes, None),
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
