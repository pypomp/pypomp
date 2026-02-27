from functools import partial
import jax
from jax import jit
import jax.numpy as jnp
from typing import Callable
from .pfilter import (
    _pfilter_internal,
    _vmapped_pfilter_internal,
    _pfilter_internal_mean,
)
from .mop import _mop_internal_mean, _mop_internal
from .dpop import (
    _dpop_internal_mean,
)  # DPOP mean negative log-likelihood per observation


_grad_pfilter_internal_mean = jax.grad(_pfilter_internal_mean)
_vg_pfilter_internal_mean = jax.value_and_grad(_pfilter_internal_mean)
_hess_pfilter_internal_mean = jax.hessian(_pfilter_internal_mean)
_grad_mop_internal_mean = jax.grad(_mop_internal_mean)
_vg_mop_internal_mean = jax.value_and_grad(_mop_internal_mean)
_hess_mop_internal_mean = jax.hessian(_mop_internal_mean)


_panel_mop_internal_vmap = jax.vmap(
    _mop_internal,
    in_axes=(
        0,  # theta
        0,  # ys
        None,  # dt_array_extended
        None,  # nstep_array
        None,  # t0
        None,  # times
        None,  # J
        None,  # rinitializer
        None,  # rprocess_interp
        None,  # dmeasure
        None,  # accumvars
        0,  # covars_extended
        None,  # alpha
        0,  # key
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
    ),
)
def _chunked_panel_mop_internal(
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
            alpha,
            key_chunk,
        )
        return carry + jnp.sum(res), None

    total_neg_loglik, _ = jax.lax.scan(scan_fn, 0.0, jnp.arange(n_chunks))

    return total_neg_loglik / (U * ys.shape[1])


_vg_chunked_panel_mop_internal = jax.value_and_grad(
    _chunked_panel_mop_internal, argnums=(0, 1)
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
    ),
)
def _panel_train_internal(
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
    optimizer: str,
    M: int,
    eta_shared: jax.Array,
    eta_spec: jax.Array,
    alpha: float,
    n_obs: int,  # ys.shape[1]
    U: int,  # ys.shape[0]
):
    times = times.astype(float)
    ylen = n_obs * U

    def scan_step(carry, i):
        (
            shared_ests,
            unit_ests,
            m_adam_shared,
            v_adam_shared,
            m_adam_unit,
            v_adam_unit,
        ) = carry

        iter_keys = keys[i]  # (U, ...)

        loglik, (grad_shared, grad_unit) = _vg_chunked_panel_mop_internal(
            shared_ests,
            unit_ests,
            unit_param_permutations,
            dt_array_extended,
            nstep_array,
            t0,
            times,
            ys,
            covars_extended,
            iter_keys,
            J,
            rinitializer,
            rprocess_interp,
            dmeasure,
            accumvars,
            chunk_size,
            alpha,
        )

        loglik *= ylen

        if optimizer == "SGD":
            direction_shared = -grad_shared
            direction_unit = -grad_unit

        elif optimizer == "Adam":
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8

            m_adam_shared = beta1 * m_adam_shared + (1 - beta1) * grad_shared
            v_adam_shared = beta2 * v_adam_shared + (1 - beta2) * (grad_shared**2)
            m_hat_shared = m_adam_shared / (1 - beta1 ** (i + 1))
            v_hat_shared = v_adam_shared / (1 - beta2 ** (i + 1))
            direction_shared = -m_hat_shared / (jnp.sqrt(v_hat_shared) + epsilon)

            m_adam_unit = beta1 * m_adam_unit + (1 - beta1) * grad_unit
            v_adam_unit = beta2 * v_adam_unit + (1 - beta2) * (grad_unit**2)
            m_hat_unit = m_adam_unit / (1 - beta1 ** (i + 1))
            v_hat_unit = v_adam_unit / (1 - beta2 ** (i + 1))
            direction_unit = -m_hat_unit / (jnp.sqrt(v_hat_unit) + epsilon)
        else:
            raise ValueError(f"Optimizer '{optimizer}' not supported for panel train")

        shared_ests = shared_ests + eta_shared * direction_shared

        # unit_ests: (n_spec, U)
        # direction_unit: (n_spec, U)
        # eta_spec: (n_spec,) -> expand dims to (n_spec, 1)
        unit_ests = unit_ests + eta_spec[:, None] * direction_unit

        new_carry = (
            shared_ests,
            unit_ests,
            m_adam_shared,
            v_adam_shared,
            m_adam_unit,
            v_adam_unit,
        )

        return new_carry, (loglik, shared_ests, unit_ests)

    initial_carry = (
        shared_array,
        unit_array,
        jnp.zeros_like(shared_array),
        jnp.zeros_like(shared_array),
        jnp.zeros_like(unit_array),
        jnp.zeros_like(unit_array),
    )

    _, (logliks_history, shared_history, unit_history) = jax.lax.scan(
        scan_step,
        initial_carry,
        jnp.arange(M),
    )

    logliks = jnp.concatenate((jnp.array([jnp.nan]), logliks_history))

    # shared_history: (M, n_shared)
    # shared_array: (n_shared,) -> prepend to shared_history -> (M+1, n_shared)
    shared_copies = jnp.concatenate((shared_array[jnp.newaxis, ...], shared_history))

    # unit_history: (M, n_spec, U) -> (M, (n_spec * U))? No, keep it (M, n_spec, U).
    # unit_array: (n_spec, U) -> prepend to unit_history -> (M+1, n_spec, U)
    unit_copies = jnp.concatenate((unit_array[jnp.newaxis, ...], unit_history))

    return logliks, shared_copies, unit_copies


_vmapped_panel_train_internal = jax.vmap(
    _panel_train_internal,
    in_axes=(0, 0) + (None,) * 7 + (0,) + (None,) * 13,
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
        "c",
        "max_ls_itn",
        "thresh",
        "scale",
        "ls",
        "alpha",
        "n_monitors",
        "n_obs",
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
    optimizer: str,  # static
    M: int,  # static
    eta: jax.Array,
    c: float,  # static
    max_ls_itn: int,  # static
    thresh: float,  # static
    scale: bool,  # static
    ls: bool,  # static
    alpha: float,  # static
    key: jax.Array,
    n_monitors: int,  # static
    n_obs: int,  # static
):
    """
    Internal function for conducting the MOP gradient estimate method.
    """
    times = times.astype(float)
    ylen = ys.shape[0]
    if n_monitors < 1 and ls:
        raise ValueError("Line search requires at least one monitor")

    def scan_step(carry, i):
        (
            theta_ests,
            key,
            hess,
            prev_grad,
            prev_hess,
            m_adam,
            v_adam,
        ) = carry

        if n_monitors == 1:
            key, subkey = jax.random.split(key)
            loglik, grad = _jvg_mop(
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
                alpha=alpha,
                key=subkey,
            )
            loglik *= ylen
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
                alpha=alpha,
                key=subkey,
            )
            if n_monitors > 0:
                # TODO: need to handle parameter transformations correctly when n_monitors > 1; currently, pfilter will not transform the parameters back to the natural scale, so the logLiks should be incorrect.
                key, *subkeys = jax.random.split(key, n_monitors + 1)
                loglik = jnp.mean(
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
                    )["neg_loglik"]
                )
            else:
                loglik = jnp.array(jnp.nan)

        if optimizer == "Newton":
            key, subkey = jax.random.split(key)
            hess = _jhess_mop(
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
                alpha=alpha,
                key=subkey,
            )
            direction = -jnp.linalg.pinv(hess, hermitian=True) @ grad

        elif optimizer == "WeightedNewton":
            key, subkey = jax.random.split(key)
            hess = _jhess_mop(
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
                alpha=alpha,
                key=subkey,
            )

            def dir_weighted(_):
                i_f = i.astype(theta_ests.dtype)
                wt = (i_f ** jnp.log(i_f)) / ((i_f + 1) ** jnp.log(i_f + 1))
                weighted_hess = wt * prev_hess + (1 - wt) * hess
                return -jnp.linalg.pinv(weighted_hess, hermitian=True) @ grad

            direction = jax.lax.cond(
                i == 0,
                lambda _: -jnp.linalg.pinv(hess, hermitian=True) @ grad,
                dir_weighted,
                None,
            )

        elif optimizer == "BFGS":

            def bfgs_true(_):
                prev_direction = jax.lax.cond(
                    i > 0,
                    lambda __: -prev_grad,
                    lambda __: -grad,
                    operand=None,
                )
                s_k = jnp.mean(eta) * prev_direction  # Use mean for BFGS
                y_k = grad - prev_grad
                rho_k = jnp.reciprocal(jnp.dot(y_k, s_k))

                Hy = hess @ y_k
                yHy = jnp.dot(y_k, Hy)
                term1 = rho_k * jnp.outer(s_k, Hy)
                term2 = rho_k * jnp.outer(Hy, s_k)
                term3 = rho_k * (rho_k * yHy + 1.0) * jnp.outer(s_k, s_k)

                new_hess = hess - term1 - term2 + term3
                new_hess = jnp.where(jnp.isfinite(rho_k), new_hess, hess)
                new_direction = -new_hess @ grad
                return new_hess, new_direction

            def bfgs_false(_):
                return hess, -grad

            hess, direction = jax.lax.cond(i > 1, bfgs_true, bfgs_false, operand=None)

        elif optimizer == "SGD":
            direction = -grad

        elif optimizer == "Adam":
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8

            m_adam = beta1 * m_adam + (1 - beta1) * grad
            v_adam = beta2 * v_adam + (1 - beta2) * (grad**2)
            m_hat = m_adam / (1 - beta1 ** (i + 1))
            v_hat = v_adam / (1 - beta2 ** (i + 1))
            direction = -m_hat / (jnp.sqrt(v_hat) + epsilon)
        else:
            raise ValueError(f"Optimizer '{optimizer}' not supported")

        if scale:
            direction = direction / jnp.linalg.norm(direction)

        if ls:

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
                )["neg_loglik"]

                return jnp.squeeze(neg_loglik)

            eta_scalar = _line_search(
                _obj_neg_loglik,
                curr_obj=loglik,
                pt=theta_ests,
                grad=grad,
                direction=direction,
                k=i + 1,
                eta=jnp.mean(eta),  # TODO: use a better solution
                xi=10,
                tau=max_ls_itn,
                c=c,
                frac=0.5,
                stoch=False,
            )
            theta_ests = theta_ests + eta_scalar * direction

        else:
            theta_ests = theta_ests + eta * direction

        prev_grad = grad
        prev_hess = hess

        new_carry = (
            theta_ests,
            key,
            hess,
            prev_grad,
            prev_hess,
            m_adam,
            v_adam,
        )

        return new_carry, (loglik, theta_ests)

    hess = jnp.eye(theta_ests.shape[-1])  # default one
    prev_grad = jnp.zeros_like(theta_ests)
    prev_hess = hess
    m_adam = jnp.zeros_like(theta_ests)
    v_adam = jnp.zeros_like(theta_ests)

    initial_carry = (
        theta_ests,
        key,
        hess,
        prev_grad,
        prev_hess,
        m_adam,
        v_adam,
    )

    _, (logliks_history, Acopies_history) = jax.lax.scan(
        scan_step,
        initial_carry,
        jnp.arange(M),
    )

    logliks = jnp.concatenate((jnp.array([jnp.nan]), logliks_history))
    Acopies = jnp.concatenate((theta_ests[jnp.newaxis, ...], Acopies_history))

    return logliks, Acopies


# Map over theta and key
_vmapped_train_internal = jax.vmap(
    _train_internal,
    in_axes=(0,) + (None,) * 20 + (0,) + (None,) * 2,
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
        lambda carry: carry[2], line_search_body, (eta, 0, False)
    )
    return eta_final


def _jgrad(
    theta_ests: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    ys_extended: jax.Array,
    ys_observed: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
    dmeasure: Callable,  # static
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    thresh: float,
    key: jax.Array,
    n_obs: int,
):
    """
    calculates the gradient of a mean particle filter objective (function
    'pfilter_internal_mean') w.r.t. the current estimated parameter value using
    JAX's automatic differentiation.

    Returns:
        array-like: the gradient of the pfilter_internal_mean function w.r.t.
            theta_ests.
    """
    return _grad_pfilter_internal_mean(
        theta_ests,  # for some reason this needs to be given as a positional argument
        dt_array_extended=dt_array_extended,
        nstep_array=nstep_array,
        t0=t0,
        times=times,
        ys_extended=ys_extended,
        ys_observed=ys_observed,
        J=J,
        rinitializer=rinitializer,
        rprocess=rprocess,
        dmeasure=dmeasure,
        accumvars=accumvars,
        covars_extended=covars_extended,
        thresh=thresh,
        key=key,
        n_obs=n_obs,
    )


def _jvg(
    theta_ests: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    ys_extended: jax.Array,
    ys_observed: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
    dmeasure: Callable,  # static
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    thresh: float,
    key: jax.Array,
    n_obs: int,
):
    """
    Calculates the both the value and gradient of a mean particle filter
    objective (function 'pfilter_internal_mean') w.r.t. the current estimated
    parameter value using JAX's automatic differentiation.

    Returns:
        tuple: A tuple containing:
        - The mean of negative log-likelihood value across the measurements
            using pfilter_internal_mean function.
        - The gradient of the function pfilter_internal_mean function w.r.t.
            theta_ests.
    """
    return _vg_pfilter_internal_mean(
        theta_ests,
        dt_array_extended=dt_array_extended,
        nstep_array=nstep_array,
        t0=t0,
        times=times,
        ys_extended=ys_extended,
        ys_observed=ys_observed,
        J=J,
        rinitializer=rinitializer,
        rprocess=rprocess,
        dmeasure=dmeasure,
        accumvars=accumvars,
        covars_extended=covars_extended,
        thresh=thresh,
        key=key,
        n_obs=n_obs,
    )


def _jgrad_mop(
    theta_ests: jax.Array,
    ys: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
    dmeasure: Callable,  # static
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    alpha: float,
    key: jax.Array,
):
    """
    Calculates the gradient of a mean MOP objective (function
    'mop_internal_mean') w.r.t. the current estimated parameter value using
    JAX's automatic differentiation.

    Returns:
        array-like: the gradient of the mop_internal_mean function w.r.t.
            theta_ests.
    """
    return _grad_mop_internal_mean(
        theta_ests,
        ys=ys,
        dt_array_extended=dt_array_extended,
        nstep_array=nstep_array,
        t0=t0,
        times=times,
        J=J,
        rinitializer=rinitializer,
        rprocess_interp=rprocess,
        dmeasure=dmeasure,
        accumvars=accumvars,
        covars_extended=covars_extended,
        alpha=alpha,
        key=key,
    )


def _jvg_mop(
    theta_ests: jax.Array,
    ys: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
    dmeasure: Callable,  # static
    covars_extended: jax.Array | None,
    accumvars: tuple[int, ...] | None,
    alpha: float,
    key: jax.Array,
) -> tuple:
    """
    Calculates the both the value and gradient of a mean MOP objective (function
    'mop_internal_mean') w.r.t. the current estimated parameter value using
    JAX's automatic differentiation.

    Returns:
        tuple: A tuple containing:
        - The mean of negative log-likelihood value across the measurements
            using mop_internal_mean function.
        - The gradient of the function mop_internal_mean function w.r.t.
            theta_ests.
    """
    return _vg_mop_internal_mean(
        theta_ests,
        ys=ys,
        dt_array_extended=dt_array_extended,
        nstep_array=nstep_array,
        t0=t0,
        times=times,
        J=J,
        rinitializer=rinitializer,
        rprocess_interp=rprocess,
        dmeasure=dmeasure,
        covars_extended=covars_extended,
        accumvars=accumvars,
        alpha=alpha,
        key=key,
    )


def _jhess(
    theta_ests: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    ys_extended: jax.Array,
    ys_observed: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
    dmeasure: Callable,  # static
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    thresh: float,
    key: jax.Array,
    n_obs: int,
):
    """
    calculates the Hessian matrix of a mean particle filter objective (function
    'pfilter_internal_mean') w.r.t. the current estimated parameter value using
    JAX's automatic differentiation.

    Returns:
        array-like: the Hessian matrix of the pfilter_internal_mean function
            w.r.t. theta_ests.
    """
    return _hess_pfilter_internal_mean(
        theta_ests,
        dt_array_extended=dt_array_extended,
        nstep_array=nstep_array,
        t0=t0,
        times=times,
        ys_extended=ys_extended,
        ys_observed=ys_observed,
        J=J,
        rinitializer=rinitializer,
        rprocess=rprocess,
        dmeasure=dmeasure,
        accumvars=accumvars,
        covars_extended=covars_extended,
        thresh=thresh,
        key=key,
        n_obs=n_obs,
    )


# get the hessian matrix from mop
def _jhess_mop(
    theta_ests: jax.Array,
    ys: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
    dmeasure: Callable,  # static
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    alpha: float,
    key: jax.Array,
):
    """
    calculates the Hessian matrix of a mean MOP objective (function
    'mop_internal_mean') w.r.t. the current estimated parameter value using
    JAX's automatic differentiation.

    Returns:
        array-like: the Hessian matrix of the mop_internal_mean function w.r.t.
            theta_ests.
    """
    return _hess_mop_internal_mean(
        theta_ests,
        ys=ys,
        dt_array_extended=dt_array_extended,
        nstep_array=nstep_array,
        t0=t0,
        times=times,
        J=J,
        rinitializer=rinitializer,
        rprocess_interp=rprocess,
        dmeasure=dmeasure,
        accumvars=accumvars,
        covars_extended=covars_extended,
        alpha=alpha,
        key=key,
    )
