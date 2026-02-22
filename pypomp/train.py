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
from .mop import _mop_internal_mean
from .dpop import _dpop_internal_mean  # DPOP mean negative log-likelihood per observation



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

    def train_step(i, carry):
        (
            theta_ests,
            key,
            hess,
            Acopies,
            logliks,
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
            direction = -jnp.linalg.pinv(hess) @ grad

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
                return -jnp.linalg.pinv(weighted_hess) @ grad

            direction = jax.lax.cond(
                i == 0, lambda _: -jnp.linalg.pinv(hess) @ grad, dir_weighted, None
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
                sy_k = s_k[:, jnp.newaxis] * y_k[jnp.newaxis, :]
                w = jnp.eye(theta_ests.shape[-1], dtype=rho_k.dtype) - rho_k * sy_k
                new_hess = (
                    jnp.einsum("ij,jk,lk", w, hess, w)
                    + rho_k * s_k[:, jnp.newaxis] * s_k[jnp.newaxis, :]
                )
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

        # Update carry state
        Acopies = Acopies.at[i + 1].set(theta_ests)
        logliks = logliks.at[i + 1].set(loglik)
        prev_grad = grad
        prev_hess = hess

        return (
            theta_ests,
            key,
            hess,
            Acopies,
            logliks,
            prev_grad,
            prev_hess,
            m_adam,
            v_adam,
        )

    # Initialize arrays for storing results
    Acopies = jnp.full((M + 1, *theta_ests.shape), jnp.nan)
    logliks = jnp.full(M + 1, jnp.nan)

    # Set initial values
    Acopies = Acopies.at[0].set(theta_ests)
    hess = jnp.eye(theta_ests.shape[-1])  # default one
    prev_grad = jnp.zeros_like(theta_ests)
    prev_hess = hess

    # Initialize Adam state (momentum and variance estimates)
    m_adam = jnp.zeros_like(theta_ests)
    v_adam = jnp.zeros_like(theta_ests)

    # Run the optimization loop
    (
        final_theta,
        final_key,
        final_hess,
        Acopies,
        logliks,
        prev_grad,
        prev_hess,
        final_m_adam,
        final_v_adam,
    ) = jax.lax.fori_loop(
        0,
        M,
        train_step,
        (theta_ests, key, hess, Acopies, logliks, prev_grad, prev_hess, m_adam, v_adam),
    )

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
    return jax.grad(_pfilter_internal_mean)(
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

    Args:
        theta_ests (array-like): Estimated parameter.
        ys (array-like): The measurements.
        J (int): Number of particles.
        rinit (function): Simulator for the initial-state distribution.
        rprocess (function): Simulator for the process model.
        dmeasure (function): Density evaluation for the measurement model.
        covars (array-like): Covariates or None if not applicable.
        thresh (float): Threshold value to determine whether to resample
            particles.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        tuple: A tuple containing:
        - The mean of negative log-likelihood value across the measurements
            using pfilter_internal_mean function.
        - The gradient of the function pfilter_internal_mean function w.r.t.
            theta_ests.
    """
    return jax.value_and_grad(_pfilter_internal_mean)(
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
    return jax.grad(_mop_internal_mean)(
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
    return jax.value_and_grad(_mop_internal_mean)(
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
    return jax.hessian(_pfilter_internal_mean)(
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
    return jax.hessian(_mop_internal_mean)(
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

# ----------------------------------------------------------------------
# DPOP gradient helpers and a simple SGD-with-decay optimizer
# ----------------------------------------------------------------------

def _jgrad_dpop(
    theta_ests: jax.Array,
    ys: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    J: int,  # static conceptually (number of particles)
    rinitializer: Callable,  # static conceptually
    rprocess: Callable,      # static conceptually
    dmeasure: Callable,      # static conceptually
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    alpha: float,
    process_weight_index: int | None,
    ntimes: int,  # static - number of observation times
    key: jax.Array,
) -> jax.Array:
    """
    Gradient of the DPOP mean negative log-likelihood with respect to theta_ests.

    This wraps `_dpop_internal_mean` with `jax.grad`. The objective is the mean
    negative log-likelihood per observation, so the gradient is scaled
    accordingly (which is fine for optimization).
    """
    return jax.grad(_dpop_internal_mean)(
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
        process_weight_index=process_weight_index,
        ntimes=ntimes,
        key=key,
    )


def _jvg_dpop(
    theta_ests: jax.Array,
    ys: jax.Array,
    dt_array_extended: jax.Array,
    nstep_array: jax.Array,
    t0: float,
    times: jax.Array,
    J: int,  # static conceptually (number of particles)
    rinitializer: Callable,  # static conceptually
    rprocess: Callable,      # static conceptually
    dmeasure: Callable,      # static conceptually
    accumvars: tuple[int, ...] | None,
    covars_extended: jax.Array | None,
    alpha: float,
    process_weight_index: int | None,
    ntimes: int,  # static - number of observation times
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Value and gradient of the DPOP mean negative log-likelihood.

    Returns
    -------
    value : scalar jax.Array
        Mean negative log-likelihood per observation under DPOP.
    grad : jax.Array, same shape as theta_ests
        Gradient of the objective with respect to theta_ests.
    """
    return jax.value_and_grad(_dpop_internal_mean)(
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
        process_weight_index=process_weight_index,
        ntimes=ntimes,
        key=key,
    )



