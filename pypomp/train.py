from functools import partial
import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from typing import Callable
from .pfilter import (
    _pfilter_internal,
    _vmapped_pfilter_internal,
    _pfilter_internal_mean,
)
from .mop import _mop_internal_mean


def _train_internal(
    theta_ests: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    rinitializer: Callable,
    rprocess: Callable,
    dmeasure: Callable,
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    J: int,
    optimizer: str,
    itns: int,
    eta: float,
    c: float,
    max_ls_itn: int,
    thresh: float,
    verbose: bool,
    scale: bool,
    ls: bool,
    alpha: float,
    key: jax.Array,
    n_monitors: int,
):
    """
    Internal function for conducting the MOP gradient estimate method.
    """
    eta2 = eta
    Acopies = []
    grads = []
    hesses = []
    logliks = []
    hess = jnp.eye(theta_ests.shape[-1])  # default one

    for i in tqdm(range(itns)):
        if n_monitors == 1:
            key, subkey = jax.random.split(key)
            loglik, grad = _jvg_mop(
                theta_ests=theta_ests,
                t0=t0,
                times=times,
                ys=ys,
                J=J,
                rinitializer=rinitializer,
                rprocess=rprocess,
                dmeasure=dmeasure,
                ctimes=ctimes,
                covars=covars,
                alpha=alpha,
                key=subkey,
            )

            loglik *= len(ys)
        else:
            key, subkey = jax.random.split(key)
            grad = _jgrad_mop(
                theta_ests=theta_ests,
                t0=t0,
                times=times,
                ys=ys,
                J=J,
                rinitializer=rinitializer,
                rprocess=rprocess,
                dmeasure=dmeasure,
                ctimes=ctimes,
                covars=covars,
                alpha=alpha,
                key=subkey,
            )

            key, *subkeys = jax.random.split(key, n_monitors + 1)
            loglik = jnp.mean(
                _vmapped_pfilter_internal(
                    theta_ests,
                    t0,
                    times,
                    ys,
                    J,
                    rinitializer,
                    rprocess,
                    dmeasure,
                    ctimes,
                    covars,
                    0,
                    jnp.array(subkeys),
                )
            )

        if optimizer == "Newton":
            key, subkey = jax.random.split(key)
            hess = _jhess_mop(
                theta_ests=theta_ests,
                t0=t0,
                times=times,
                ys=ys,
                J=J,
                rinitializer=rinitializer,
                rprocess=rprocess,
                dmeasure=dmeasure,
                ctimes=ctimes,
                covars=covars,
                alpha=alpha,
                key=subkey,
            )

            # flatten
            # theta_flat = theta_ests.flatten()
            # grad_flat = grad.flatten()
            # hess_flat = hess.reshape(theta_flat.size, theta_flat.size)
            # hess_flat_pinv = np.linalg.pinv(hess_flat)
            # direction_flat = -hess_flat_pinv @ grad_flat
            # direction = direction_flat.reshape(theta_ests.shape)

            direction = -jnp.linalg.pinv(hess) @ grad
        elif optimizer == "WeightedNewton":
            if i == 0:
                key, subkey = jax.random.split(key)
                hess = _jhess_mop(
                    theta_ests=theta_ests,
                    t0=t0,
                    times=times,
                    ys=ys,
                    J=J,
                    rinitializer=rinitializer,
                    rprocess=rprocess,
                    dmeasure=dmeasure,
                    ctimes=ctimes,
                    covars=covars,
                    alpha=alpha,
                    key=subkey,
                )
                # theta_flat = theta_ests.flatten()
                # grad_flat = grad.flatten()
                # hess_flat = hess.reshape(theta_flat.size, theta_flat.size)
                # hess_flat_pinv = np.linalg.pinv(hess_flat)
                # direction_flat = -hess_flat_pinv @ grad_flat
                # direction = direction_flat.reshape(theta_ests.shape)
                direction = -jnp.linalg.pinv(hess) @ grad

            else:
                key, subkey = jax.random.split(key)
                hess = _jhess_mop(
                    theta_ests=theta_ests,
                    t0=t0,
                    times=times,
                    ys=ys,
                    J=J,
                    rinitializer=rinitializer,
                    rprocess=rprocess,
                    dmeasure=dmeasure,
                    ctimes=ctimes,
                    covars=covars,
                    alpha=alpha,
                    key=subkey,
                )
                wt = (i ** np.log(i)) / ((i + 1) ** (np.log(i + 1)))
                # theta_flat = theta_ests.flatten()
                # grad_flat = grad.flatten()
                weighted_hess = wt * hesses[-1] + (1 - wt) * hess
                # weighted_hess_flat = weighted_hess.reshape(theta_flat.size, theta_flat.size)
                # weighted_hess_flat_pinv = np.linalg.pinv(weighted_hess_flat)
                # direction_flat = -weighted_hess_flat_pinv @ grad_flat
                # direction = direction_flat.reshape(theta_ests.shape)
                direction = -jnp.linalg.pinv(weighted_hess) @ grad

        elif optimizer == "BFGS" and i > 1:
            s_k = eta * direction
            # not grad but grads
            y_k = grad - grads[-1]
            rho_k = jnp.reciprocal(jnp.dot(y_k, s_k))
            sy_k = s_k[:, jnp.newaxis] * y_k[jnp.newaxis, :]
            w = jnp.eye(theta_ests.shape[-1], dtype=rho_k.dtype) - rho_k * sy_k
            # H_(k+1) = W_k^T@H_k@W_k + pho_k@s_k@s_k^T
            hess = (
                jnp.einsum("ij,jk,lk", w, hess, w)
                + rho_k * s_k[:, jnp.newaxis] * s_k[jnp.newaxis, :]
            )
            hess = jnp.where(jnp.isfinite(rho_k), hess, hess)

            # theta_flat = theta_ests.flatten()
            # grad_flat = grad.flatten()
            # hess_flat = hess.reshape(theta_flat.size, theta_flat.size)

            # direction_flat = -hess_flat @ grad_flat
            # direction = direction_flat.reshape(theta_ests.shape)

            direction = -hess @ grad

        else:
            direction = -grad

        Acopies.append(theta_ests)
        logliks.append(loglik)
        grads.append(grad)
        hesses.append(hess)

        if scale:
            direction = direction / jnp.linalg.norm(direction)

        if ls:
            eta2 = _line_search(
                partial(
                    _pfilter_internal,
                    t0=t0,
                    times=times,
                    ys=ys,
                    J=J,
                    rinitializer=rinitializer,
                    rprocess=rprocess,
                    dmeasure=dmeasure,
                    ctimes=ctimes,
                    covars=covars,
                    thresh=thresh,
                    key=subkey,
                ),
                curr_obj=float(loglik),
                pt=theta_ests,
                grad=grad,
                direction=direction,
                k=i + 1,
                eta=eta,
                xi=10,
                tau=max_ls_itn,
                c=c,
                frac=0.5,
                stoch=False,
            )

        if verbose:
            print(theta_ests, eta2, logliks[i])

        theta_ests += eta2 * direction

    key, *subkeys = jax.random.split(key, n_monitors + 1)
    logliks.append(
        jnp.mean(
            _vmapped_pfilter_internal(
                theta_ests,
                t0,
                times,
                ys,
                J,
                rinitializer,
                rprocess,
                dmeasure,
                ctimes,
                covars,
                0,
                jnp.array(subkeys),
            )
        )
    )
    Acopies.append(theta_ests)

    return jnp.array(logliks), jnp.array(Acopies)


def _line_search(
    obj: Callable,
    curr_obj: float,
    pt: jax.Array,
    grad: jax.Array,
    direction: jax.Array,
    k: int,
    eta: float,
    xi: int,
    tau: int,
    c: float,
    frac: float,
    stoch: bool,
) -> float:
    """
    Conducts line search algorithm to determine the step size under stochastic
    Quasi-Newton methods. The implentation of the algorithm refers to
    https://arxiv.org/pdf/1909.01238.pdf.

    Args:
        obj (function): The objective function aiming to minimize
        curr_obj (float): The value of the objective function at the current
            point.
        pt (array-like): The array containing current parameter values.
        grad (array-like): The gradient of the objective function at the current
            point.
        direction (array-like): The direction to update the parameters.
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
        float: optimal step size
    """
    itn = 0
    eta = min([eta, xi / k]) if stoch else eta
    next_obj = obj(pt + eta * direction)
    # check whether the new point(new_obj)satisfies the stochastic Armijo condition
    # if not, repeat until the condition is met
    # previous: grad.T @ direction
    while next_obj > curr_obj + eta * c * jnp.sum(grad * direction) or jnp.isnan(
        next_obj
    ):
        eta *= frac
        itn += 1
        if itn > tau:
            break
    return eta


@partial(jit, static_argnums=(4, 5, 6, 7))
def _jgrad(
    theta_ests: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
    dmeasure: Callable,  # static
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    thresh: float,
    key: jax.Array,
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
        t0=t0,
        times=times,
        ys=ys,
        J=J,
        rinitializer=rinitializer,
        rprocess=rprocess,
        dmeasure=dmeasure,
        ctimes=ctimes,
        covars=covars,
        thresh=thresh,
        key=key,
    )


@partial(jit, static_argnums=(4, 5, 6, 7))
def _jvg(
    theta_ests: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
    dmeasure: Callable,  # static
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    thresh: float,
    key: jax.Array,
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
        t0=t0,
        times=times,
        ys=ys,
        J=J,
        rinitializer=rinitializer,
        rprocess=rprocess,
        dmeasure=dmeasure,
        ctimes=ctimes,
        covars=covars,
        thresh=thresh,
        key=key,
    )


@partial(jit, static_argnums=(4, 5, 6, 7))
def _jgrad_mop(
    theta_ests: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
    dmeasure: Callable,  # static
    ctimes: jax.Array | None,
    covars: jax.Array | None,
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
        t0=t0,
        times=times,
        ys=ys,
        J=J,
        rinitializer=rinitializer,
        rprocess=rprocess,
        dmeasure=dmeasure,
        ctimes=ctimes,
        covars=covars,
        alpha=alpha,
        key=key,
    )


@partial(jit, static_argnums=(4, 5, 6, 7))
def _jvg_mop(
    theta_ests: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
    dmeasure: Callable,  # static
    ctimes: jax.Array | None,
    covars: jax.Array | None,
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
        t0=t0,
        times=times,
        ys=ys,
        J=J,
        rinitializer=rinitializer,
        rprocess=rprocess,
        dmeasure=dmeasure,
        ctimes=ctimes,
        covars=covars,
        alpha=alpha,
        key=key,
    )


@partial(jit, static_argnums=(4, 5, 6, 7))
def _jhess(
    theta_ests: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
    dmeasure: Callable,  # static
    ctimes: jax.Array | None,
    covars: jax.Array | None,
    thresh: float,
    key: jax.Array,
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
        t0=t0,
        times=times,
        ys=ys,
        J=J,
        rinitializer=rinitializer,
        rprocess=rprocess,
        dmeasure=dmeasure,
        ctimes=ctimes,
        covars=covars,
        thresh=thresh,
        key=key,
    )


# get the hessian matrix from mop
@partial(jit, static_argnums=(4, 5, 6, 7))
def _jhess_mop(
    theta_ests: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    J: int,  # static
    rinitializer: Callable,  # static
    rprocess: Callable,  # static
    dmeasure: Callable,  # static
    ctimes: jax.Array | None,
    covars: jax.Array | None,
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
        t0=t0,
        times=times,
        ys=ys,
        J=J,
        rinitializer=rinitializer,
        rprocess=rprocess,
        dmeasure=dmeasure,
        ctimes=ctimes,
        covars=covars,
        alpha=alpha,
        key=key,
    )
