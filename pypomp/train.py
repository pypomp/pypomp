from functools import partial
import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from .pfilter import _pfilter_internal
from .pfilter import _pfilter_internal_mean
from .mop import _mop_internal_mean

MONITORS = 1  # TODO: figure out what this is for and remove it if possible

# TODO: add external train function


def _train_internal(
    theta_ests,
    ys,
    rinit,
    rprocess,
    dmeasure,
    covars=None,
    J=5000,
    Jh=1000,
    method="GD",
    itns=20,
    beta=0.9,
    eta=0.0025,
    c=0.1,
    max_ls_itn=10,
    thresh=100,
    verbose=False,
    scale=False,
    ls=False,
    alpha=1,
    key=None,
):
    """
    Internal function for conducting the MOP gradient estimate method, is called
     in 'fit_internal' function.

    Args:
        theta_ests (array-like): Initial value of parameter values before SGD.
        ys (array-like): The measurement array.
        rinit (function): Simulator for the initial-state distribution.
        rprocess (function): Simulator for the process model.
        dmeasure (function): Density evaluation for the measurement model.
        covars (array-like, optional): Covariates or None if not applicable.
            Defaults to None. Defaults to None.
        J (int, optional): The number of particles in the MOP objective for
            obtaining the gradient. Defaults to 5000.
        Jh (int, optional): The number of particles in the MOP objective for
            obtaining the Hessian matrix. Defaults to 1000.
        method (str, optional): The optimization method to use, including
            Newton's method, weighted Newton's, BFGS, and gradient descent.
            Defaults to gradient descent.
        itns (int, optional): Maximum iteration for the gradient descent
            optimization. Defaults to 20.
        beta (float, optional): Initial step size for the line search
            algorithm. Defaults to 0.9.
        eta (float, optional): Initial step size. Defaults to 0.0025.
        c (float, optional): The user-defined Armijo condition constant.
            Defaults to 0.1.
        max_ls_itn (int, optional): The maximum number of iterations for the
            line search algorithm. Defaults to 10.
        thresh (int, optional): Threshold value to determine whether to resample
            particles in pfilter function. Defaults to 100.
        verbose (bool, optional): Boolean flag controlling whether to print out
            the log-likelihood and parameter information. Defaults to False.
        scale (bool, optional): Boolean flag controlling normalizing the
            direction or not. Defaults to False.
        ls (bool, optional): Boolean flag controlling whether to use the line
            search or not. Defaults to False.
        alpha (int, optional): Discount factor. Defaults to 1.

    Returns:
        tuple: A tuple containing:
        - An array of negative log-likelihood through the iterations
        - An array of parameters through the iterations
    """
    Acopies = []
    grads = []
    hesses = []
    logliks = []
    hess = jnp.eye(theta_ests.shape[-1])  # default one

    for i in tqdm(range(itns)):
        # key = jax.random.PRNGKey(np.random.choice(int(1e18)))
        if MONITORS == 1:
            loglik, grad = _jvg_mop(
                theta_ests,
                ys,
                J,
                rinit,
                rprocess,
                dmeasure,
                covars=covars,
                alpha=alpha,
                key=key,
            )

            loglik *= len(ys)
        else:
            grad = _jgrad_mop(
                theta_ests,
                ys,
                J,
                rinit,
                rprocess,
                dmeasure,
                covars=covars,
                alpha=alpha,
                key=key,
            )
            loglik = jnp.mean(
                jnp.array(
                    [
                        _pfilter_internal(
                            theta_ests,
                            ys,
                            J,
                            rinit,
                            rprocess,
                            dmeasure,
                            covars=covars,
                            thresh=-1,
                            key=key,
                        )
                        for i in range(MONITORS)
                    ]
                )
            )

        if method == "Newton":
            hess = _jhess_mop(
                theta_ests,
                ys,
                Jh,
                rinit,
                rprocess,
                dmeasure,
                covars=covars,
                alpha=alpha,
                key=key,
            )

            # flatten
            # theta_flat = theta_ests.flatten()
            # grad_flat = grad.flatten()
            # hess_flat = hess.reshape(theta_flat.size, theta_flat.size)
            # hess_flat_pinv = np.linalg.pinv(hess_flat)
            # direction_flat = -hess_flat_pinv @ grad_flat
            # direction = direction_flat.reshape(theta_ests.shape)

            direction = -jnp.linalg.pinv(hess) @ grad
        elif method == "WeightedNewton":
            if i == 0:
                hess = _jhess_mop(
                    theta_ests,
                    ys,
                    Jh,
                    rinit,
                    rprocess,
                    dmeasure,
                    covars=covars,
                    alpha=alpha,
                    key=key,
                )
                # theta_flat = theta_ests.flatten()
                # grad_flat = grad.flatten()
                # hess_flat = hess.reshape(theta_flat.size, theta_flat.size)
                # hess_flat_pinv = np.linalg.pinv(hess_flat)
                # direction_flat = -hess_flat_pinv @ grad_flat
                # direction = direction_flat.reshape(theta_ests.shape)
                direction = -jnp.linalg.pinv(hess) @ grad

            else:
                hess = _jhess_mop(
                    theta_ests,
                    ys,
                    Jh,
                    rinit,
                    rprocess,
                    dmeasure,
                    covars=covars,
                    alpha=alpha,
                    key=key,
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

        elif method == "BFGS" and i > 1:
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
            eta = _line_search(
                partial(
                    _pfilter_internal,
                    ys=ys,
                    J=J,
                    rinit=rinit,
                    rprocess=rprocess,
                    dmeasure=dmeasure,
                    covars=covars,
                    thresh=thresh,
                    key=key,
                ),
                loglik,
                theta_ests,
                grad,
                direction,
                k=i + 1,
                eta=beta,
                c=c,
                tau=max_ls_itn,
            )

        # try:
        #     et = eta if len(eta) == 1 else eta[i] # Not entirely sure why this is needed.
        # except Exception:
        #     et = eta
        # if i % 1 == 0 and verbose: # Does the lefthand side not always evaluate to True?
        if verbose:
            print(theta_ests, eta, logliks[i])

        theta_ests += eta * direction

    logliks.append(
        jnp.mean(
            jnp.array(
                [
                    _pfilter_internal(
                        theta_ests,
                        ys,
                        J,
                        rinit,
                        rprocess,
                        dmeasure,
                        covars=covars,
                        thresh=thresh,
                        key=key,
                    )
                    for i in range(MONITORS)
                ]
            )
        )
    )
    Acopies.append(theta_ests)

    return jnp.array(logliks), jnp.array(Acopies)


def _line_search(
    obj,
    curr_obj,
    pt,
    grad,
    direction,
    k=1,
    eta=0.9,
    xi=10,
    tau=10,
    c=0.1,
    frac=0.5,
    stoch=False,
):
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
        k (int, optional): Iteration index. Defaults to 1.
        eta (float, optional): Initial step size. Defaults to 0.9.
        xi (int, optional): Reduction limit. Defaults to 10.
        tau (int, optional): The maximum number of iterations. Defaults to 10.
        c (float, optional): The user-defined Armijo condition constant.
            Defaults to 0.1.
        frac (float, optional): The fact. Defaults to 0.5.
        stoch (bool, optional): Boolean argument controlling whether to adjust
            the initial step size. Defaults to False.

    Returns:
        float: optimal step size
    """
    itn = 0
    eta = min([eta, xi / k]) if stoch else eta  # if stoch is false, do not change
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


@partial(jit, static_argnums=(2, 3, 4, 5))
def _jgrad(theta_ests, ys, J, rinit, rprocess, dmeasure, covars, thresh, key=None):
    """
    calculates the gradient of a mean particle filter objective (function
    'pfilter_internal_mean') w.r.t. the current estimated parameter value using
    JAX's automatic differentiation.

    Args:
        theta_ests (array-like): Estimated parameter
        ys (array-like): The measurements
        J (int): Number of particles
        rinit (function): Simulator for the initial-state distribution
        rprocess (function): Simulator for the process model
        dmeasure (function): Density evaluation for the measurement model
        covars (array-like): Covariates or None if not applicable
        thresh (float): Threshold value to determine whether to resample
            particles.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        array-like: the gradient of the pfilter_internal_mean function w.r.t.
            theta_ests.
    """
    return jax.grad(_pfilter_internal_mean)(
        theta_ests,
        ys,
        J,
        rinit,
        rprocess,
        dmeasure,
        covars=covars,
        thresh=thresh,
        key=key,
    )


@partial(jit, static_argnums=(2, 3, 4, 5))
def _jvg(theta_ests, ys, J, rinit, rprocess, dmeasure, covars, thresh, key=None):
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
        ys,
        J,
        rinit,
        rprocess,
        dmeasure,
        covars=covars,
        thresh=thresh,
        key=key,
    )


@partial(jit, static_argnums=(2, 3, 4, 5))
def _jgrad_mop(
    theta_ests, ys, J, rinit, rprocess, dmeasure, covars, alpha=0.97, key=None
):
    """
    Calculates the gradient of a mean MOP objective (function
    'mop_internal_mean') w.r.t. the current estimated parameter value using
    JAX's automatic differentiation.

    Args:
        theta_ests (array-like): Estimated parameter.
        ys (array-like): The measurements.
        J (int): Number of particles.
        rinit (function): Simulator for the initial-state distribution.
        rprocess (function): Simulator for the process model.
        dmeasure (function): Density evaluation for the measurement model.
        covars (array-like): Covariates or None if not applicable.
        alpha (float, optional): Discount factor. Defaults to 0.97.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        array-like: the gradient of the mop_internal_mean function w.r.t.
            theta_ests.
    """
    return jax.grad(_mop_internal_mean)(
        theta_ests,
        ys,
        J,
        rinit,
        rprocess,
        dmeasure,
        covars=covars,
        alpha=alpha,
        key=key,
    )


@partial(jit, static_argnums=(2, 3, 4, 5))
def _jvg_mop(
    theta_ests, ys, J, rinit, rprocess, dmeasure, covars, alpha=0.97, key=None
):
    """
    calculates the both the value and gradient of a mean MOP objective (function
    'mop_internal_mean') w.r.t. the current estimated parameter value using
    JAX's automatic differentiation.

    Args:
        theta_ests (array-like): Estimated parameter.
        ys (array-like): The measurements.
        J (int): Number of particles.
        rinit (function): Simulator for the initial-state distribution.
        rprocess (function): Simulator for the process model.
        dmeasure (function): Density evaluation for the measurement model.
        covars (array-like): Covariates or None if not applicable.
        alpha (float, optional): Discount factor. Defaults to 0.97.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        tuple: A tuple containing:
        - The mean of negative log-likelihood value across the measurements
            using mop_internal_mean function.
        - The gradient of the function mop_internal_mean function w.r.t.
            theta_ests.
    """
    return jax.value_and_grad(_mop_internal_mean)(
        theta_ests,
        ys,
        J,
        rinit,
        rprocess,
        dmeasure,
        covars=covars,
        alpha=alpha,
        key=key,
    )


@partial(jit, static_argnums=(2, 3, 4, 5))
def _jhess(theta_ests, ys, J, rinit, rprocess, dmeasure, covars, thresh, key=None):
    """
    calculates the Hessian matrix of a mean particle filter objective (function
    'pfilter_internal_mean') w.r.t. the current estimated parameter value using
    JAX's automatic differentiation.

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
        array-like: the Hessian matrix of the pfilter_internal_mean function
            w.r.t. theta_ests.
    """
    return jax.hessian(_pfilter_internal_mean)(
        theta_ests,
        ys,
        J,
        rinit,
        rprocess,
        dmeasure,
        covars=covars,
        thresh=thresh,
        key=key,
    )


# get the hessian matrix from mop
@partial(jit, static_argnums=(2, 3, 4, 5))
def _jhess_mop(theta_ests, ys, J, rinit, rprocess, dmeasure, covars, alpha, key=None):
    """
    calculates the Hessian matrix of a mean MOP objective (function
    'mop_internal_mean') w.r.t. the current estimated parameter value using
    JAX's automatic differentiation.

    Args:
        theta_ests (array-like): Estimated parameter.
        ys (array-like): The measurements.
        J (int): Number of particles.
        rinit (function): Simulator for the initial-state distribution.
        rprocess (function): Simulator for the process model.
        dmeasure (function): Density evaluation for the measurement model.
        covars (array-like): Covariates or None if not applicable.
        alpha (float): Discount factor.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Returns:
        array-like: the Hessian matrix of the mop_internal_mean function w.r.t.
            theta_ests.
    """
    return jax.hessian(_mop_internal_mean)(
        theta_ests,
        ys,
        J,
        rinit,
        rprocess,
        dmeasure,
        covars=covars,
        alpha=alpha,
        key=key,
    )
