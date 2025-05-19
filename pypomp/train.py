from functools import partial
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from .internal_functions import _jgrad_mop
from .internal_functions import _jhess_mop
from .internal_functions import _pfilter_internal
from .internal_functions import _line_search
from .internal_functions import _jvg_mop

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
