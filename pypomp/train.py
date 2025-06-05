from functools import partial
import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
import xarray as xr
from tqdm import tqdm
from typing import Optional
from .pfilter import _pfilter_internal
from .pfilter import _pfilter_internal_mean
from .mop import _mop_internal_mean

MONITORS = 1  # TODO: figure out what this is for and remove it if possible


def train(
    rinit,
    rproc,
    dmeas,
    ys,
    theta,
    J,
    Jh,
    key,
    covars=None,
    method="Newton",
    itns=20,
    beta=0.9,
    eta=0.0025,
    c=0.1,
    max_ls_itn=10,
    thresh=0,
    verbose=False,
    scale=False,
    ls=False,
    alpha=0.97,
):
    """
    This function runs the MOP gradient-based iterative optimization method.

    Parameters
    ----------
    rinit : RInit
        Simulator for the initial-state distribution.
    rproc : RProc
        Simulator for the process model.
    dmeas : DMeas
        Density evaluation for the measurement model.
    ys : array-like
        The measurement array.
    theta : dict
        Initial parameters for the POMP model. Each value should be a float.
    J : int
        The number of particles for the MOP gradient-based iterative
        optimization method.
    Jh : int
        The number of particles for the Hessian evaluation.
    key : jax.random.PRNGKey
        The random key for reproducibility.
    covars : array-like or None
        Covariates or None if not applicable.
    method : str
        The gradient-based iterative optimization method to use, including
        Newton method, weighted Newton method, BFGS method, gradient descent.
    itns : int
        Maximum iteration for the gradient descent optimization.
    beta : float
        Initial step size for the line search algorithm.
    eta : float
        Initial step size.
    c : float
        The user-defined Armijo condition constant.
    max_ls_itn : int
        The maximum number of iterations for the line search algorithm.
    thresh : int
        Threshold value to determine whether to resample particles in pfilter
        function.
    verbose : bool
        Boolean flag controlling whether to print out the log-likelihood and
        parameter information.
    scale : bool
        Boolean flag controlling whether to scale the parameters.
    ls : bool
        Boolean flag controlling whether to use the line search algorithm.
    alpha : float
        The forgetting factor for the MOP algorithm.

    Returns
    -------
    dict: a dictionary containing:
        - xarray of log-likelihood values through iterations.
        - xarray of parameters through iterations.
    """
    if J < 1:
        raise ValueError("J should be greater than 0")
    if Jh < 1:
        raise ValueError("Jh should be greater than 0")

    if not isinstance(theta, dict):
        raise TypeError("theta must be a dictionary")
    if not all(isinstance(val, float) for val in theta.values()):
        raise TypeError("Each value of theta must be a float")

    nLLs, theta_ests = _train_internal(
        theta_ests=jnp.array(list(theta.values())),
        t0=rinit.t0,
        times=jnp.array(ys.index),
        ys=jnp.array(ys),
        rinitializer=rinit.struct_pf,
        rprocess=rproc.struct_pf,
        dmeasure=dmeas.struct_pf,
        J=J,
        Jh=Jh,
        ctimes=jnp.array(covars.index) if covars is not None else None,
        covars=jnp.array(covars) if covars is not None else None,
        method=method,
        itns=itns,
        beta=beta,
        eta=eta,
        c=c,
        max_ls_itn=max_ls_itn,
        thresh=thresh,
        verbose=verbose,
        scale=scale,
        ls=ls,
        alpha=alpha,
        key=key,
    )
    return {
        "logLik": xr.DataArray(-nLLs, dims=["iteration"]),
        "thetas": xr.DataArray(
            theta_ests,
            dims=["iteration", "theta"],
            coords={
                "iteration": range(0, itns + 1),
                "theta": list(theta.keys()),
            },
        ),
    }


def _train_internal(
    theta_ests: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    rinitializer: callable,
    rprocess: callable,
    dmeasure: callable,
    ctimes: Optional[jax.Array],
    covars: Optional[jax.Array],
    J: int,
    Jh: int,
    method: str,
    itns: int,
    beta: float,
    eta: float,
    c: float,
    max_ls_itn: int,
    thresh: float,
    verbose: bool,
    scale: bool,
    ls: bool,
    alpha: float,
    key: jax.Array,
):
    """
    Internal function for conducting the MOP gradient estimate method.
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
                key=key,
            )

            loglik *= len(ys)
        else:
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
                key=key,
            )
            loglik = jnp.mean(
                jnp.array(
                    [
                        _pfilter_internal(
                            theta=theta_ests,
                            t0=t0,
                            times=times,
                            ys=ys,
                            J=J,
                            rinitializer=rinitializer,
                            rprocess=rprocess,
                            dmeasure=dmeasure,
                            ctimes=ctimes,
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
                ),
                curr_obj=loglik,
                pt=theta_ests,
                grad=grad,
                direction=direction,
                k=i + 1,
                eta=beta,
                xi=10,
                tau=max_ls_itn,
                c=c,
                frac=0.5,
                stoch=False,
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
                        theta=theta_ests,
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
                    for i in range(MONITORS)
                ]
            )
        )
    )
    Acopies.append(theta_ests)

    return jnp.array(logliks), jnp.array(Acopies)


def _line_search(
    obj: callable,
    curr_obj: float,
    pt: jnp.ndarray,
    grad: jnp.ndarray,
    direction: jnp.ndarray,
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
        frac (float, optional): The fact.
        stoch (bool, optional): Boolean argument controlling whether to adjust
            the initial step size.

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


@partial(jit, static_argnums=(4, 5, 6, 7))
def _jgrad(
    theta_ests: jax.Array,
    t0: float,
    times: jax.Array,
    ys: jax.Array,
    J: int,  # static
    rinitializer: callable,  # static
    rprocess: callable,  # static
    dmeasure: callable,  # static
    ctimes: Optional[jax.Array],
    covars: Optional[jax.Array],
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
    rinitializer: callable,  # static
    rprocess: callable,  # static
    dmeasure: callable,  # static
    ctimes: Optional[jax.Array],
    covars: Optional[jax.Array],
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
    rinitializer: callable,  # static
    rprocess: callable,  # static
    dmeasure: callable,  # static
    ctimes: Optional[jax.Array],
    covars: Optional[jax.Array],
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
    rinitializer: callable,  # static
    rprocess: callable,  # static
    dmeasure: callable,  # static
    ctimes: Optional[jax.Array],
    covars: Optional[jax.Array],
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
    rinitializer: callable,  # static
    rprocess: callable,  # static
    dmeasure: callable,  # static
    ctimes: Optional[jax.Array],
    covars: Optional[jax.Array],
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
    rinitializer: callable,  # static
    rprocess: callable,  # static
    dmeasure: callable,  # static
    ctimes: Optional[jax.Array],
    covars: Optional[jax.Array],
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
