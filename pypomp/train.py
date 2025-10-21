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

# --- parameter transforms (minimal integration) ---
from .parameter_trans import ParTrans, _pt_inverse
_IDENTITY_PARTRANS = ParTrans(False, (), (), (), None, None)


@partial(
    jit,
    static_argnames=(
        "rinitializer",
        "rprocess_interp",
        "dmeasure",
        "J",
        "optimizer",
        "M",
        "eta",
        "c",
        "max_ls_itn",
        "thresh",
        "scale",
        "ls",
        "alpha",
        "n_monitors",
        "n_obs",
        "partrans",     # NEW: treat partrans as static (contains callables)
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
    eta: float,  # static
    c: float,  # static
    max_ls_itn: int,  # static
    thresh: float,  # static
    scale: bool,  # static
    ls: bool,  # static
    alpha: float,  # static
    key: jax.Array,
    n_monitors: int,  # static
    n_obs: int,  # static
    *, 
    partrans: ParTrans = _IDENTITY_PARTRANS, 
):
    """
    Internal function for conducting the MOP-based gradient optimization.
    All updates happen on the *estimation scale*; models always receive
    *natural-scale* parameters via `_pt_inverse`.
    """
    times = times.astype(float)
    ylen = ys.shape[0]
    if n_monitors < 1 and ls:
        raise ValueError("Line search requires at least one monitor")

    def train_step(i, carry):
        theta_ests, key, hess, Acopies, logliks, grads, hesses = carry

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
                partrans=partrans,   # NEW
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
                partrans=partrans,   # NEW
            )
            if n_monitors > 0:
                key, *subkeys = jax.random.split(key, n_monitors + 1)
                # Convert estimation-scale θ to natural-scale for PF monitors
                theta_nat_single = _pt_inverse(theta_ests, partrans)
                theta_nat_batch  = jnp.broadcast_to(theta_nat_single[None, :], (n_monitors, theta_nat_single.shape[0]))
                final_neg_ll = _vmapped_pfilter_internal(
                    theta_nat_batch,
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
                    jnp.stack(subkeys, axis=0),
                    False,
                    False,
                    False,
                    False,
                )["neg_loglik"]
                loglik = jnp.mean(final_neg_ll)
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
                partrans=partrans,   # NEW
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
                partrans=partrans,   # NEW
            )

            def dir_weighted(_):
                i_f = i.astype(theta_ests.dtype)
                wt = (i_f ** jnp.log(i_f)) / ((i_f + 1) ** jnp.log(i_f + 1))
                weighted_hess = wt * hesses[-1] + (1 - wt) * hess
                return -jnp.linalg.pinv(weighted_hess) @ grad

            direction = jax.lax.cond(
                i == 0, lambda _: -jnp.linalg.pinv(hess) @ grad, dir_weighted, None
            )

        elif optimizer == "BFGS":

            def bfgs_true(_):
                prev_direction = jax.lax.cond(
                    i > 0,
                    lambda __: -grads[i - 1],
                    lambda __: -grad,
                    operand=None,
                )
                s_k = eta * prev_direction
                y_k = grad - grads[i - 1]
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

        else:
            # For BFGS when i <= 1, or other optimizers, use gradient descent
            direction = -grad

        if scale:
            direction = direction / jnp.linalg.norm(direction)

        if ls:

            def _obj_neg_loglik(theta):
                # line-search objective should use natural-scale parameters
                theta_nat = _pt_inverse(theta, partrans)
                neg_loglik = _pfilter_internal(
                    theta_nat,
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

            eta2 = _line_search(
                _obj_neg_loglik,
                curr_obj=loglik,
                pt=theta_ests,
                grad=grad,
                direction=direction,
                k=i + 1,
                eta=jnp.array(eta),
                xi=10,
                tau=max_ls_itn,
                c=c,
                frac=0.5,
                stoch=False,
            )

        else:
            eta2 = eta

        theta_ests = theta_ests + eta2 * direction

        # Update carry state
        Acopies = Acopies.at[i].set(theta_ests)
        logliks = logliks.at[i].set(loglik)
        grads = grads.at[i].set(grad)
        hesses = hesses.at[i].set(hess)

        return (theta_ests, key, hess, Acopies, logliks, grads, hesses)

    # Initialize arrays for storing results
    Acopies = jnp.zeros((M + 1, *theta_ests.shape))
    logliks = jnp.zeros(M + 1)
    grads = jnp.zeros((M + 1, *theta_ests.shape))
    hesses = jnp.zeros((M + 1, theta_ests.shape[-1], theta_ests.shape[-1]))

    # Set initial values
    Acopies = Acopies.at[0].set(theta_ests)
    hess = jnp.eye(theta_ests.shape[-1])  # default one

    # Run the optimization loop
    final_theta, final_key, final_hess, Acopies, logliks, grads, hesses = (
        jax.lax.fori_loop(
            0,
            M,
            train_step,
            (theta_ests, key, hess, Acopies, logliks, grads, hesses),
        )
    )

    # Final evaluation (optional monitors via PF)
    if n_monitors > 0:
        final_key, *subkeys = jax.random.split(final_key, n_monitors + 1)
        theta_nat_final_single = _pt_inverse(final_theta, partrans)  # (n_params,)
        theta_nat_final_batch  = jnp.broadcast_to(
            theta_nat_final_single[None, :],
        (n_monitors, theta_nat_final_single.shape[0])
        )
        final_loglik = jnp.mean(
            _vmapped_pfilter_internal(
                theta_nat_final_batch,
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
                jnp.stack(subkeys, axis=0),
                False,
                False,
                False,
                False,
            )["neg_loglik"]
        )
    else:
        final_loglik = jnp.array(jnp.nan)

    # Update final results
    logliks = logliks.at[-1].set(final_loglik)
    Acopies = Acopies.at[-1].set(final_theta)

    return logliks, Acopies


_vmapped_train_internal = jax.vmap(
    _train_internal,
    in_axes=(0,) + (None,) * 20 + (0,) + (None,) * 3
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
    Quasi-Newton methods. The implementation follows
    https://arxiv.org/pdf/1909.01238.pdf.
    """
    eta = jnp.where(stoch, jnp.minimum(eta, xi / k), eta)

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


# -------------------------
# PF-based AD helpers (optional) with partrans
# -------------------------
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
    partrans: ParTrans = _IDENTITY_PARTRANS,  # NEW
):
    """
    Gradient of mean PF objective w.r.t. estimation-scale parameters.
    """
    def _wrapped(th):
        th_nat = _pt_inverse(th, partrans)
        return _pfilter_internal_mean(
            th_nat,
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
    return jax.grad(_wrapped)(theta_ests)


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
    partrans: ParTrans = _IDENTITY_PARTRANS,  # NEW
):
    """
    Value and gradient of mean PF objective w.r.t. estimation-scale parameters.
    """
    def _wrapped(th):
        th_nat = _pt_inverse(th, partrans)
        return _pfilter_internal_mean(
            th_nat,
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
    return jax.value_and_grad(_wrapped)(theta_ests)


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
    partrans: ParTrans = _IDENTITY_PARTRANS,  # NEW
):
    """
    Hessian of mean PF objective w.r.t. estimation-scale parameters.
    """
    def _wrapped(th):
        th_nat = _pt_inverse(th, partrans)
        return _pfilter_internal_mean(
            th_nat,
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
    return jax.hessian(_wrapped)(theta_ests)


# -------------------------
# MOP-based AD helpers with partrans
# -------------------------
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
    partrans: ParTrans = _IDENTITY_PARTRANS,  # NEW
):
    """
    Gradient of mean MOP objective w.r.t. estimation-scale parameters.
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
        partrans=partrans,  # NEW
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
    partrans: ParTrans = _IDENTITY_PARTRANS,  # NEW
) -> tuple:
    """
    Value and gradient of mean MOP objective w.r.t. estimation-scale parameters.
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
        partrans=partrans,  # NEW
    )


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
    partrans: ParTrans = _IDENTITY_PARTRANS,  # NEW
):
    """
    Hessian of mean MOP objective w.r.t. estimation-scale parameters.
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
        partrans=partrans,  # NEW
    )
