import jax
import jax.numpy as jnp
import numpy as np

import pypomp as pp
import pypomp.functional as F
from tests.helpers.params import uniform_rw_sd

M = 3
# The full-trace baseline runs a longer chain so the cooling schedule has room
# to show up across iterations.
TRACE_M = 4


def test_mif_regression(lg_struct, tol, num_regression):
    struct, theta0, key, J, n_reps, param_names = lg_struct
    keys = jax.random.split(key, n_reps)
    rw_sd = uniform_rw_sd(param_names, cooling=0.5)
    thetas_mif = jnp.repeat(theta0[:, jnp.newaxis, :], J, axis=1)

    logliks_M, thetas_traces_Md, _ = F.mif(
        struct, thetas_mif, J, M, rw_sd, keys, thresh=0.0, n_monitors=0
    )

    num_regression.check(
        {
            "final_loglik": np.asarray(logliks_M[:, -1]).ravel(),
            "final_theta": np.asarray(thetas_traces_Md[:, -1, :]).ravel(),
        },
        default_tolerance=tol,
    )


def test_mif_full_trace_regression(lg_struct_multi, tol, num_regression):
    """Every iteration, not just the last.

    The baseline above locks only the final iteration, so it cannot separate a
    genuine algorithm change from accumulated drift. Locking each iteration
    identifies where the divergence starts.
    """
    struct, thetas, key, J, n_reps, param_names = lg_struct_multi
    keys = jax.random.split(key, n_reps)
    rw_sd = uniform_rw_sd(param_names, cooling=0.5)
    thetas_mif = jnp.repeat(thetas[:, jnp.newaxis, :], J, axis=1)

    logliks_M, thetas_traces_Md, _ = F.mif(
        struct,
        thetas_mif,
        J,
        TRACE_M,
        rw_sd,
        keys,
        thresh=0.0,
        n_monitors=1,
    )

    logliks = np.asarray(logliks_M)
    traces = np.asarray(thetas_traces_Md)

    data = {}
    for rep in range(n_reps):
        for m in range(logliks.shape[1]):
            data[f"loglik_rep{rep}_m{m}"] = np.array([logliks[rep, m]])
        for m in range(traces.shape[1]):
            for p, name in enumerate(param_names):
                data[f"{name}_rep{rep}_m{m}"] = np.array([traces[rep, m, p]])

    num_regression.check(data, default_tolerance=tol)


def test_sir_mif_regression(sir_struct, tol, num_regression):
    """Non-Gaussian counterpart: accumvars and a discrete measurement density."""
    struct, theta0, key, J, n_reps, param_names = sir_struct
    keys = jax.random.split(key, n_reps)
    rw_sd = pp.RWSigma({name: 0.01 for name in param_names}).geometric_cooling(0.5)
    thetas_mif = jnp.repeat(theta0[:, jnp.newaxis, :], J, axis=1)

    logliks_M, thetas_traces_Md, _ = F.mif(
        struct, thetas_mif, J, 2, rw_sd, keys, thresh=0.0, n_monitors=1
    )

    num_regression.check(
        {
            "logliks": np.asarray(logliks_M).ravel(),
            "final_theta": np.asarray(thetas_traces_Md)[:, -1, :].ravel(),
        },
        default_tolerance=tol,
    )
