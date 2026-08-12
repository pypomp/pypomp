"""Locked full IF2 trace.

``test_mif_regression`` locks only the final iteration. Locking every iteration
identifies the iteration at which a change first takes effect, which separates a
genuine algorithm change from accumulated drift.
"""

import jax
import jax.numpy as jnp
import numpy as np

import pypomp as pp
import pypomp.functional as F

M = 4


def test_mif_full_trace_regression(lg_struct_multi, tol, num_regression):
    struct, thetas, key, J, n_reps, param_names = lg_struct_multi
    keys = jax.random.split(key, n_reps)
    rw_sd = pp.RWSigma({name: 0.02 for name in param_names}).geometric_cooling(0.5)
    thetas_mif = jnp.repeat(thetas[:, jnp.newaxis, :], J, axis=1)

    logliks_M, thetas_traces_Md, _ = F.mif(
        struct, thetas_mif, rw_sd, M=M, J=J, thresh=0.0, keys=keys, n_monitors=1
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


def test_mif_trace_starts_at_initial_theta(lg_struct_multi):
    """Invariant: iteration 0 of the trace is the starting parameter vector."""
    struct, thetas, key, J, n_reps, param_names = lg_struct_multi
    keys = jax.random.split(key, n_reps)
    rw_sd = pp.RWSigma({name: 0.02 for name in param_names}).geometric_cooling(0.5)
    thetas_mif = jnp.repeat(thetas[:, jnp.newaxis, :], J, axis=1)

    _, thetas_traces_Md, _ = F.mif(
        struct, thetas_mif, rw_sd, M=M, J=J, thresh=0.0, keys=keys, n_monitors=1
    )

    np.testing.assert_allclose(
        np.asarray(thetas_traces_Md)[:, 0, :], np.asarray(thetas), rtol=1e-6, atol=1e-6
    )
