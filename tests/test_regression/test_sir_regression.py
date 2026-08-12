"""Locked baselines on a non-Gaussian model.

The LG fixture exercises no accumulator reset, no discrete measurement density,
and no non-trivial parameter transform. SIR covers those paths, so a change to
accumvar handling or to the measurement density is caught here rather than
slipping through a linear-Gaussian-only baseline.
"""

import jax
import jax.numpy as jnp
import numpy as np

import pypomp as pp
import pypomp.functional as F


def test_sir_pfilter_regression(sir_struct, tol, num_regression):
    struct, theta0, key, J, n_reps, _ = sir_struct
    keys = jax.random.split(key, n_reps).reshape(n_reps, 1)

    results = F.pfilter(struct, theta0, J, thresh=0.0, keys=keys, CLL=True, ESS=True)

    data = {"logLik": np.asarray(results["logLik"]).ravel()}
    for name in ("CLL", "ESS"):
        flat = np.asarray(results[name]).reshape(-1)
        for step, value in enumerate(flat):
            data[f"{name}_t{step}"] = np.array([value])

    num_regression.check(data, default_tolerance=tol)


def test_sir_mif_regression(sir_struct, tol, num_regression):
    struct, theta0, key, J, n_reps, param_names = sir_struct
    keys = jax.random.split(key, n_reps)
    rw_sd = pp.RWSigma({name: 0.01 for name in param_names}).geometric_cooling(0.5)
    thetas_mif = jnp.repeat(theta0[:, jnp.newaxis, :], J, axis=1)

    logliks_M, thetas_traces_Md, _ = F.mif(
        struct, thetas_mif, rw_sd, M=2, J=J, thresh=0.0, keys=keys, n_monitors=1
    )

    num_regression.check(
        {
            "logliks": np.asarray(logliks_M).ravel(),
            "final_theta": np.asarray(thetas_traces_Md)[:, -1, :].ravel(),
        },
        default_tolerance=tol,
    )


def test_sir_simulate_state_invariants(sir_struct):
    """Invariants a broken rproc or accumvar reset would violate."""
    struct, theta0, key, J, n_reps, _ = sir_struct
    keys = jax.random.split(key, n_reps)

    X_sims, Y_sims = F.simulate(struct, theta0, 3, keys, times=struct.times)

    obs = np.asarray(Y_sims)
    assert np.all(obs >= 0), "simulated SIR case counts must be non-negative"
    assert np.all(np.isfinite(obs))

    # S, I, R, cases are counts; logw and W are unconstrained, so check only
    # the compartments and the accumulator.
    states = np.asarray(X_sims)
    counts = states[..., :4]
    assert np.all(counts >= 0), "SIR compartments and case accumulator must be >= 0"
    assert np.all(np.isfinite(states))

    # accumvars are recorded as indices on the struct; the reset only makes
    # sense if those indices actually point into the state vector.
    assert struct.accumvars is not None, "SIR model should declare accumvars"
    assert all(0 <= i < states.shape[-1] for i in struct.accumvars)
