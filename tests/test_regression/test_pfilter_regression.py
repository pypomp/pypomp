import jax
import numpy as np

import pypomp.functional as F


def test_pfilter_regression(lg_struct, tol, num_regression):
    struct, theta0, key, J, n_reps, _ = lg_struct
    keys = jax.random.split(key, n_reps).reshape(n_reps, 1)

    results = F.pfilter(struct, theta0, J, thresh=0.0, keys=keys)

    num_regression.check(
        {"logLik": np.asarray(results["logLik"]).ravel()},
        default_tolerance=tol,
    )


def test_pfilter_diagnostics_regression(lg_struct_multi, tol, num_regression):
    """Per-observation diagnostics, over two replicates.

    The scalar baseline above reports that something changed but not where.
    Locking each observation time identifies the first step at which a change
    takes effect.
    """
    struct, thetas, key, J, n_reps, _ = lg_struct_multi
    keys = jax.random.split(key, n_reps).reshape(n_reps, 1)

    results = F.pfilter(
        struct,
        thetas,
        J,
        thresh=0.0,
        keys=keys,
        CLL=True,
        ESS=True,
        filter_mean=True,
        prediction_mean=True,
    )

    # One column per (replicate, timestep) so a mismatch names the exact step.
    data = {"logLik": np.asarray(results["logLik"]).ravel()}
    for name in ("CLL", "ESS", "filter_mean", "prediction_mean"):
        flat = np.asarray(results[name]).reshape(n_reps, -1)
        for rep in range(n_reps):
            for step in range(flat.shape[1]):
                data[f"{name}_rep{rep}_t{step}"] = np.array([flat[rep, step]])

    num_regression.check(data, default_tolerance=tol)


def test_sir_pfilter_regression(sir_struct, tol, num_regression):
    """Non-Gaussian counterpart: accumvars and a discrete measurement density."""
    struct, theta0, key, J, n_reps, _ = sir_struct
    keys = jax.random.split(key, n_reps).reshape(n_reps, 1)

    results = F.pfilter(struct, theta0, J, thresh=0.0, keys=keys, CLL=True, ESS=True)

    data = {"logLik": np.asarray(results["logLik"]).ravel()}
    for name in ("CLL", "ESS"):
        for step, value in enumerate(np.asarray(results[name]).reshape(-1)):
            data[f"{name}_t{step}"] = np.array([value])

    num_regression.check(data, default_tolerance=tol)
