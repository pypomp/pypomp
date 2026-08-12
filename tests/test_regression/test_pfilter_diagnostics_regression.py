"""Locked per-timestep particle filter diagnostics.

``test_pfilter_regression`` locks a single scalar log-likelihood, which reports
that something changed but not where. Locking the per-observation diagnostics
instead identifies the first timestep at which a change takes effect.
"""

import jax
import numpy as np

import pypomp.functional as F


def test_pfilter_diagnostics_regression(lg_struct_multi, tol, num_regression):
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
        arr = np.asarray(results[name])
        flat = arr.reshape(n_reps, -1)
        for rep in range(n_reps):
            for step in range(flat.shape[1]):
                data[f"{name}_rep{rep}_t{step}"] = np.array([flat[rep, step]])

    num_regression.check(data, default_tolerance=tol)
