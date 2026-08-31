import jax
import jax.numpy as jnp
import numpy as np

import pypomp as pp
from pypomp.functional.abc import abc

M = 3


def test_abc_regression(lg_struct, tol, num_regression):
    struct, theta0, key, J, n_reps, param_names = lg_struct
    keys = jax.random.split(key, n_reps)
    prop = pp.MVNDiagRW({name: 0.01 for name in param_names})
    probes = {
        "mean": lambda y: jnp.mean(y["Y1"]),
        "std": lambda y: jnp.std(y["Y1"]),
    }
    scale = {"mean": 10.0, "std": 10.0}

    dist_traces, _, theta_traces, accepts = abc(
        struct,
        theta0,
        proposal=prop,
        probes=probes,
        scale=scale,
        epsilon=1e6,
        M=M,
        keys=keys,
    )

    num_regression.check(
        {
            "final_dist": np.asarray(dist_traces[:, -1]).ravel(),
            "final_theta": np.asarray(theta_traces[:, -1, :]).ravel(),
            "accepts": np.asarray(accepts, dtype=float).ravel(),
        },
        default_tolerance=tol,
    )
