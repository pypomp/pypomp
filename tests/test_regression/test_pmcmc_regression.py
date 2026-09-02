import jax
import numpy as np

import pypomp as pp
from pypomp.functional.pmcmc import pmcmc

M = 3


def test_pmcmc_regression(lg_struct, tol, num_regression):
    struct, theta0, key, J, n_reps, param_names = lg_struct
    keys = jax.random.split(key, n_reps)
    prop = pp.MVNDiagRW({name: 0.01 for name in param_names})

    ll_traces, _, theta_traces, accepts = pmcmc(
        struct, theta0, proposal=prop, J=J, M=M, thresh=0.0, keys=keys
    )

    num_regression.check(
        {
            "final_loglik": np.asarray(ll_traces[:, -1]).ravel(),
            "final_theta": np.asarray(theta_traces[:, -1, :]).ravel(),
            "accepts": np.asarray(accepts, dtype=float).ravel(),
        },
        default_tolerance=tol,
    )
