import jax
import numpy as np

import pypomp as pp
import pypomp.functional as F

M = 3


def test_dpop_train_regression(sir_struct, tol, num_regression):
    struct, theta0, key, J, n_reps, param_names = sir_struct
    keys = jax.random.split(key, n_reps)
    eta = pp.LearningRate({name: 0.01 for name in param_names})

    neg_logliks, theta_traces = F.train(
        struct,
        theta0,
        J,
        M,
        eta,
        keys,
        optimizer=pp.Adam(),
        alpha=0.8,
        dpop=True,
    )

    num_regression.check(
        {
            "final_neg_loglik": np.asarray(neg_logliks[:, -1]).ravel(),
            "final_theta": np.asarray(theta_traces[:, -1, :]).ravel(),
        },
        default_tolerance=tol,
    )
