import jax
import numpy as np

import pypomp as pp
from pypomp.functional.dpop import dpop_train

M = 3


def test_dpop_train_regression(lg_struct, tol, num_regression):
    struct, theta0, key, J, n_reps, param_names = lg_struct
    keys = jax.random.split(key, n_reps)
    eta = pp.LearningRate({name: 0.01 for name in param_names})

    neg_logliks, theta_traces = dpop_train(
        struct,
        theta0,
        J,
        optimizer=pp.Adam(),
        M=M,
        eta=eta,
        alpha=0.8,
        process_weight_index=0,
        keys=keys,
    )

    num_regression.check(
        {
            "final_neg_loglik": np.asarray(neg_logliks[:, -1]).ravel(),
            "final_theta": np.asarray(theta_traces[:, -1, :]).ravel(),
        },
        default_tolerance=tol,
    )
