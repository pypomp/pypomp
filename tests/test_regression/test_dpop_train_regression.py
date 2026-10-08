import jax
import numpy as np

import pypomp as pp
from pypomp.functional.dpop import dpop_train
from pypomp.models.sir import get_process_weight_index

M = 3


def test_dpop_train_regression(sir_struct, tol, num_regression):
    struct, theta0, key, J, n_reps, param_names = sir_struct
    keys = jax.random.split(key, n_reps)
    eta = pp.LearningRate({name: 0.01 for name in param_names})

    neg_logliks, theta_traces = dpop_train(
        struct,
        theta0,
        J,
        M,
        eta,
        keys,
        get_process_weight_index(),
        optimizer=pp.Adam(),
        alpha=0.8,
    )

    num_regression.check(
        {
            "final_neg_loglik": np.asarray(neg_logliks[:, -1]).ravel(),
            "final_theta": np.asarray(theta_traces[:, -1, :]).ravel(),
        },
        default_tolerance=tol,
    )
