import jax
import numpy as np

import pypomp as pp
import pypomp.functional as F

M = 3


def test_train_regression(lg_struct, tol, num_regression):
    struct, theta0, key, J, n_reps, param_names = lg_struct
    keys = jax.random.split(key, n_reps)
    eta = pp.LearningRate({name: 0.01 for name in param_names})

    neg_logliks, theta_traces = F.train(
        struct,
        theta0,
        J,
        optimizer=pp.Adam(scale=False, ls=False, c=0.0, max_ls_itn=1),
        M=M,
        eta=eta,
        thresh=0.0,
        alpha=0.0,
        keys=keys,
        alpha_cooling=1.0,
        n_monitors=1,
    )

    num_regression.check(
        {
            "final_neg_loglik": np.asarray(neg_logliks[:, -1]).ravel(),
            "final_theta": np.asarray(theta_traces[:, -1, :]).ravel(),
        },
        default_tolerance=tol,
    )
