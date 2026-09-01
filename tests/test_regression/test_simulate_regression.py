import jax
import numpy as np

import pypomp.functional as F


def test_simulate_regression(lg_struct, tol, num_regression):
    struct, theta0, key, J, n_reps, _ = lg_struct
    keys = jax.random.split(key, n_reps)
    nsim = 1

    X_sims, Y_sims = F.simulate(struct, nsim, theta0, keys=keys)

    num_regression.check(
        {
            "X_final": np.asarray(X_sims[:, :, -1, :]).ravel(),
            "Y": np.asarray(Y_sims).ravel(),
        },
        default_tolerance=tol,
    )
