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
