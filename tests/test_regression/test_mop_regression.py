import jax
import numpy as np

import pypomp.functional as F


def test_mop_regression(lg_struct, tol, num_regression):
    struct, theta0, key, J, n_reps, _ = lg_struct
    keys = jax.random.split(key, n_reps)

    result = F.mop(struct, theta0, J, alpha=0.5, keys=keys)

    num_regression.check(
        {"mop": np.asarray(result).ravel()},
        default_tolerance=tol,
    )
