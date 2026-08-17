import jax
import numpy as np

from pypomp.functional.dpop import dpop


def test_dpop_regression(lg_struct, tol, num_regression):
    struct, theta0, key, J, n_reps, _ = lg_struct
    keys = jax.random.split(key, n_reps)

    result = dpop(struct, theta0, J, alpha=0.5, process_weight_index=0, keys=keys)

    num_regression.check(
        {"dpop": np.asarray(result).ravel()},
        default_tolerance=tol,
    )
