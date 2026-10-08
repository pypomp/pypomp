import jax
import numpy as np

import pypomp.functional as F
from pypomp.models.sir import get_process_weight_index


def test_dpop_regression(sir_struct, tol, num_regression):
    struct, theta0, key, J, n_reps, param_names = sir_struct
    keys = jax.random.split(key, n_reps)
    theta_est = struct.par_trans._transform_array(
        theta0, param_names, direction="to_est"
    )

    # The value equals MOP's for the same keys, so the gradient is what
    # locks the DPOP-specific process-score term.
    value, grad = jax.value_and_grad(
        lambda th: F.mop(struct, th, J, 0.5, keys, get_process_weight_index()).sum()
    )(theta_est)

    num_regression.check(
        {"dpop": np.asarray(value).ravel(), "grad": np.asarray(grad).ravel()},
        default_tolerance=tol,
    )
