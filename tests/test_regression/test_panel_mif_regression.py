import jax
import jax.numpy as jnp
import numpy as np

import pypomp.functional as F
from tests.helpers.params import uniform_rw_sd

M = 3


def test_panel_mif_regression(lg_panel_struct, tol, num_regression):
    struct, shared0, unit0, _, _, key, J, n_reps = lg_panel_struct
    shared_mif = jnp.repeat(shared0[:, jnp.newaxis, :], J, axis=1)
    unit_mif = jnp.repeat(unit0[:, jnp.newaxis, :, :], J, axis=1)
    all_param_names = list(struct.shared_param_names) + list(struct.unit_param_names)
    rw_sd = uniform_rw_sd(all_param_names, cooling=0.5)
    keys = jax.random.split(key, n_reps)

    shared_traces, unit_traces, _, _ = F.panel_mif(
        struct,
        shared_mif,
        unit_mif,
        rw_sd,
        M=M,
        J=J,
        thresh=0.0,
        keys=keys,
        n_monitors=0,
    )

    num_regression.check(
        {
            "final_shared": np.asarray(shared_traces[:, -1, :]).ravel(),
            "final_unit": np.asarray(unit_traces[:, -1, :, :]).ravel(),
        },
        default_tolerance=tol,
    )
