import jax
import numpy as np

import pypomp as pp
import pypomp.functional as F

M = 3


def test_panel_train_regression(lg_panel_struct, tol, num_regression):
    struct, shared0, unit0, _, _, key, J, n_reps = lg_panel_struct
    U = unit0.shape[1]
    all_param_names = list(struct.shared_param_names) + list(struct.unit_param_names)
    eta = pp.LearningRate({name: 0.01 for name in all_param_names})
    keys = jax.random.split(key, n_reps * M * U).reshape(n_reps, M, U)

    neg_logliks, shared_history, unit_history = F.panel_train(
        struct,
        shared0,
        unit0,
        J=J,
        optimizer=pp.Adam(),
        M=M,
        eta=eta,
        alpha=0.97,
        keys=keys,
        alpha_cooling=1.0,
        chunk_size=1,
    )

    num_regression.check(
        {
            "final_neg_loglik": np.asarray(neg_logliks[:, -1]).ravel(),
            "final_shared": np.asarray(shared_history[:, -1, :]).ravel(),
            "final_unit": np.asarray(unit_history[:, -1, :, :]).ravel(),
        },
        default_tolerance=tol,
    )
