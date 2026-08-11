import jax
import jax.numpy as jnp
import numpy as np

import pypomp.functional as F


def test_panel_pfilter_regression(lg_panel_struct, tol, num_regression):
    struct, shared0, unit0, _, _, key, J, n_reps = lg_panel_struct
    U = unit0.shape[1]
    thetas_panel = jnp.stack(
        [jnp.concatenate([shared0, unit0[:, u, :]], axis=-1) for u in range(U)],
        axis=1,
    )
    keys = jax.random.split(key, n_reps * U).reshape(n_reps, U, *key.shape)

    results = F.panel_pfilter(
        struct, thetas_panel, J=J, thresh=0.0, keys=keys, chunk_size=1
    )

    num_regression.check(
        {"logLik": np.asarray(results["logLik"]).ravel()},
        default_tolerance=tol,
    )
