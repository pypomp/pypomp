import jax
import jax.numpy as jnp
import numpy as np

import pypomp as pp
import pypomp.functional as F

M = 3


def test_mif_regression(lg_struct, tol, num_regression):
    struct, theta0, key, J, n_reps, param_names = lg_struct
    keys = jax.random.split(key, n_reps)
    rw_sd = pp.RWSigma({name: 0.02 for name in param_names}).geometric_cooling(0.5)
    thetas_mif = jnp.repeat(theta0[:, jnp.newaxis, :], J, axis=1)

    logliks_M, thetas_traces_Md, _ = F.mif(
        struct, thetas_mif, rw_sd, M=M, J=J, thresh=0.0, keys=keys, n_monitors=0
    )

    num_regression.check(
        {
            "final_loglik": np.asarray(logliks_M[:, -1]).ravel(),
            "final_theta": np.asarray(thetas_traces_Md[:, -1, :]).ravel(),
        },
        default_tolerance=tol,
    )
