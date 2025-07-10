import pypomp.internal_functions as ifunc
import jax.numpy as jnp


def test_precompute_interp_covars():
    t0 = -1.0
    times = jnp.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    ctimes = jnp.array([0, 1, 2, 3, 4, 5])
    covars = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    order = "linear"
    dt = 0.5
    nstep = 1
    interp_covars = ifunc._precompute_interp_covars(
        t0, times, ctimes, covars, dt, None, order
    )
    interp_covars2 = ifunc._precompute_interp_covars(
        t0, times, ctimes, covars, None, nstep, order
    )
