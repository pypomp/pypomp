# tests/test_mop_partrans_integration.py
import jax
import jax.numpy as jnp

from pypomp.mop import _mop_internal_mean
from pypomp.parameter_trans import ParTrans, parameter_trans, _pt_forward

def riniter(theta_nat, keys, covars0, t0):
    J = keys.shape[0]
    return jnp.zeros((J, 1))

def rproc_interp(particlesF, theta_nat, keys, covars_extended, dt_array_extended, t, t_idx, nstep, accumvars):
    return particlesF, t_idx

def dmeas(y, particlesP, theta_nat, covars_t, t):
    J = particlesP.shape[0]
    s = jnp.sum(theta_nat)
    return jnp.full((J,), -0.5 * (s - 1.23) ** 2)

def _grids(T=4):
    times = jnp.arange(T, dtype=float)
    dt_array_extended = jnp.zeros((1,))
    nstep_array = jnp.ones((T,), dtype=jnp.int32)
    ys = jnp.zeros((T,))
    return dt_array_extended, nstep_array, 0.0, times, ys

def test_mop_estimation_vs_natural_equivalence_under_partrans():
    key = jax.random.key(0)
    J = 16
    alpha = 0.9
    dt_ext, nstep_arr, t0, times, ys = _grids(T=4)

    theta_nat = jnp.array([0.8, 1.1, 1.5, -0.2])
    pt = parameter_trans(custom=[0,1,2,3], to_est=lambda x: 2 * x + 1, from_est=lambda z: 0.5 * (z - 1))
    theta_est = _pt_forward(theta_nat, pt)

    val_est = _mop_internal_mean(
        theta=theta_est,
        ys=ys,
        dt_array_extended=dt_ext,
        nstep_array=nstep_arr,
        t0=t0,
        times=times,
        J=J,
        rinitializer=riniter,
        rprocess_interp=rproc_interp,
        dmeasure=dmeas,
        accumvars=None,
        covars_extended=None,
        alpha=alpha,
        key=key,
        partrans=pt,
    )

    val_nat = _mop_internal_mean(
        theta=theta_nat,
        ys=ys,
        dt_array_extended=dt_ext,
        nstep_array=nstep_arr,
        t0=t0,
        times=times,
        J=J,
        rinitializer=riniter,
        rprocess_interp=rproc_interp,
        dmeasure=dmeas,
        accumvars=None,
        covars_extended=None,
        alpha=alpha,
        key=key,
        partrans=ParTrans(),
    )

    assert jnp.allclose(val_est, val_nat, atol=1e-8)
