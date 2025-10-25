# tests/test_mif_partrans_integration.py
import jax
import jax.numpy as jnp

from pypomp.mif import _mif_internal
from pypomp.parameter_trans import ParTrans, parameter_trans, _pt_forward, _pt_inverse

# ---- Minimal dummy model for stable, fast tests ----
def riniter(theta_nat, keys, covars0, t0):
    J = theta_nat.shape[0]
    return jnp.zeros((J, 1))  # 1D state, unused by likelihood

def rproc_interp(particlesF, thetas, keys, covars_extended, dt_array_extended, t, t_idx, nstep, accumvars):
    return particlesF, t_idx  # no dynamics

def dmeas(y, particlesP, thetas_nat_per_particle, covars_t, t):
    s = jnp.sum(thetas_nat_per_particle, axis=-1)  # (J,)
    return -0.5 * (s - 1.23) ** 2  # per-particle log-likelihood

def _grids(T=3):
    times = jnp.arange(T, dtype=float)
    dt_array_extended = jnp.zeros((1,))
    nstep_array = jnp.ones((T,), dtype=jnp.int32)
    ys = jnp.zeros((T,))
    return dt_array_extended, nstep_array, 0.0, times, ys

def test_mif_partrans_zero_noise_identity_vs_custom_same_result():
    key = jax.random.key(0)
    J, d, M = 4, 3, 2
    a = 0.5
    thresh = 1e12  # avoid resampling
    sigmas = 0.0
    sigmas_init = 0.0

    dt_ext, nstep_arr, t0, times, ys = _grids(T=3)
    theta_nat = jnp.tile(jnp.arange(1.0, d + 1), (J, 1))

    pt_id = ParTrans()
    nll_id, trace_id = _mif_internal(
        theta=theta_nat,
        dt_array_extended=dt_ext,
        nstep_array=nstep_arr,
        t0=t0,
        times=times,
        ys=ys,
        rinitializers=riniter,
        rprocesses_interp=rproc_interp,
        dmeasures=dmeas,
        sigmas=sigmas,
        sigmas_init=sigmas_init,
        accumvars=None,
        covars_extended=None,
        partrans=pt_id,
        M=M,
        a=a,
        J=J,
        thresh=thresh,
        key=key,
    )

    # Custom linear transform z = 2x + 1, x = 0.5(z - 1)
    pt = parameter_trans(custom=[0, 1, 2], to_est=lambda x: 2 * x + 1, from_est=lambda z: 0.5 * (z - 1))
    nll_pt, trace_pt = _mif_internal(
        theta=theta_nat,
        dt_array_extended=dt_ext,
        nstep_array=nstep_arr,
        t0=t0,
        times=times,
        ys=ys,
        rinitializers=riniter,
        rprocesses_interp=rproc_interp,
        dmeasures=dmeas,
        sigmas=sigmas,
        sigmas_init=sigmas_init,
        accumvars=None,
        covars_extended=None,
        partrans=pt,
        M=M,
        a=a,
        J=J,
        thresh=thresh,
        key=key,
    )

    assert jnp.allclose(nll_id, nll_pt, atol=1e-8)
    assert jnp.allclose(trace_id, trace_pt, atol=1e-8)

def test_mif_log_and_logit_indices_enforce_constraints_with_noise():
    # Check that perturbation done on estimation scale enforces positivity/(0,1) after inverse
    key = jax.random.key(1)
    J, M, a = 8, 1, 0.5
    thresh = 1e12
    sigmas = 0.2
    sigmas_init = 0.2
    dt_ext, nstep_arr, t0, times, ys = _grids(T=2)

    # theta columns: [log, free, logit]
    theta_nat = jnp.stack(
        [
            jnp.full((J,), 1.0),    # positive
            jnp.full((J,), 0.0),    # unbounded
            jnp.full((J,), 0.5),    # in (0,1)
        ],
        axis=-1,
    )

    pt = parameter_trans(log=[0], logit=[2])
    nll, trace = _mif_internal(
        theta=theta_nat,
        dt_array_extended=dt_ext,
        nstep_array=nstep_arr,
        t0=t0,
        times=times,
        ys=ys,
        rinitializers=riniter,
        rprocesses_interp=rproc_interp,
        dmeasures=dmeas,
        sigmas=sigmas,
        sigmas_init=sigmas_init,
        accumvars=None,
        covars_extended=None,
        partrans=pt,
        M=M,
        a=a,
        J=J,
        thresh=thresh,
        key=key,
    )

    # trace shape: (M+1, J, d); take the last (after M iteration)
    theta_last = trace[-1]
    assert jnp.all(theta_last[:, 0] > 0.0)      # log idx stays positive
    assert jnp.all((theta_last[:, 2] > 0.0) & (theta_last[:, 2] < 1.0))  # logit idx stays in (0,1)
