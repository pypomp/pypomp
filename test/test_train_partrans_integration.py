# tests/test_train_partrans_integration.py
import jax
import jax.numpy as jnp

from pypomp.train import _train_internal
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

def _grids(T=3):
    times = jnp.arange(T, dtype=float)
    dt_array_extended = jnp.zeros((1,))
    nstep_array = jnp.ones((T,), dtype=jnp.int32)
    ys = jnp.zeros((T,))
    return dt_array_extended, nstep_array, 0.0, times, ys

def test_train_M0_monitor_identity_vs_custom_same_final_loglik():
    key = jax.random.key(0)
    d = 3
    theta_nat = jnp.arange(1.0, d + 1)
    pt = parameter_trans(custom=list(range(d)), to_est=lambda x: 2 * x + 1, from_est=lambda z: 0.5 * (z - 1))
    theta_est = _pt_forward(theta_nat, pt)

    dt_ext, nstep_arr, t0, times, ys = _grids(T=3)

    common = dict(
        ys=ys,
        dt_array_extended=dt_ext,
        nstep_array=nstep_arr,
        t0=t0,
        times=times,
        rinitializer=riniter,
        rprocess_interp=rproc_interp,
        dmeasure=dmeas,
        accumvars=None,
        covars_extended=None,
        J=6,
        optimizer="SGD",
        M=0,               # no parameter update; just final monitor
        eta=0.1,
        c=1e-4,
        max_ls_itn=5,
        thresh=1e12,
        scale=False,
        ls=False,
        alpha=0.9,
        key=key,
        n_monitors=1,      # enable final PF monitor
        n_obs=len(ys),
    )

    logliks_pt, traces_pt = _train_internal(theta_ests=theta_est, partrans=pt, **common)
    logliks_id, traces_id = _train_internal(theta_ests=theta_nat, partrans=ParTrans(), **common)

    assert jnp.allclose(logliks_pt[-1], logliks_id[-1], atol=1e-8)

def test_train_line_search_requires_monitor():
    key = jax.random.key(1)
    dt_ext, nstep_arr, t0, times, ys = _grids(T=2)
    theta = jnp.array([1.0, 2.0])

    with jax.disable_jit():  # avoid hiding Python-side ValueError inside JIT
        try:
            _ = _train_internal(
                theta_ests=theta,
                ys=ys,
                dt_array_extended=dt_ext,
                nstep_array=nstep_arr,
                t0=t0,
                times=times,
                rinitializer=riniter,
                rprocess_interp=rproc_interp,
                dmeasure=dmeas,
                accumvars=None,
                covars_extended=None,
                J=4,
                optimizer="SGD",
                M=1,
                eta=0.1,
                c=1e-4,
                max_ls_itn=5,
                thresh=10.0,
                scale=False,
                ls=True,           # line search on
                alpha=0.9,
                key=key,
                n_monitors=0,      # invalid with ls=True
                n_obs=len(ys),
                partrans=ParTrans(),
            )
            assert False, "Expected ValueError when ls=True and n_monitors<1"
        except ValueError as e:
            assert "Line search requires at least one monitor" in str(e)
