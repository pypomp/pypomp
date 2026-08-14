from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pypomp as pp

J_DEFAULT = 2
M_DEFAULT = 2


@pytest.fixture(scope="module")
def simple_sir_for_dpop():
    """
    Build a small SIR Pomp model for testing the DPOP optimizers.
    """
    # Mirror the shrinkage in test_pomp_dpop.py::simple_sir to keep setup fast.
    model = pp.models.sir(delta_t=0.1, times=np.array([0.2, 0.4]))
    return model


@pytest.mark.parametrize(
    "optimizer, eta_type",
    [
        (pp.Adam(), "constant"),
        (pp.SGD(), "constant"),
        (pp.SGD(), "hyperbolic"),
    ],
)
def test_dpop_train_variants(simple_sir_for_dpop, optimizer, eta_type):
    """
    Test dpop_train with various optimizer configurations.
    """
    model = simple_sir_for_dpop
    if eta_type == "constant":
        eta = pp.LearningRate({name: 0.01 for name in model.canonical_param_names})
    else:
        eta = pp.LearningRate(
            {name: 0.01 for name in model.canonical_param_names}
        ).hyperbolic_decay(0.1, M=M_DEFAULT)

    model.results_history.clear()
    ret = model.dpop_train(
        J=J_DEFAULT,
        M=M_DEFAULT,
        eta=eta,
        optimizer=optimizer,
        alpha=0.8,
        process_weight_state="logw",
        key=jax.random.key(1),
    )
    assert ret is None
    res = model.results_history[-1]
    assert res.method == "dpop_train"
    assert res.kind == "trace"
    traces = res.traces()
    assert not traces.empty
    assert "logLik" in traces.columns


def test_dpop_train_param_order_invariance(simple_sir_for_dpop):
    """
    Check that dpop_train is invariant to the ordering of
    parameter dictionary keys (in natural space).
    """
    model = simple_sir_for_dpop

    J = J_DEFAULT
    M = M_DEFAULT
    eta = pp.LearningRate({name: 0.01 for name in model.canonical_param_names})
    initial_theta = deepcopy(model.theta)

    # First run: default theta ordering
    key1 = jax.random.key(123)
    model.results_history.clear()
    model.dpop_train(
        J=J,
        M=M,
        eta=eta,
        optimizer=pp.SGD(),
        alpha=0.8,
        key=key1,
        theta=deepcopy(initial_theta),
        process_weight_state="logw",
    )
    res1 = model.results_history[-1]

    # Build a permuted theta with reversed key order
    theta_orig = initial_theta.params(as_list=True)  # list[dict]
    param_keys = list(theta_orig[0].keys())
    rev_keys = list(reversed(param_keys))
    permuted_theta = [{k: th[k] for k in rev_keys} for th in theta_orig]

    # Second run: same random key & hyper-parameters, but permuted theta
    key2 = jax.random.key(123)
    model.dpop_train(
        J=J,
        M=M,
        eta=eta,
        optimizer=pp.SGD(),
        alpha=0.8,
        key=key2,
        theta=pp.PompParameters(permuted_theta),
        process_weight_state="logw",
    )
    res2 = model.results_history[-1]

    # Histories should match exactly up to numerical precision
    np.testing.assert_allclose(
        res1.traces()["logLik"], res2.traces()["logLik"], atol=1e-7
    )


def test_dpop_train_alpha_cooling_one_matches_default(simple_sir_for_dpop):
    """alpha_cooling=1.0 should preserve fixed-alpha DPOP behavior."""
    model = simple_sir_for_dpop
    eta = pp.LearningRate({name: 0.01 for name in model.canonical_param_names})
    initial_theta = deepcopy(model.theta)
    kwargs = dict(
        J=J_DEFAULT,
        M=M_DEFAULT,
        eta=eta,
        optimizer=pp.SGD(),
        alpha=0.8,
        process_weight_state="logw",
        key=jax.random.key(321),
    )

    model.results_history.clear()
    model.dpop_train(theta=deepcopy(initial_theta), **kwargs)
    res_default = model.results_history[-1]

    model.dpop_train(theta=deepcopy(initial_theta), alpha_cooling=1.0, **kwargs)
    res_fixed = model.results_history[-1]

    np.testing.assert_allclose(
        res_default.traces()["logLik"], res_fixed.traces()["logLik"], atol=1e-7
    )


def test_jgrad_and_jvg_dpop(simple_sir_for_dpop):
    """Directly test _jgrad_dpop and _jvg_dpop to ensure they are covered."""
    from pypomp.core.algorithms.train_dpop import _jgrad_dpop, _jvg_dpop

    model = simple_sir_for_dpop
    theta_dict = model.theta[0]
    theta_est_dict = model.par_trans.to_est(
        {k: jnp.array(v) for k, v in theta_dict.items()}
    )
    theta_ests = jnp.array([theta_est_dict[p] for p in model.canonical_param_names])

    ys = jnp.array(model.ys)
    dt_array_extended = model._dt_array_extended
    nstep_array = model._nstep_array
    t0 = model.t0
    times = jnp.array(model.ys.index)
    J = J_DEFAULT
    rinitializer = model.rinit.mechanics_pf
    rprocess = model.rproc.mechanics_pf_interp
    dmeasure = model.dmeas.mechanics_pf
    accumvars = model.rproc.accumvars
    covars_extended = (
        jnp.array(model._covars_extended)
        if model._covars_extended is not None
        else None
    )
    alpha = 0.8
    process_weight_index = int(model.statenames.index("logw"))
    ntimes = len(times)
    key = jax.random.key(123)

    grad = _jgrad_dpop(
        theta_ests=theta_ests,
        ys=ys,
        dt_array_extended=dt_array_extended,
        nstep_array=nstep_array,
        t0=t0,
        times=times,
        J=J,
        rinitializer=rinitializer,
        rprocess=rprocess,
        dmeasure=dmeasure,
        accumvars=accumvars,
        covars_extended=covars_extended,
        alpha=alpha,
        process_weight_index=process_weight_index,
        ntimes=ntimes,
        key=key,
    )
    assert grad.shape == theta_ests.shape
    assert jnp.all(jnp.isfinite(grad))

    val, grad_v = _jvg_dpop(
        theta_ests=theta_ests,
        ys=ys,
        dt_array_extended=dt_array_extended,
        nstep_array=nstep_array,
        t0=t0,
        times=times,
        J=J,
        rinitializer=rinitializer,
        rprocess=rprocess,
        dmeasure=dmeasure,
        accumvars=accumvars,
        covars_extended=covars_extended,
        alpha=alpha,
        process_weight_index=process_weight_index,
        ntimes=ntimes,
        key=key,
    )
    assert jnp.isfinite(val)
    assert grad_v.shape == theta_ests.shape
    assert jnp.all(jnp.isfinite(grad_v))
    np.testing.assert_allclose(grad, grad_v, atol=1e-7)
