"""The CuTe rproc backend: needs nvidia-cutlass-dsl and a GPU, skipped otherwise."""

import warnings
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import pypomp as pp
import pypomp.functional as F
from pypomp.cute import available

ok, why = available()
pytestmark = pytest.mark.skipif(not ok, reason=f"CuTe backend unavailable: {why}")


def sde_model(T=40):
    def rinit(theta_, key, covars, t0):
        return {"x": theta_["z0m"] + theta_["z0s"] * jax.random.normal(key)}

    def rproc(X_, theta_, key, covars, t, dt):
        z = X_["x"]
        eps = jax.random.normal(key)
        return {"x": z + (theta_["r"] * (1 - jnp.exp(z)) - theta_["sig"] ** 2 / 2) * dt
                + theta_["sig"] * jnp.sqrt(dt) * eps}

    def dmeas(Y_, X_, theta_, covars, t):
        return jax.scipy.stats.norm.logpdf(Y_["y"], X_["x"], theta_["tau"])

    rng = np.random.default_rng(1)
    ys = pd.DataFrame({"y": np.log(0.5) + 0.3 * rng.standard_normal(T)},
                      index=np.arange(1, T + 1, dtype=float))
    theta = {"r": 1.0, "sig": 0.3, "tau": 0.2, "z0m": float(np.log(0.2)), "z0s": 0.2}
    return pp.Pomp(ys=ys, theta=pp.PompParameters(theta), statenames=["x"], t0=0.0, rinit=rinit,
                   rproc=rproc, dmeas=dmeas, nstep=10)


def he10_model():
    theta = {"R0": 56.8, "sigma": 28.9, "gamma": 30.4, "iota": 2.9, "rho": 0.488, "sigmaSE": 0.0878,
             "psi": 0.116, "cohort": 0.557, "amplitude": 0.554, "S_0": 2.97e-02, "E_0": 5.17e-05,
             "I_0": 5.14e-05, "R_0": 9.70e-01}
    return pp.models.UKMeasles.pomp(unit="London", theta=pp.PompParameters(theta), clean=True, model="001b")


def one_interval(model, struct, key, i_obs=3, J=1024, R=2):
    """Propagate R filters of J particles over observation interval i_obs."""
    th = model.theta.to_jax_array(model.canonical_param_names)[0]
    cov0 = None if struct.covars_extended is None else struct.covars_extended[0]
    X0 = jax.vmap(lambda k: struct.rinit_pf(th, jax.random.split(k, J), cov0, struct.t0, False))(
        jax.random.split(jax.random.key(9), R))
    nsteps = np.asarray(struct.nstep_array)
    t_idx = int(nsteps[:i_obs].sum())
    t = float(np.asarray(struct.times)[i_obs - 1]) if i_obs else float(struct.t0)

    def f(X, k):
        return struct.rproc_pf(X, th, k, struct.covars_extended, struct.dt_array_extended, t, t_idx,
                               int(nsteps[i_obs]), struct.accumvars, False)[0]

    keys = jax.random.split(key, R)
    return jax.jit(jax.vmap(f))(X0, keys), jax.jit(f)(X0[0], keys[0])


@pytest.mark.parametrize("make", [sde_model, he10_model], ids=["per_particle_sde", "vectorized_he10"])
def test_cute_matches_xla_with_the_same_philox_words(make):
    model = make()
    xla = model.to_struct()
    cm = deepcopy(model)
    cm.set_rproc_backend("cute")
    cute = cm.to_struct()
    assert cute.rproc_pf is not xla.rproc_pf
    key = jax.random.key(5, impl=pp.random.PHILOX)
    a, _ = one_interval(model, xla, key)
    b, b0 = one_interval(model, cute, key)
    rel = np.asarray(jnp.abs(a - b) / (jnp.abs(a) + 1.0))
    # Discrete draws can land on the other side of a CDF step from float rounding.
    assert (rel.max(-1) > 1e-4).mean() < 0.005
    assert np.median(rel) < 1e-6
    np.testing.assert_array_equal(np.asarray(b0), np.asarray(b[0]))  # one launch for all filters


def test_cute_loglik_agrees_with_stock_threefry():
    model = sde_model()
    cm = deepcopy(model)
    cm.set_rproc_backend("cute")
    th = jnp.tile(model.theta.to_jax_array(model.canonical_param_names), (200, 1))
    keys = jax.random.split(jax.random.key(0), 200)[:, None]
    stock = np.asarray(F.pfilter(model.to_struct(), th, 500, keys)["logLik"]).ravel()
    cute = np.asarray(F.pfilter(cm.to_struct(), th, 500, keys)["logLik"]).ravel()
    se = np.sqrt(stock.var() / stock.size + cute.var() / cute.size)
    assert abs(stock.mean() - cute.mean()) < 4 * se


def test_mif_runs_on_cute():
    model = sde_model()
    model.set_rproc_backend("cute")
    rw = pp.RWSigma(sigmas={"r": 0.02, "sig": 0.02, "tau": 0.02, "z0m": 0.0, "z0s": 0.0},
                    init_names=[]).geometric_cooling(0.5)
    model.mif(J=256, M=3, rw_sd=rw, key=jax.random.key(0))
    assert np.isfinite(model.results()["logLik"].to_numpy(dtype=float)).all()


def test_unsupported_rproc_falls_back_to_xla():
    model = sde_model()

    def rproc(X_, theta_, key, covars, t, dt):
        z = X_["x"]
        return {"x": z + jnp.sort(jnp.stack([z, -z]))[0] * dt + 0.1 * jax.random.normal(key)}

    model2 = pp.Pomp(ys=model.ys, theta=model.theta, statenames=["x"], t0=0.0, rinit=model.rinit.original_func,
                     rproc=rproc, dmeas=model.dmeas.original_func, nstep=10)
    model2.set_rproc_backend("cute")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = model2.to_struct()
    assert any("sort" in str(x.message) for x in w)
    assert s.rproc_pf is model2.rproc.mechanics_pf_interp


@pytest.mark.parametrize("should_trans", [False, True], ids=["natural", "estimation_scale"])
@pytest.mark.parametrize("make", [sde_model, he10_model], ids=["per_particle_sde", "vectorized_he10"])
def test_cute_gradients_match_xla(make, should_trans):
    model = make()
    xla = model.to_struct()
    cm = deepcopy(model)
    cm.set_rproc_backend("cute")
    cute = cm.to_struct()
    th = model.theta.to_jax_array(model.canonical_param_names)[0]
    if should_trans:
        th = jnp.asarray(model.par_trans.to_est_array(th, model.canonical_param_names)) \
            if hasattr(model.par_trans, "to_est_array") else th
    J, R, i_obs = 256, 2, 3
    cov0 = None if xla.covars_extended is None else xla.covars_extended[0]
    X0 = jax.vmap(lambda k: xla.rinit_pf(model.theta.to_jax_array(model.canonical_param_names)[0],
                                         jax.random.split(k, J), cov0, xla.t0, False))(
        jax.random.split(jax.random.key(9), R))
    nsteps = np.asarray(xla.nstep_array)
    t_idx = int(nsteps[:i_obs].sum())
    t = float(np.asarray(xla.times)[i_obs - 1])
    keys = jax.random.split(jax.random.key(5, impl=pp.random.PHILOX), R)
    w = jnp.cos(0.1 * jnp.arange(X0.size, dtype=jnp.float32)).reshape(X0.shape)

    def loss(struct, X, th):
        out = jax.vmap(lambda X, k: struct.rproc_pf(X, th, k, struct.covars_extended, struct.dt_array_extended, t,
                                                    t_idx, int(nsteps[i_obs]), struct.accumvars, should_trans)[0]
                       )(X, keys)
        return jnp.sum(out * w)

    ga = jax.jit(jax.grad(lambda X, th: loss(xla, X, th), argnums=(0, 1)))(X0, th)
    gb = jax.jit(jax.grad(lambda X, th: loss(cute, X, th), argnums=(0, 1)))(X0, th)
    for a, b in zip(ga, gb):
        a, b = np.asarray(a), np.asarray(b)
        rel = np.abs(a - b) / (np.abs(a) + 1e-3 * np.abs(a).max() + 1e-12)
        assert np.median(rel) < 1e-4, np.median(rel)
        assert (rel > 1e-2).mean() < 0.01


def test_train_on_cute_matches_xla():
    model = sde_model()
    cm = deepcopy(model)
    cm.set_rproc_backend("cute")
    eta = pp.LearningRate({p: 0.05 for p in model.canonical_param_names})
    key = jax.random.key(2, impl=pp.random.PHILOX)
    model.train(J=256, M=3, eta=eta, optimizer=pp.Adam(scale=True), key=key)
    cm.train(J=256, M=3, eta=eta, optimizer=pp.Adam(scale=True), key=key)
    a = np.asarray(model.results_history[-1].traces_da.sel(theta_idx=0))
    b = np.asarray(cm.results_history[-1].traces_da.sel(theta_idx=0))
    assert np.isfinite(b).all()
    np.testing.assert_allclose(b, a, rtol=2e-3, atol=2e-3)
