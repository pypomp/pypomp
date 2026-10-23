"""The CuTe rproc backend. Needs nvidia-cutlass-dsl and a GPU; skipped otherwise."""

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
from pypomp.random.philox import PHILOX

ok, why = available()
pytestmark = pytest.mark.skipif(not ok, reason=f"CuTe backend unavailable: {why}")


def sde_rproc(X_, theta_, key, covars, t, dt):
    z = X_["x"]
    eps = jax.random.normal(key)
    drift = theta_["r"] * (1 - jnp.exp(z)) - theta_["sig"] ** 2 / 2
    return {"x": z + drift * dt + theta_["sig"] * jnp.sqrt(dt) * eps}


def sde_model(rproc=sde_rproc):
    def rinit(theta_, key, covars, t0):
        return {"x": theta_["z0m"] + theta_["z0s"] * jax.random.normal(key)}

    def dmeas(Y_, X_, theta_, covars, t):
        return jax.scipy.stats.norm.logpdf(Y_["y"], X_["x"], theta_["tau"])

    T = 40
    ys = pd.DataFrame(
        {"y": np.log(0.5) + 0.3 * np.random.default_rng(1).standard_normal(T)},
        index=np.arange(1, T + 1, dtype=float),
    )
    theta = {"r": 1.0, "sig": 0.3, "tau": 0.2, "z0m": float(np.log(0.2)), "z0s": 0.2}
    return pp.Pomp(
        ys=ys,
        theta=pp.PompParameters(theta),
        statenames=["x"],
        t0=0.0,
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        nstep=10,
    )


def he10_model():
    theta = {
        "R0": 56.8,
        "sigma": 28.9,
        "gamma": 30.4,
        "iota": 2.9,
        "rho": 0.488,
        "sigmaSE": 0.0878,
        "psi": 0.116,
        "cohort": 0.557,
        "amplitude": 0.554,
        "S_0": 2.97e-02,
        "E_0": 5.17e-05,
        "I_0": 5.14e-05,
        "R_0": 9.70e-01,
    }
    return pp.models.UKMeasles.pomp(
        unit="London", theta=pp.PompParameters(theta), clean=True, model="001b"
    )


def xla_and_cute(model):
    cm = deepcopy(model)
    cm.set_rproc_backend("cute")
    return model.to_struct(rproc_backend="xla"), cm.to_struct()


def interval(model, struct, J, R, i_obs=3):
    """Initial particles of R filters, and a function advancing one filter."""
    th = model.theta.to_jax_array(model.canonical_param_names)[0]
    cov0 = None if struct.covars_extended is None else struct.covars_extended[0]
    X0 = jax.vmap(
        lambda k: struct.rinit_pf(th, jax.random.split(k, J), cov0, struct.t0, False)
    )(jax.random.split(jax.random.key(9), R))
    nsteps = np.asarray(struct.nstep_array)
    t = float(np.asarray(struct.times)[i_obs - 1])
    args = (
        struct.covars_extended,
        struct.dt_array_extended,
        t,
        int(nsteps[:i_obs].sum()),
    )

    def advance(X, th, key, should_trans=False):
        return struct.rproc_pf(
            X, th, key, *args, int(nsteps[i_obs]), struct.accumvars, should_trans
        )[0]

    return X0, th, advance


@pytest.mark.parametrize("make", [sde_model, he10_model], ids=["sde", "he10"])
def test_cute_matches_xla_with_the_same_philox_words(make):
    model = make()
    xla, cute = xla_and_cute(model)
    assert cute.rproc_pf is not xla.rproc_pf
    keys = jax.random.split(jax.random.key(5, impl=PHILOX), 2)
    X0, th, adv_xla = interval(model, xla, 1024, 2)
    _, _, adv_cute = interval(model, cute, 1024, 2)
    a = jax.jit(jax.vmap(lambda X, k: adv_xla(X, th, k)))(X0, keys)
    b = jax.jit(jax.vmap(lambda X, k: adv_cute(X, th, k)))(X0, keys)
    rel = np.asarray(jnp.abs(a - b) / (jnp.abs(a) + 1.0))
    # Discrete draws can land on the other side of a CDF step from float rounding.
    assert (rel.max(-1) > 1e-4).mean() < 0.005
    assert np.median(rel) < 1e-6
    # All filters share one launch; a filter on its own gives the same particles.
    b0 = jax.jit(lambda X, k: adv_cute(X, th, k))(X0[0], keys[0])
    np.testing.assert_array_equal(np.asarray(b0), np.asarray(b[0]))


@pytest.mark.parametrize("should_trans", [False, True], ids=["natural", "estimation"])
@pytest.mark.parametrize("make", [sde_model, he10_model], ids=["sde", "he10"])
def test_cute_gradients_match_xla(make, should_trans):
    model = make()
    xla, cute = xla_and_cute(model)
    X0, th, adv_xla = interval(model, xla, 256, 2)
    _, _, adv_cute = interval(model, cute, 256, 2)
    keys = jax.random.split(jax.random.key(5, impl=PHILOX), 2)
    w = jnp.cos(0.1 * jnp.arange(X0.size, dtype=jnp.float32)).reshape(X0.shape)

    def grads(advance):
        def loss(X, th):
            out = jax.vmap(lambda X, k: advance(X, th, k, should_trans))(X, keys)
            return jnp.sum(out * w)

        return jax.jit(jax.grad(loss, argnums=(0, 1)))(X0, th)

    for a, b in zip(grads(adv_xla), grads(adv_cute), strict=True):
        a, b = np.asarray(a), np.asarray(b)
        rel = np.abs(a - b) / (np.abs(a) + 1e-3 * np.abs(a).max() + 1e-12)
        assert np.median(rel) < 1e-4
        assert (rel > 1e-2).mean() < 0.01


def test_cute_loglik_agrees_with_xla():
    model = sde_model()
    xla, cute = xla_and_cute(model)
    th = jnp.tile(model.theta.to_jax_array(model.canonical_param_names), (200, 1))
    keys = jax.random.split(jax.random.key(0), 200)[:, None]
    a = np.asarray(F.pfilter(xla, th, 500, keys)["logLik"]).ravel()
    b = np.asarray(F.pfilter(cute, th, 500, keys)["logLik"]).ravel()
    se = np.sqrt(a.var() / a.size + b.var() / b.size)
    assert abs(a.mean() - b.mean()) < 4 * se


def test_mif_and_train_on_cute():
    model = sde_model()
    model.set_rproc_backend("xla")
    cm = deepcopy(model)
    cm.set_rproc_backend("cute")
    rw = pp.RWSigma(
        sigmas={"r": 0.02, "sig": 0.02, "tau": 0.02, "z0m": 0.0, "z0s": 0.0},
        init_names=[],
    ).geometric_cooling(0.5)
    searched = deepcopy(cm)
    searched.mif(J=256, M=3, rw_sd=rw, key=jax.random.key(0))
    assert np.isfinite(searched.results()["logLik"].to_numpy(dtype=float)).all()

    eta = pp.LearningRate({p: 0.05 for p in model.canonical_param_names})
    key = jax.random.key(2, impl=PHILOX)
    model.train(J=256, M=3, eta=eta, optimizer=pp.Adam(scale=True), key=key)
    cm.train(J=256, M=3, eta=eta, optimizer=pp.Adam(scale=True), key=key)
    a = np.asarray(model.results_history[-1].traces_da.sel(theta_idx=0))
    b = np.asarray(cm.results_history[-1].traces_da.sel(theta_idx=0))
    np.testing.assert_allclose(b, a, rtol=2e-3, atol=2e-3)


def test_unsupported_rproc_falls_back_to_xla():
    def rproc(X_, theta_, key, covars, t, dt):
        z = X_["x"]
        return {"x": z + jnp.sort(jnp.stack([z, -z]))[0] * dt}

    model = sde_model(rproc)
    model.set_rproc_backend("cute")
    with pytest.warns(UserWarning, match="sort"):
        s = model.to_struct()
    assert s.rproc_pf is model.rproc.mechanics_pf_interp


def test_auto_uses_cute_only_with_several_steps_per_observation():
    model = sde_model()
    assert model.to_struct().rproc_pf is not model.rproc.mechanics_pf_interp
    one_step = pp.models.lg(T=10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = one_step.to_struct()
    assert s.rproc_pf is one_step.rproc.mechanics_pf_interp
