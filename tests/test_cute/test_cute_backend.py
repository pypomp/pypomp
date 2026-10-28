"""The CuTe backend. Needs nvidia-cutlass-dsl and a GPU; skipped otherwise."""

import os
import stat
import warnings
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import pypomp as pp
import pypomp.functional as F
from pypomp.cute import KernelRProc, available
from pypomp.random import binominv
from pypomp.random.philox import PHILOX

ok, why = available()
pytestmark = pytest.mark.skipif(not ok, reason=f"CuTe backend unavailable: {why}")


def sde_rproc(X_, theta_, key, covars, t, dt):
    z = X_["x"]
    eps = jax.random.normal(key)
    drift = theta_["r"] * (1 - jnp.exp(z)) - theta_["sig"] ** 2 / 2
    return {"x": z + drift * dt + theta_["sig"] * jnp.sqrt(dt) * eps}


def reshape_rproc(X_, theta_, key, covars, t, dt):
    z = X_["x"]
    m = jnp.stack([z, 2 * z, 3 * z, 4 * z]).reshape(2, 2)
    v = jax.lax.reshape(m, (4,), dimensions=(1, 0))  # [z, 3z, 2z, 4z]
    return sde_rproc({"x": z + (v[1] - v[2]) * dt}, theta_, key, covars, t, dt)


@pp.vectorized
def vectorized_rproc(X_, theta_, key, covars, t, dt):
    z = jnp.stack([X_["x"], 2 * X_["x"]])[0]  # stacking keeps each particle's values
    z = jnp.where(jnp.isnan(z), jnp.nan, z)  # a NaN constant, the same at any J
    eps = jax.random.normal(key, z.shape)
    drift = theta_["r"] * (1 - jnp.exp(z)) - theta_["sig"] ** 2 / 2
    return {"x": z + drift * dt + theta_["sig"] * jnp.sqrt(dt) * eps}


def int_ops_rproc(X_, theta_, key, covars, t, dt):
    z = X_["x"]
    n = jnp.round(z * 100).astype(jnp.int32)  # negative, as z is near log(0.2)
    m = (
        n // 7
        + jax.lax.rem(n, 7)
        + jax.lax.div(n, 3)
        + jax.lax.shift_right_logical(n, 28)
        + jax.lax.shift_right_arithmetic(n, 3)
    )
    w = jnp.sign(z) * jnp.fmod(z, 0.3)
    return sde_rproc({"x": z + (0.01 * m + w) * dt}, theta_, key, covars, t, dt)


def two_binomials_rproc(X_, theta_, key, covars, t, dt):
    z = X_["x"]
    u = jax.random.uniform(key, (2,))
    n = jnp.round(100 * jnp.exp(z))
    # Runtime p in both calls, so they share one pattern of constant inputs.
    a = binominv(u[0], n, theta_["sig"])
    b = binominv(u[1], n + 7.0, 0.1 * theta_["sig"])
    return sde_rproc({"x": z + 0.01 * (a - 30 * b) * dt}, theta_, key, covars, t, dt)


# Tables read at runtime indices: a unit's row, picked by a covariate, at the current
# week; its neighbours' rows; and grid points picked by the state, one of them read
# in "fill" mode, which the state takes out of range now and then.
_WEEKLY = np.random.default_rng(3).normal(size=(4, 60)).astype(np.float32)
_NEAR = np.array([[1, 2], [0, 3], [3, 0], [2, 1]], np.int32)
_GRID = np.linspace(-1.0, 1.0, 101, dtype=np.float32)


def table_rproc(X_, theta_, key, covars, t, dt):
    z = X_["x"]
    i = jnp.round(covars["u"]).astype(jnp.int32)
    k = jnp.clip(jnp.floor(t), 0, 59).astype(jnp.int32)
    own = jnp.asarray(_WEEKLY)[i][k]
    near = jnp.asarray(_WEEKLY)[jnp.asarray(_NEAR)[i], k]
    g = jnp.floor((z + 2.0) * 25.0).astype(jnp.int32)
    grid = jnp.asarray(_GRID)[jnp.clip(g, 0, 100)]
    edge = jnp.asarray(_GRID).at[g].get(mode="fill", fill_value=-5.0)
    nudge = 0.1 * own + 0.05 * near.sum() + 0.1 * grid + 0.01 * edge
    return sde_rproc({"x": z + nudge * dt}, theta_, key, covars, t, dt)


def picks_rproc(X_, theta_, key, covars, t, dt):
    # Runtime values read at runtime indices, some out of range: a negative index
    # wraps, then dynamic_slice clamps and "fill" mode fills.
    z = X_["x"]
    w = jnp.stack([z, theta_["r"] * z, -z, z * z])
    i = jnp.floor((z + 2.0) * 3.0).astype(jnp.int32) - 2
    rows = jax.lax.dynamic_slice(w.reshape(2, 2), (i % 3 - 1, 0), (1, 2))
    edge = w.at[i].get(mode="fill", fill_value=0.5)
    nudge = 0.1 * w[i] + 0.05 * rows.sum() + 0.1 * edge
    return sde_rproc({"x": z + nudge * dt}, theta_, key, covars, t, dt)


def dpop_rproc(X_, theta_, key, covars, t, dt):
    out = sde_rproc(X_, theta_, key, covars, t, dt)
    # Any function of theta shows whether _logw is reset at each observation.
    return {"x": out["x"], "_logw": X_["_logw"] + theta_["r"] * out["x"] * dt}


def sde_dmeas(Y_, X_, theta_, covars, t):
    return jax.scipy.stats.norm.logpdf(Y_["y"], X_["x"], theta_["tau"])


def sde_model(
    rproc=sde_rproc,
    statenames=("x",),
    covars=None,
    dmeas=sde_dmeas,
    times=None,
    dt=None,
):
    def rinit(theta_, key, covars, t0):
        x0 = theta_["z0m"] + theta_["z0s"] * jax.random.normal(key)
        return {"x": x0, **{n: 0.0 for n in statenames[1:]}}

    if times is None:
        times = np.arange(1, 41, dtype=float)
    T = len(times)
    ys = pd.DataFrame(
        {"y": np.log(0.5) + 0.3 * np.random.default_rng(1).standard_normal(T)},
        index=np.asarray(times, dtype=float),
    )
    theta = {"r": 1.0, "sig": 0.3, "tau": 0.2, "z0m": float(np.log(0.2)), "z0s": 0.2}
    return pp.Pomp(
        ys=ys,
        theta=pp.PompParameters(theta),
        statenames=list(statenames),
        t0=0.0,
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        nstep=None if dt else 10,
        dt=dt,
        covars=covars,
    )


def uneven_model(rproc=sde_rproc):
    """sde_model observed at uneven times, so its intervals take 4, 2, 8 and 3
    Euler steps in turn."""
    return sde_model(rproc, times=np.cumsum(np.resize([1, 0.5, 2, 0.75], 40)), dt=0.25)


def table_model(u=2):
    covars = pd.DataFrame({"u": float(u)}, index=np.arange(0.0, 41.0))
    return sde_model(table_rproc, covars=covars)


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


def xla_and_cute(model, rng="threefry"):
    cm = deepcopy(model)
    cm.set_backend("cute", rng=rng)
    return model.to_struct(backend="xla"), cm.to_struct()


def rng_key(seed, rng="threefry"):
    """A key of the generator `rng`, which XLA and the kernel draw alike from."""
    return jax.random.key(seed, impl="threefry2x32" if rng == "threefry" else PHILOX)


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


@pytest.mark.parametrize(
    "make",
    [
        sde_model,
        he10_model,
        lambda: sde_model(reshape_rproc),
        lambda: sde_model(vectorized_rproc),
        lambda: sde_model(int_ops_rproc),
        table_model,
        lambda: sde_model(picks_rproc),
        uneven_model,
    ],
    ids=[
        "sde",
        "he10",
        "reshape",
        "vectorized",
        "int_ops",
        "tables",
        "picks",
        "uneven",
    ],
)
@pytest.mark.parametrize("rng", ["threefry", "philox"])
def test_cute_matches_xla_with_the_same_random_numbers(make, rng):
    model = make()
    xla, cute = xla_and_cute(model, rng)
    assert cute.rproc_pf is not xla.rproc_pf
    keys = jax.random.split(rng_key(5, rng), 2)
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
    from pypomp.cute import kernel

    assert stat.S_IMODE(os.stat(kernel._generated_dir()).st_mode) == 0o700


@pytest.mark.parametrize("should_trans", [False, True], ids=["natural", "estimation"])
@pytest.mark.parametrize(
    "make",
    [sde_model, he10_model, table_model, uneven_model],
    ids=["sde", "he10", "tables", "uneven"],
)
def test_cute_gradients_match_xla(make, should_trans):
    model = make()
    xla, cute = xla_and_cute(model)
    X0, th, adv_xla = interval(model, xla, 256, 2)
    _, _, adv_cute = interval(model, cute, 256, 2)
    keys = jax.random.split(rng_key(5), 2)
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


def test_filters_with_different_times_share_a_launch():
    model = he10_model()
    xla, cute = xla_and_cute(model)
    X0, th, _ = interval(model, xla, 512, 2)
    keys = jax.random.split(rng_key(5), 2)
    nsteps = np.asarray(xla.nstep_array)
    i_obs = np.array([3, 7])  # different times and covariates for each filter
    ts = jnp.asarray(np.asarray(xla.times)[i_obs - 1])
    t_idx = jnp.asarray(np.cumsum(nsteps)[i_obs - 1])

    def run(s):
        def f(X, k, t, t_i):
            return s.rproc_pf(
                X,
                th,
                k,
                s.covars_extended,
                s.dt_array_extended,
                t,
                t_i,
                int(nsteps[3]),
                s.accumvars,
                False,
            )[0]

        return jax.jit(jax.vmap(f))(X0, keys, ts, t_idx)

    a, b = run(xla), run(cute)
    rel = np.asarray(jnp.abs(a - b) / (jnp.abs(a) + 1.0))
    assert (rel.max(-1) > 1e-4).mean() < 0.005
    assert np.median(rel) < 1e-6


def test_intervals_of_different_lengths_share_a_launch():
    model = uneven_model()
    assert model.rproc.nstep is None
    xla, cute = xla_and_cute(model)
    assert isinstance(cute.rproc_pf, KernelRProc)
    X0, th, _ = interval(model, xla, 512, 3)
    keys = jax.random.split(rng_key(5), 3)
    nsteps = np.asarray(xla.nstep_array)
    i_obs = np.array([1, 2, 3])  # 2, 8 and 3 Euler steps
    ts = jnp.asarray(np.asarray(xla.times)[i_obs - 1])
    t_idx = jnp.asarray(np.cumsum(nsteps)[i_obs - 1])
    n = jnp.asarray(nsteps[i_obs])

    def run(s):
        def f(X, k, t, t_i, n):
            return s.rproc_pf(
                X,
                th,
                k,
                s.covars_extended,
                s.dt_array_extended,
                t,
                t_i,
                n,
                s.accumvars,
                False,
            )

        return jax.jit(jax.vmap(f))(X0, keys, ts, t_idx, n)

    (a, ta), (b, tb) = run(xla), run(cute)
    np.testing.assert_array_equal(np.asarray(tb), np.asarray(ta))
    rel = np.asarray(jnp.abs(a - b) / (jnp.abs(a) + 1.0))
    assert (rel.max(-1) > 1e-4).mean() < 0.005
    assert np.median(rel) < 1e-6


def test_repeated_calls_share_a_device_function(monkeypatch):
    from pypomp.cute import translate

    defined, called = [], []
    define, call = translate._define, translate._call_outlined

    def spy(log, f):
        return lambda *a: log.append(id(a[0])) or f(*a)

    monkeypatch.setattr(translate, "_define", spy(defined, define))
    monkeypatch.setattr(translate, "_call_outlined", spy(called, call))
    model = sde_model(two_binomials_rproc)
    xla, cute = xla_and_cute(model)
    X0, th, adv_xla = interval(model, xla, 1024, 2)
    _, _, adv_cute = interval(model, cute, 1024, 2)
    keys = jax.random.split(rng_key(5), 2)
    a = jax.jit(jax.vmap(lambda X, k: adv_xla(X, th, k)))(X0, keys)
    b = jax.jit(jax.vmap(lambda X, k: adv_cute(X, th, k)))(X0, keys)
    rel = np.asarray(jnp.abs(a - b) / (jnp.abs(a) + 1.0))
    assert (rel.max(-1) > 1e-4).mean() < 0.005
    assert np.median(rel) < 1e-6
    # Each kernel emits binominv once and calls it twice.
    assert any(called.count(j) == 2 * defined.count(j) for j in set(defined))


@pytest.mark.parametrize("make", [sde_model, uneven_model], ids=["sde", "uneven"])
def test_cute_loglik_agrees_with_xla(make):
    model = make()
    xla, cute = xla_and_cute(model)
    th = jnp.tile(model.theta.to_jax_array(model.canonical_param_names), (200, 1))
    keys = jax.random.split(jax.random.key(0), 200)[:, None]
    a = np.asarray(F.pfilter(xla, th, 500, keys)["logLik"]).ravel()
    b = np.asarray(F.pfilter(cute, th, 500, keys)["logLik"]).ravel()
    se = np.sqrt(a.var() / a.size + b.var() / b.size)
    assert abs(a.mean() - b.mean()) < 4 * se


def table_panel():
    units = {f"u{u}": table_model(u) for u in (1, 3)}
    names = units["u1"].canonical_param_names
    vals = units["u1"].theta.to_jax_array(names)[0]
    v = dict(zip(names, np.asarray(vals, float).tolist(), strict=True))
    # One unit-specific parameter, so that the parameters name the units.
    shared = pd.DataFrame(
        {"shared": [v[n] for n in names if n != "z0m"]},
        index=[n for n in names if n != "z0m"],
    )
    unit = pd.DataFrame({u: [v["z0m"]] for u in units}, index=["z0m"])
    theta = pp.PanelParameters({"shared": shared, "unit_specific": unit})
    return pp.PanelPomp(units, theta), names, vals


@pytest.mark.parametrize("rng", ["threefry", "philox"])
def test_panel_runs_its_first_units_kernel_for_every_unit(rng):
    panel, names, th = table_panel()
    panel.set_backend("cute", rng)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cute = panel.to_struct()
    assert isinstance(cute.rproc_pf, KernelRProc)
    assert isinstance(cute.rproc_per, KernelRProc)
    assert all(u._theta is None for u in panel.unit_objects.values())
    xla = panel.to_struct(backend="xla")
    assert xla.rproc_pf is panel.unit_objects["u1"].rproc.mechanics_pf_interp

    # Each unit's covariate picks its own rows of the tables.
    J, i_obs = 1024, 3
    nsteps = np.asarray(xla.nstep_array)
    t, t_idx = float(np.asarray(xla.times)[i_obs - 1]), int(nsteps[:i_obs].sum())
    covs = xla.covars_per_unit
    X0 = jax.vmap(
        lambda k, c: xla.rinit_pf(th, jax.random.split(k, J), c[0], xla.t0, False)
    )(jax.random.split(jax.random.key(9), 2), covs)
    keys = jax.random.split(rng_key(5, rng), 2)

    def advance(s):
        def f(X, k, c):
            return s.rproc_pf(
                X,
                th,
                k,
                c,
                s.dt_array_extended,
                t,
                t_idx,
                int(nsteps[i_obs]),
                s.accumvars,
                False,
            )[0]

        return np.asarray(jax.jit(jax.vmap(f))(X0, keys, covs))

    a, b = advance(xla), advance(cute)
    assert not np.allclose(a[0], a[1])
    rel = np.abs(a - b) / (np.abs(a) + 1.0)
    assert (rel.max(-1) > 1e-4).mean() < 0.005
    assert np.median(rel) < 1e-6

    panel.pfilter(J=256, key=jax.random.key(0))
    assert np.isfinite(panel.results()["shared logLik"].to_numpy(dtype=float)).all()


def test_mif_and_train_on_cute():
    model = sde_model()
    model.set_backend("xla")
    cm = deepcopy(model)
    cm.set_backend("cute")
    rw = pp.RWSigma(
        sigmas={"r": 0.02, "sig": 0.02, "tau": 0.02, "z0m": 0.0, "z0s": 0.0},
        init_names=[],
    ).geometric_cooling(0.5)
    searched = deepcopy(cm)
    searched.mif(J=256, M=3, rw_sd=rw, key=jax.random.key(0))
    assert np.isfinite(searched.results()["logLik"].to_numpy(dtype=float)).all()

    eta = pp.LearningRate({p: 0.05 for p in model.canonical_param_names})
    key = rng_key(2)
    model.train(J=256, M=3, eta=eta, optimizer=pp.Adam(scale=True), key=key)
    cm.train(J=256, M=3, eta=eta, optimizer=pp.Adam(scale=True), key=key)
    a = np.asarray(model.results_history[-1].traces_da.sel(theta_idx=0))
    b = np.asarray(cm.results_history[-1].traces_da.sel(theta_idx=0))
    np.testing.assert_allclose(b[:, 1:], a[:, 1:], rtol=2e-3, atol=2e-3)
    # Float rounding can flip a resampling decision, which moves the log-likelihood
    # (column 0) by about its Monte Carlo error.
    np.testing.assert_allclose(b[:, 0], a[:, 0], atol=1.0)


def test_mif_with_intervals_of_different_lengths():
    model = uneven_model()
    model.set_backend("cute")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert isinstance(model.to_struct().rproc_per, KernelRProc)
    rw = pp.RWSigma(
        sigmas={"r": 0.02, "sig": 0.02, "tau": 0.02, "z0m": 0.0, "z0s": 0.0},
        init_names=[],
    ).geometric_cooling(0.5)
    model.mif(J=256, M=3, rw_sd=rw, key=jax.random.key(0))
    assert np.isfinite(model.results()["logLik"].to_numpy(dtype=float)).all()


def test_dpop_train_on_cute():
    model = sde_model(dpop_rproc, statenames=("x", "_logw"))
    assert model.rproc.accumvars == (1,)  # _logw is zeroed at each observation
    model.set_backend("xla")
    cm = deepcopy(model)
    cm.set_backend("cute")
    eta = pp.LearningRate({p: 0.05 for p in model.canonical_param_names})
    key = rng_key(2)
    opt = pp.Adam(scale=True)
    model.train(J=256, M=3, eta=eta, optimizer=opt, key=key, dpop=True)
    cm.train(J=256, M=3, eta=eta, optimizer=opt, key=key, dpop=True)
    a = np.asarray(model.results_history[-1].traces_da.sel(theta_idx=0))
    b = np.asarray(cm.results_history[-1].traces_da.sel(theta_idx=0))
    np.testing.assert_allclose(b[:, 1:], a[:, 1:], rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(b[:, 0], a[:, 0], atol=1.0)  # as in the test above


def sorts(X_, theta_, key, covars, t, dt):
    z = X_["x"]
    return {"x": z + jnp.sort(jnp.stack([z, -z]))[0] * dt}


def runtime_index_of_many(X_, theta_, key, covars, t, dt):
    z = X_["x"]
    i = jnp.asarray(t, jnp.int32)
    return {"x": z + (z * jnp.arange(100.0))[i] * dt}


def rounds(X_, theta_, key, covars, t, dt):
    return {"x": jax.lax.reduce_precision(X_["x"], exponent_bits=8, mantissa_bits=7)}


@pp.vectorized
def particle_mean(X_, theta_, key, covars, t, dt):
    z = X_["x"]
    return {"x": z + jnp.mean(z) * dt + 0.1 * jax.random.normal(key, z.shape)}


@pp.vectorized
def particles_first(X_, theta_, key, covars, t, dt):
    z = X_["x"]
    return {"x": z + 0.1 * jax.random.normal(key, (z.shape[0], 2))[:, 0]}


@pp.vectorized
def uses_J(X_, theta_, key, covars, t, dt):
    z = X_["x"]
    return {"x": z + dt / z.shape[0]}


@pytest.mark.parametrize(
    "rproc",
    [sorts, runtime_index_of_many, rounds, particle_mean, particles_first, uses_J],
    ids=lambda f: f.__name__,
)
def test_rprocs_the_kernel_cannot_run_exactly_stay_on_xla(rproc):
    model = sde_model(rproc)
    model.set_backend("cute")
    with pytest.warns(UserWarning, match="Using XLA"):
        s = model.to_struct()
    assert s.rproc_pf is model.rproc.mechanics_pf_interp


def test_unchecked_kernels_skip_the_numerical_checks(monkeypatch):
    import pypomp.cute.verify as verify

    def ran(*args):
        raise AssertionError("the check ran")

    for name in ("_check", "_check_grads", "_check_dmeas", "_check_perturb_step"):
        monkeypatch.setattr(verify, name, ran)
    model = sde_model()
    model.theta = None
    model.set_backend("cute", check=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = model.to_struct()
    assert isinstance(s.rproc_pf, KernelRProc)
    assert isinstance(s.rproc_per, KernelRProc)


def test_kernel_whose_gradients_differ_is_differentiated_on_xla(monkeypatch):
    import pypomp.cute.verify as verify
    from pypomp.cute.verify import _grads, _interval

    monkeypatch.setattr(verify, "_check_grads", lambda *args: "50% differ")
    model = sde_model()
    model.set_backend("cute")
    with pytest.warns(UserWarning, match="differentiating through XLA"):
        s = model.to_struct()
    assert isinstance(s.rproc_pf, KernelRProc)
    iv = _interval(model, "threefry")
    np.testing.assert_allclose(
        _grads(iv, s.rproc_per), _grads(iv, model.rproc.mechanics_per_interp), rtol=1e-5
    )


def test_gradients_that_cancel_to_rounding_pass_the_check():
    from pypomp.models import spx

    model = spx()  # mu's gradient through S cancels to float32 rounding
    model.set_backend("cute")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = model.to_struct()
    assert isinstance(s.rproc_per, KernelRProc)


def test_newton_training_uses_xla():
    model = sde_model()
    model.set_backend("cute")
    eta = pp.LearningRate({p: 0.05 for p in model.canonical_param_names})
    with pytest.warns(UserWarning, match="Hessians"):
        model.train(J=64, M=1, eta=eta, optimizer=pp.Newton(), key=jax.random.key(0))
    assert np.isfinite(np.asarray(model.results_history[-1].traces_da)).all()


def test_keys_of_another_generator_run_on_xla_with_threefry():
    model = sde_model()
    xla, cute = xla_and_cute(model, "threefry")
    X0, th, adv_xla = interval(model, xla, 256, 1)
    _, _, adv_cute = interval(model, cute, 256, 1)
    key = rng_key(5, "philox")
    a = jax.jit(lambda X: adv_xla(X, th, key))(X0[0])
    b = jax.jit(lambda X: adv_cute(X, th, key))(X0[0])
    np.testing.assert_array_equal(np.asarray(b), np.asarray(a))


def test_rbg_keys_run_on_xla():
    model = sde_model()
    model.set_backend("cute")
    model.pfilter(J=100, key=jax.random.key(0, impl="rbg"))
    assert np.isfinite(model.results()["logLik"].to_numpy(dtype=float)).all()


def test_rbg_default_keeps_xla_quietly():
    jax.config.update("jax_default_prng_impl", "rbg")
    try:
        model = sde_model()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            s = model.to_struct()
        assert s.rproc_pf is model.rproc.mechanics_pf_interp
    finally:
        jax.config.update("jax_default_prng_impl", "threefry2x32")


def test_float64_particles_run_on_xla():
    jax.config.update("jax_enable_x64", True)
    try:
        model = sde_model()
        model.pfilter(J=100, key=jax.random.key(0))
        assert np.isfinite(model.results()["logLik"].to_numpy(dtype=float)).all()
    finally:
        jax.config.update("jax_enable_x64", False)


def test_auto_uses_cute_with_one_step_per_observation():
    one_step = pp.models.lg(T=10)
    assert one_step.rproc.nstep == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = one_step.to_struct()
    assert isinstance(s.rproc_pf, KernelRProc)


def test_gpus_older_than_ampere_are_unavailable(monkeypatch):
    class Volta:
        platform = "gpu"
        compute_capability = "7.0"

    monkeypatch.setattr(jax, "devices", lambda backend=None: [Volta()])
    ok, why = available()
    assert not ok and "7.0" in why


def test_builds_without_a_cpu_backend(monkeypatch):
    devices = jax.devices

    def no_cpu(backend=None):
        if backend == "cpu":
            raise RuntimeError("Unknown backend cpu. Available backends are ['cuda']")
        return devices(backend)

    monkeypatch.setattr(jax, "devices", no_cpu)
    model = sde_model()
    assert model.to_struct().rproc_pf is not model.rproc.mechanics_pf_interp


def test_kernel_that_differs_from_xla_is_not_used(monkeypatch):
    from pypomp.cute import kernel

    model = sde_model()
    make = kernel._make_rprocess_interp

    def off(model, traced, per_particle, rng, dm_j):
        pf = make(model, traced, per_particle, rng, dm_j)
        return lambda X_, *args: (pf(X_, *args)[0] + 0.1, pf(X_, *args)[1])

    monkeypatch.setattr(kernel, "_make_rprocess_interp", off)
    with pytest.warns(UserWarning, match="differ"):
        s = model.to_struct()
    assert s.rproc_pf is model.rproc.mechanics_pf_interp


def test_report_says_every_part_runs_on_cute():
    model = sde_model()
    model.set_backend("cute")
    report = model.backend_report()
    assert report.available and not report.reason
    assert all(p.cute for p in report.parts), str(report)


def dmeas_at_runtime_index(Y_, X_, theta_, covars, t):
    # Too many runtime values to select among in registers.
    v = X_["x"] * jnp.arange(100.0)
    i = jnp.clip(jnp.round(10 * X_["x"]).astype(jnp.int32), 0, 99)
    return jax.scipy.stats.norm.logpdf(Y_["y"], v[i], theta_["tau"])


def test_report_points_at_what_keeps_dmeasure_on_xla():
    model = sde_model(dmeas=dmeas_at_runtime_index)
    model.set_backend("cute")
    with pytest.warns(UserWarning, match="dmeas"):
        s = model.to_struct()
    assert isinstance(s.rproc_pf, KernelRProc) and s.rproc_pf.with_dmeasure is None
    report = model.backend_report()
    assert report["rproc"].cute and report["gradients"].cute
    assert not report["dmeasure"].cute and not report["perturbation"].cute
    (problem,) = report["dmeasure"].problems
    assert "runtime index" in problem.what
    assert any("dmeas_at_runtime_index" in w for w in problem.where), problem


@jax.custom_jvp
def cumsum_tangent(x):
    return x


@cumsum_tangent.defjvp
def _cumsum_tangent_jvp(primals, tangents):
    # The kernel can't run cumsum, which only the derivative uses.
    return primals[0], jnp.cumsum(jnp.stack([tangents[0], 0 * tangents[0]]))[-1]


def cumsum_grad_rproc(X_, theta_, key, covars, t, dt):
    out = sde_rproc(X_, theta_, key, covars, t, dt)
    return {"x": cumsum_tangent(out["x"])}


def test_gradients_that_dont_translate_go_through_xla():
    from pypomp.cute.verify import _grads, _interval

    model = sde_model(cumsum_grad_rproc)
    model.set_backend("cute")
    with pytest.warns(UserWarning, match="differentiating through XLA"):
        s = model.to_struct()
    assert isinstance(s.rproc_pf, KernelRProc)
    report = model.backend_report()
    assert report["rproc"].cute and not report["gradients"].cute
    assert [p.what for p in report["gradients"].problems] == ["cumsum"]
    iv = _interval(model, "threefry")
    np.testing.assert_allclose(
        _grads(iv, s.rproc_per), _grads(iv, model.rproc.mechanics_per_interp), rtol=1e-5
    )
