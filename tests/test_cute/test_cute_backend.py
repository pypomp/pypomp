"""The CuTe backend. Needs nvidia-cutlass-dsl and a GPU; skipped otherwise."""

import os
import re
import stat
import sys
import warnings
from copy import deepcopy
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import pypomp as pp
import pypomp.functional as F
from pypomp.core.algorithms.engine import Interval, engine_for
from pypomp.cute import CuteBackend, Kernels, available
from pypomp.random import binominv
from pypomp.random.philox import PHILOX

ok, why = available()
pytestmark = pytest.mark.skipif(not ok, reason=f"CuTe backend unavailable: {why}")


def check_interval(model, rng="threefry"):
    """Return the checks' test interval for `model` at its first parameter set."""
    from pypomp.cute.verify import _interval

    th = model.theta.to_jax_array(model.canonical_param_names)[0]
    return _interval(model, th, rng)


def sde_rproc(X_, theta_, key, covars, t, dt):
    z = X_["x"]
    eps = jax.random.normal(key)
    drift = theta_["r"] * (1 - jnp.exp(z)) - theta_["sig"] ** 2 / 2
    return {"x": z + drift * dt + theta_["sig"] * jnp.sqrt(dt) * eps}


def scan_rproc(X_, theta_, key, covars, t, dt):
    # Horner's rule as a lax.scan with a constant, a carry and stacked inputs, as
    # jnp.polyval was before JAX 0.12 (it is now a lax.polynomial).
    z = X_["x"]
    eps = jax.random.normal(key)
    coefs = jnp.array([1 / 24, 1 / 6, 1 / 2, 1.0, 0.0])
    expm1, _ = jax.lax.scan(lambda c, a: (c * z + a, c), jnp.float32(0.0), coefs)
    drift = -theta_["r"] * expm1 - theta_["sig"] ** 2 / 2
    return {"x": z + drift * dt + theta_["sig"] * jnp.sqrt(dt) * eps}


def control_rproc(X_, theta_, key, covars, t, dt):
    # Loops with static trip counts, a switch on a constant, a select among three
    # cases at a runtime index, derived keys, an iota, a constant-index update and a
    # draw of two axes.
    z = X_["x"]
    _, w = jax.lax.while_loop(
        lambda c: c[0] < 3, lambda c: (c[0] + 1, 0.9 * c[1] + 0.1 * z), (0, z)
    )
    f = jax.lax.fori_loop(0, 3, lambda i, c: 0.5 * c + 0.1 * i * z, z)
    s = jax.lax.switch(1, [lambda: z, lambda: 2.0 * z, lambda: -z])
    i = jnp.clip(jnp.floor((z + 2.0) * 2.0), 0, 2).astype(jnp.int32)
    pick = jax.lax.select_n(i, z, 0.5 * z, -0.5 * z)
    k1, k2 = jax.random.split(jax.random.fold_in(key, 7))
    v = jax.lax.dynamic_update_slice(jnp.arange(3.0) * z, s[None], (1,))
    m = jax.random.normal(k2, (2, 2))
    nudge = 0.1 * (w + f + v.sum() + pick) + 0.01 * (jax.random.normal(k1) + m.sum())
    return sde_rproc({"x": z + nudge * dt}, theta_, key, covars, t, dt)


@pp.vectorized
def particlewise_rproc(X_, theta_, key, covars, t, dt):
    # Shape operations that keep each particle's values in its own column, and draws
    # with particles on their last axis, from one key and from a key per particle.
    z = X_["x"]
    J = z.shape[0]
    m = jnp.pad(jnp.stack([z, 2 * z]), ((0, 1), (0, 0)))
    m = jnp.concatenate([m, z[None]], 0)
    v = m.reshape(2, 2, J)[1, 0]
    w = jax.lax.dynamic_slice(m, (1, 0), (2, J)).sum(0)
    u = jax.lax.dynamic_update_slice(m, w[None], (0, 0))[0]
    eps = jax.random.normal(key, (2, J))[1]
    keys = jax.random.split(jax.random.fold_in(key, 1), J)
    xi = jax.vmap(jax.random.normal)(keys)
    drift = theta_["r"] * (1 - jnp.exp(z)) - theta_["sig"] ** 2 / 2
    noise = theta_["sig"] * jnp.sqrt(dt) * (0.8 * eps + 0.6 * xi)
    return {"x": z + (drift + 0.01 * (v + w + u)) * dt + noise}


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


@jax.jit
def _lookup(table, k, z):
    v = table[k]  # long enough to be outlined
    for _ in range(30):
        v = jnp.sin(v) * 0.9 + 0.01 * z
    return v


def table_arg_rproc(X_, theta_, key, covars, t, dt):
    k = jnp.clip(jnp.floor(t), 0, 59).astype(jnp.int32)
    z = X_["x"]
    v = _lookup(jnp.asarray(_WEEKLY[0]), k, z)
    return sde_rproc({"x": z + 0.1 * v * dt}, theta_, key, covars, t, dt)


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


def sde_rmeas(X_, theta_, key, covars, t):
    return {"y": X_["x"] + theta_["tau"] * jax.random.normal(key)}


def sde_model(
    rproc=sde_rproc,
    statenames=("x",),
    covars=None,
    dmeas=sde_dmeas,
    times=None,
    dt=None,
    reverse_params=False,
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
    if reverse_params:
        theta = dict(reversed(theta.items()))
    model = pp.Pomp(
        ys=ys,
        theta=pp.PompParameters(theta),
        statenames=list(statenames),
        t0=0.0,
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        rmeas=sde_rmeas,
        nstep=None if dt else 10,
        dt=dt,
        covars=covars,
    )
    model.set_backend("auto")
    return model


def uneven_model(rproc=sde_rproc):
    """sde_model observed at uneven times, so its intervals take 4, 2, 8 and 3
    Euler steps in turn."""
    return sde_model(rproc, times=np.cumsum(np.resize([1, 0.5, 2, 0.75], 40)), dt=0.25)


def table_model(u=2, reverse_params=False):
    covars = pd.DataFrame({"u": float(u)}, index=np.arange(0.0, 41.0))
    return sde_model(table_rproc, covars=covars, reverse_params=reverse_params)


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
    model = pp.models.UKMeasles.pomp(
        unit="London", theta=pp.PompParameters(theta), clean=True, model="001b"
    )
    model.set_backend("auto")
    return model


def kernels(struct, per_particle=False) -> Kernels:
    """Return `struct`'s CuTe kernels of one flavor, failing if it has none."""
    assert isinstance(struct.backend, CuteBackend)
    return struct.backend.per if per_particle else struct.backend.pf


def propagate(struct):
    """Return `struct`'s ``rproc_pf`` as ``pfilter`` runs it: on its kernel, if
    it has one."""
    engine = engine_for(struct, per_particle=False)
    return lambda X, theta, *rest: engine.propagate(X, theta, Interval(*rest))


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
        return propagate(struct)(
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
        lambda: sde_model(scan_rproc),
        lambda: sde_model(control_rproc),
        lambda: sde_model(particlewise_rproc),
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
        "scan",
        "control",
        "particlewise",
    ],
)
@pytest.mark.parametrize("rng", ["threefry", "philox"])
def test_cute_matches_xla_with_the_same_random_numbers(make, rng):
    model = make()
    xla, cute = xla_and_cute(model, rng)
    assert isinstance(cute.backend, CuteBackend)
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


@pytest.mark.skipif(len(jax.devices()) < 2, reason="needs two devices")
def test_each_device_launches_the_kernel_for_its_own_filters(monkeypatch):
    from pypomp.cute import kernel

    rows = []
    call = kernel.cutlass_call

    def spy(*a, **k):
        f = call(*a, **k)
        return lambda *args: rows.append(args[0].shape[0]) or f(*args)

    monkeypatch.setattr(kernel, "cutlass_call", spy)
    model = sde_model()
    theta = pp.PompParameters([model.theta[0]] * 4)
    J = 256
    lls = {}
    for backend in ("xla", "cute"):
        m = deepcopy(model)
        m.set_backend(backend, check=False)
        m.pfilter(J=J, theta=theta, key=jax.random.key(0))
        lls[backend] = m.results()["logLik"].to_numpy(dtype=float)
    # Two of the four parameter sets per device, rather than all four on each. (The
    # calls of J rows trace one filter, for custom_vmap's output types.)
    assert 2 * J in rows and max(rows) == 2 * J
    # Float rounding can flip a resampling decision, which moves the log-likelihood
    # by about its Monte Carlo error.
    np.testing.assert_allclose(lls["cute"], lls["xla"], atol=1.0)


def _spy_launch_rows(monkeypatch):
    """Record the particle rows of every kernel launch."""
    from pypomp.cute import kernel

    rows = []
    call = kernel.cutlass_call

    def spy(*a, **k):
        f = call(*a, **k)
        return lambda *args: rows.append(args[0].shape[0]) or f(*args)

    monkeypatch.setattr(kernel, "cutlass_call", spy)
    return rows


@pytest.mark.parametrize("panel", [False, True], ids=["pomp", "panel"])
def test_particle_batches_run_the_kernel(monkeypatch, panel):
    """Filters run a few at a time give what they give all at once."""
    module = sys.modules["pypomp.functional.pfilter"]
    J, reps = 256, 6
    if panel:
        model, _, _ = table_panel()
        model.set_backend("cute", check=False)

        def run(key):
            model.pfilter(J=J, key=key, reps=reps)
            return model.results()["shared logLik"].to_numpy(dtype=float)

    else:
        model = sde_model()
        model.set_backend("cute", check=False)
        theta = pp.PompParameters([model.theta[0]] * 3)

        def run(key):
            model.pfilter(J=J, theta=theta, key=key, reps=reps)
            return model.results()["logLik"].to_numpy(dtype=float)

    rows = _spy_launch_rows(monkeypatch)
    flat = run(jax.random.key(0))
    assert rows, "the kernel never ran"
    rows.clear()
    monkeypatch.setattr(module, "_PARTICLES_PER_BATCH", 2 * J)
    batched = run(jax.random.key(0))
    # At most two filters per launch on each device, rather than all of them.
    assert rows and max(rows) <= 2 * J
    np.testing.assert_allclose(batched, flat, rtol=1e-5)


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
            return propagate(s)(
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
    assert isinstance(cute.backend, CuteBackend)
    X0, th, _ = interval(model, xla, 512, 3)
    keys = jax.random.split(rng_key(5), 3)
    nsteps = np.asarray(xla.nstep_array)
    i_obs = np.array([1, 2, 3])  # 2, 8 and 3 Euler steps
    ts = jnp.asarray(np.asarray(xla.times)[i_obs - 1])
    t_idx = jnp.asarray(np.cumsum(nsteps)[i_obs - 1])
    n = jnp.asarray(nsteps[i_obs])

    def run(s):
        def f(X, k, t, t_i, n):
            return propagate(s)(
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


def test_tables_passed_to_long_calls_are_read():
    model = sde_model(table_arg_rproc)
    model.set_backend("cute")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = model.to_struct()
    assert isinstance(s.backend, CuteBackend)


def test_grids_with_longer_intervals_than_the_kernel_are_rejected():
    model = uneven_model()
    model.set_backend("cute")
    s = model.to_struct()
    assert isinstance(s.backend, CuteBackend)
    nstep = np.asarray(s.nstep_array).copy()
    nstep[1] += 20
    longer = s._replace(nstep_array=jnp.asarray(nstep))
    th = model.theta.to_jax_array(model.canonical_param_names)
    with pytest.raises(ValueError, match="built for at most 8"):
        F.pfilter(longer, th, 100, jax.random.split(jax.random.key(0), 1)[:, None])


def test_a_grid_without_steps_stays_on_xla():
    model = sde_model(times=np.array([0.0]))
    model.set_backend("cute", check=False)
    with pytest.warns(UserWarning, match="no process steps"):
        s = model.to_struct()
    assert s.backend is None
    th = model.theta.to_jax_array(model.canonical_param_names)
    assert np.isfinite(
        F.pfilter(s, th, 100, jax.random.split(jax.random.key(0), 1)[:, None])["logLik"]
    ).all()


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


def table_panel(reverse_last=False):
    units = {"u1": table_model(1), "u3": table_model(3, reverse_last)}
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
    assert isinstance(cute.backend, CuteBackend)
    assert all(u._theta is None for u in panel.unit_objects.values())
    xla = panel.to_struct(backend="xla")
    assert xla.backend is None

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
            return propagate(s)(
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
        assert isinstance(model.to_struct().backend, CuteBackend)
    rw = pp.RWSigma(
        sigmas={"r": 0.02, "sig": 0.02, "tau": 0.02, "z0m": 0.0, "z0s": 0.0},
        init_names=[],
    ).geometric_cooling(0.5)
    model.mif(J=256, M=3, rw_sd=rw, key=jax.random.key(0))
    assert np.isfinite(model.results()["logLik"].to_numpy(dtype=float)).all()


SDE_RW = pp.RWSigma(
    sigmas={"r": 0.02, "sig": 0.02, "tau": 0.02, "z0m": 0.0, "z0s": 0.0},
    init_names=[],
).geometric_cooling(0.5)


def test_mif_on_cute_agrees_with_xla_in_distribution():
    # With the same key, a resampling decision that rounding flips sends the two
    # backends' searches their own ways, so they are compared over replicates.
    R = 32
    out = {}
    for backend, seed in (("xla", 11), ("cute", 12)):
        model = sde_model()
        model.set_backend(backend)
        model.theta = pp.PompParameters([model.theta[0]] * R)
        if backend == "cute":
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                per = kernels(model.to_struct(), per_particle=True)
            assert per.perturb_step is not None
        model.mif(J=256, M=2, rw_sd=SDE_RW, key=jax.random.key(seed))
        out[backend] = model.results()
    for col in ("logLik", "r", "sig", "tau"):
        a, b = (out[k][col].to_numpy(dtype=float) for k in ("xla", "cute"))
        se = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / R)
        assert abs(b.mean() - a.mean()) < 4 * se, col


@pytest.mark.parametrize("block", [True, False], ids=["mpif", "pif"])
def test_panel_mif_on_cute(block):
    panel, names, th = table_panel()
    panel.set_backend("cute")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        per = kernels(panel.to_struct(), per_particle=True)
        assert per.perturb_step is not None
        panel.mif(J=256, M=2, rw_sd=SDE_RW, key=jax.random.key(0), block=block)
    res = panel.results()
    assert np.isfinite(res["shared logLik"].to_numpy(dtype=float)).all()
    assert np.isfinite(res["unit logLik"].to_numpy(dtype=float)).all()


def test_simulate_on_cute_matches_xla():
    # The model accumulates case counts, which the kernel needs as static indices.
    model = he10_model()
    assert model.rproc.accumvars
    out = {}
    for backend in ("xla", "cute"):
        m = deepcopy(model)
        m.set_backend(backend)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            X, _ = m.simulate(nsim=20, key=jax.random.key(7))
        out[backend] = X.drop(columns=["theta_idx", "sim", "time"]).to_numpy(float)
        if backend == "cute":
            assert isinstance(m.to_struct().backend, CuteBackend)
    rel = np.abs(out["xla"] - out["cute"]) / (np.abs(out["xla"]) + 1.0)
    # A binomial draw that rounding flips sends that simulation its own way.
    assert (rel.max(-1) > 1e-3).mean() < 0.1
    assert np.median(rel) < 1e-5


def test_panel_simulates_on_one_kernel_as_xla_does():
    panel, _, _ = table_panel(reverse_last=True)
    panel.set_backend("xla")
    cp = deepcopy(panel)
    cp.set_backend("cute")
    key = jax.random.key(7)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        X_c, _ = cp.simulate(nsim=20, key=key)
    assert isinstance(cp.to_struct().backend, CuteBackend)
    assert "_cute_fns" not in cp.unit_objects["u3"].__dict__
    X_x, _ = panel.simulate(nsim=20, key=key)
    a, b = X_x["x"].to_numpy(), X_c["x"].to_numpy()
    assert not np.allclose(a[X_x["unit"] == "u1"], a[X_x["unit"] == "u3"])
    rel = np.abs(a - b) / (np.abs(a) + 1.0)
    assert np.median(rel) < 1e-5
    assert rel.max() < 1e-2


def nan_ops_rproc(X_, theta_, key, covars, t, dt):
    n = jnp.sqrt(jax.random.normal(key))  # NaN for about half of the particles
    return {
        "x": jnp.maximum(n, 0.5),
        "a": jnp.minimum(0.5, n),
        "b": jax.lax.clamp(0.2, n, 0.8),
        "c": jnp.max(jnp.stack([0.5, n])),
        "d": jnp.min(jnp.stack([n, 0.5])),
    }


def unchecked_cute(model):
    """`model`'s CuTe struct, which keeps a kernel that differs from XLA."""
    cm = deepcopy(model)
    cm.set_backend("cute", check=False)
    return cm.to_struct()


def test_max_min_and_clamp_keep_nans_as_xla_does():
    model = sde_model(nan_ops_rproc, statenames=tuple("xabcd"))
    xla, cute = model.to_struct(backend="xla"), unchecked_cute(model)
    assert isinstance(cute.backend, CuteBackend)
    X0, th, adv_xla = interval(model, xla, 1024, 1)
    _, _, adv_cute = interval(model, cute, 1024, 1)
    key = rng_key(5)
    a = np.asarray(jax.jit(lambda X: adv_xla(X, th, key))(X0[0]))
    b = np.asarray(jax.jit(lambda X: adv_cute(X, th, key))(X0[0]))
    assert 0.3 < np.isnan(a).mean(0).min() and np.isnan(a).mean(0).max() < 0.7
    np.testing.assert_allclose(a, b, rtol=1e-5, equal_nan=True)


def erf_inv_rproc(X_, theta_, key, covars, t, dt):
    # Every fourth particle at -1, 1 or just inside them, and the rest uniform.
    u = jax.random.uniform(key, minval=-1.0, maxval=1.0)
    k = jnp.floor(jax.random.uniform(key, minval=0.0, maxval=16.0))
    edge = jnp.array([-1.0, 1.0, -0.99999994, 0.99999994])[jnp.int32(k) % 4]
    return {"x": jax.scipy.special.erfinv(jnp.where(k < 4, edge, u))}


def test_erf_inv_is_infinite_at_plus_and_minus_one_as_in_xla():
    model = sde_model(erf_inv_rproc)
    xla, cute = model.to_struct(backend="xla"), unchecked_cute(model)
    assert isinstance(cute.backend, CuteBackend)
    X0, th, adv_xla = interval(model, xla, 1024, 1)
    _, _, adv_cute = interval(model, cute, 1024, 1)
    key = rng_key(5)
    a = np.asarray(jax.jit(lambda X: adv_xla(X, th, key))(X0[0]))
    b = np.asarray(jax.jit(lambda X: adv_cute(X, th, key))(X0[0]))
    assert np.isposinf(a).any() and np.isneginf(a).any()
    np.testing.assert_allclose(a, b, rtol=1e-5)


def finite_select_rproc(X_, theta_, key, covars, t, dt):
    z = X_["x"]
    # logsumexp selects on isfinite of the maximum; so does the where below.
    v = jnp.stack([z, 2 * z, jnp.log(z + 1.5)])  # NaN where z < -1.5
    w = jnp.where(jnp.isfinite(v), v, -1.0)
    return sde_rproc(
        {"x": z + 0.01 * jax.nn.logsumexp(w) * dt}, theta_, key, covars, t, dt
    )


def test_selects_on_isfinite_match_xla():
    model = sde_model(finite_select_rproc)
    xla, cute = model.to_struct(backend="xla"), unchecked_cute(model)
    assert isinstance(cute.backend, CuteBackend)
    X0, th, adv_xla = interval(model, xla, 1024, 1)
    _, _, adv_cute = interval(model, cute, 1024, 1)
    key = rng_key(5)
    a = np.asarray(jax.jit(lambda X: adv_xla(X, th, key))(X0[0]))
    b = np.asarray(jax.jit(lambda X: adv_cute(X, th, key))(X0[0]))
    np.testing.assert_allclose(a, b, rtol=1e-5)


def int_clamp_rproc(X_, theta_, key, covars, t, dt):
    n = jnp.round(10 * X_["x"]).astype(jnp.int32)
    # Bounds that cross for some particles, where XLA returns the upper one.
    lo = jnp.round(10 * jax.random.normal(key)).astype(jnp.int32)
    c = jax.lax.clamp(lo, n, jnp.int32(-12))
    return sde_rproc({"x": X_["x"] + 0.01 * c * dt}, theta_, key, covars, t, dt)


def test_integer_clamp_with_crossed_bounds_matches_xla():
    model = sde_model(int_clamp_rproc)
    xla, cute = model.to_struct(backend="xla"), unchecked_cute(model)
    assert isinstance(cute.backend, CuteBackend)
    X0, th, adv_xla = interval(model, xla, 1024, 1)
    _, _, adv_cute = interval(model, cute, 1024, 1)
    key = rng_key(5)
    a = np.asarray(jax.jit(lambda X: adv_xla(X, th, key))(X0[0]))
    b = np.asarray(jax.jit(lambda X: adv_cute(X, th, key))(X0[0]))
    np.testing.assert_allclose(a, b, rtol=1e-5)


def test_estimation_checks_the_kernel_at_the_parameters_it_is_given():
    model = sde_model()
    model.set_backend("cute")
    theta = model.theta
    model.theta = None  # as for a Pomp that a PanelPomp has taken as a unit
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.pfilter(J=100, theta=theta, key=jax.random.key(0))
    assert model.backend_report()["rproc"].cute


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
    assert s.backend is None


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
    assert isinstance(s.backend, CuteBackend)
    assert "not checked against XLA (check=False)" in str(model.backend_report())


@pytest.mark.parametrize(
    "check, part",
    [("_check_dmeas", "dmeasure"), ("_check_perturb_step", "perturbation")],
    ids=["dmeasure", "perturbation"],
)
def test_fused_steps_whose_check_fails_run_separately(monkeypatch, check, part):
    import pypomp.cute.verify as verify

    def fail(*args):
        raise RuntimeError("no check")

    monkeypatch.setattr(verify, check, fail)
    model = sde_model()
    model.set_backend("cute")
    with pytest.warns(UserWarning, match=r"failed its check \(RuntimeError"):
        s = model.to_struct()
    report = model.backend_report()
    assert report["rproc"].cute and not report[part].cute
    assert "RuntimeError: no check" in report[part].note
    per = kernels(s, per_particle=True)
    assert per.perturb_step is None
    if part == "dmeasure":
        assert per.with_dmeasure is None and kernels(s).with_dmeasure is None
        assert report["perturbation"].note == "needs dmeasure in the kernel"
    else:
        assert per.with_dmeasure is not None


def test_check_perturb_step_rejects_a_kernel_that_differs_from_xla():
    from pypomp.cute.verify import _check_perturb_step

    model = sde_model()
    model.set_backend("cute")
    per = kernels(model.to_struct(), per_particle=True)
    step = per.perturb_step
    assert step is not None
    iv = check_interval(model)
    assert _check_perturb_step(model, iv, per, "threefry") is None
    whys = {
        0: "of particles differ from XLA when resampled on load",
        1: "its parameter perturbations differ from XLA's",
        3: "of measurement densities differ when resampled on load",
    }
    for i, why in whys.items():

        def shifted(*args, i=i, **kwargs):
            out = list(step(*args, **kwargs))
            out[i] = out[i] + 1.0
            return tuple(out)

        wrong = replace(per, perturb_step=shifted)
        assert why in (_check_perturb_step(model, iv, wrong, "threefry") or "")


def test_kernel_whose_gradients_differ_is_differentiated_on_xla(monkeypatch):
    import pypomp.cute.verify as verify
    from pypomp.cute.verify import _grads

    monkeypatch.setattr(verify, "_check_grads", lambda *args: "50% differ")
    model = sde_model()
    model.set_backend("cute")
    with pytest.warns(UserWarning, match="differentiating through XLA"):
        s = model.to_struct()
    assert isinstance(s.backend, CuteBackend)
    iv = check_interval(model)
    np.testing.assert_allclose(
        _grads(iv, kernels(s, per_particle=True).rproc),
        _grads(iv, model.rproc.mechanics_per_interp),
        rtol=1e-5,
    )


def test_gradients_that_cancel_to_rounding_pass_the_check():
    from pypomp.models import spx

    model = spx()  # mu's gradient through S cancels to float32 rounding
    model.set_backend("cute")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = model.to_struct()
    assert isinstance(s.backend, CuteBackend)


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
        assert s.backend is None
    finally:
        jax.config.update("jax_default_prng_impl", "threefry2x32")


def test_float64_particles_run_on_xla():
    jax.config.update("jax_enable_x64", True)
    try:
        model = sde_model()
        model.pfilter(J=100, key=jax.random.key(0))
        assert np.isfinite(model.results()["logLik"].to_numpy(dtype=float)).all()
        assert "jax_enable_x64" in model.backend_report()["rproc"].note
    finally:
        jax.config.update("jax_enable_x64", False)


def test_auto_uses_cute_with_one_step_per_observation():
    one_step = pp.models.lg(T=10)
    one_step.set_backend("auto")
    assert one_step.rproc.nstep == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = one_step.to_struct()
    assert isinstance(s.backend, CuteBackend)


def test_gpus_older_than_ampere_are_unavailable(monkeypatch):
    class Volta:
        platform = "gpu"
        compute_capability = "7.0"

    monkeypatch.setattr(jax, "devices", lambda backend=None: [Volta()])
    ok, why = available()
    assert not ok and "7.0" in why


def test_gpus_of_unknown_compute_capability_are_unavailable(monkeypatch):
    class Rocm:
        platform = "gpu"
        compute_capability = "gfx90a"

    monkeypatch.setattr(jax, "devices", lambda backend=None: [Rocm()])
    ok, why = available()
    assert not ok and "gfx90a" in why


def test_kernels_that_dont_import_are_unavailable(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "pypomp.cute.kernel", None)
    ok, why = available()
    assert not ok and "don't import" in why


def test_no_gpu_means_unavailable(monkeypatch):
    monkeypatch.setattr(jax, "devices", lambda backend=None: [])
    assert available() == (False, "no GPU device")


def test_cutlass_without_its_jax_integration_is_unavailable(monkeypatch):
    import cutlass.jax as cj  # pyright: ignore[reportMissingImports]

    monkeypatch.setattr(cj, "is_available", lambda: False)
    ok, why = available()
    assert not ok and "JAX integration unavailable" in why


def test_resampling_indices_carry_no_derivative():
    from pypomp.core.algorithms.helpers import _resample
    from pypomp.cute.resample import resample

    w = jax.nn.softmax(jnp.linspace(-1.0, 1.0, 256, dtype=jnp.float32))
    key = jax.random.key(0)
    idx, tangent = jax.jvp(lambda w: resample(w, key), (w,), (jnp.ones_like(w),))
    assert tangent.dtype == jax.dtypes.float0
    np.testing.assert_array_equal(np.asarray(idx), np.asarray(_resample(w, key)))


def test_builds_without_a_cpu_backend(monkeypatch):
    devices = jax.devices

    def no_cpu(backend=None):
        if backend == "cpu":
            raise RuntimeError("Unknown backend cpu. Available backends are ['cuda']")
        return devices(backend)

    monkeypatch.setattr(jax, "devices", no_cpu)
    model = sde_model()
    assert isinstance(model.to_struct().backend, CuteBackend)


def test_kernel_that_differs_from_xla_is_not_used(monkeypatch):
    from pypomp.cute import kernel

    model = sde_model()
    make = kernel._make_kernels

    def off(model, traced, per_particle, rng, dm_j):
        k = make(model, traced, per_particle, rng, dm_j)
        rproc = k.rproc
        return replace(
            k, rproc=lambda X_, *a: (rproc(X_, *a)[0] + 0.1, rproc(X_, *a)[1])
        )

    monkeypatch.setattr(kernel, "_make_kernels", off)
    with pytest.warns(UserWarning, match="differ"):
        s = model.to_struct()
    assert s.backend is None


@pytest.mark.parametrize("dt", [None, 0.1], ids=["nstep", "dt"])
def test_kernel_leaves_intervals_of_no_length_alone(dt):
    # Observed at t0, so the first interval is empty and the rest take 10 steps.
    model = sde_model(times=np.arange(0.0, 40.0), dt=dt)
    assert model.rproc.skip_empty
    xla, cute = xla_and_cute(model)
    assert isinstance(cute.backend, CuteBackend)
    X0, th, adv_xla = interval(model, xla, 512, 2, i_obs=1)
    _, _, adv_cute = interval(model, cute, 512, 2, i_obs=1)
    keys = jax.random.split(rng_key(5), 2)
    a = np.asarray(jax.jit(jax.vmap(lambda X, k: adv_xla(X, th, k)))(X0, keys))
    b = np.asarray(jax.jit(jax.vmap(lambda X, k: adv_cute(X, th, k)))(X0, keys))
    assert not np.allclose(a, X0)
    rel = np.abs(a - b) / (np.abs(a) + 1.0)
    assert (rel.max(-1) > 1e-4).mean() < 0.005

    def empty(s):
        def f(X, k):
            args = (s.covars_extended, s.dt_array_extended, 0.0, 0, 0, s.accumvars)
            return propagate(s)(X, th, k, *args, False)

        return jax.jit(jax.vmap(f))(X0, keys)

    for s in (xla, cute):
        X, t_idx = empty(s)
        np.testing.assert_array_equal(np.asarray(X), np.asarray(X0))
        assert (np.asarray(t_idx) == 0).all()
    assert model.backend_report(backend="cute")["rproc"].cute


def test_report_says_every_part_runs_on_cute():
    model = sde_model()
    model.set_backend("cute")
    report = model.backend_report()
    assert report.available and not report.reason
    assert all(p.cute for p in report.parts), str(report)
    text = str(report)
    assert "  rproc         CuTe" in text
    assert re.search(r"^  \w+ +XLA ", text, re.MULTILINE) is None, text
    assert "float64 particles or keys of another generator" in text


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
    assert kernels(s).with_dmeasure is None
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
    from pypomp.cute.verify import _grads

    model = sde_model(cumsum_grad_rproc)
    model.set_backend("cute")
    with pytest.warns(UserWarning, match="differentiating through XLA"):
        s = model.to_struct()
    assert isinstance(s.backend, CuteBackend)
    report = model.backend_report()
    assert report["rproc"].cute and not report["gradients"].cute
    assert [p.what for p in report["gradients"].problems] == ["cumsum"]
    iv = check_interval(model)
    np.testing.assert_allclose(
        _grads(iv, kernels(s, per_particle=True).rproc),
        _grads(iv, model.rproc.mechanics_per_interp),
        rtol=1e-5,
    )
