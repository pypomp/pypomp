"""The checks and fallbacks around the CuTe backend that run on XLA, so need no GPU."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pypomp as pp
import pypomp.cute
import pypomp.functional as F
from pypomp.cute import CuteBackend, Kernels, _with_xla_grads
from pypomp.cute.verify import _check, _check_grads, _forward, _grads, _interval

from .test_cute_backend import (
    he10_model,
    particle_mean,
    sde_model,
    sde_rproc,
    table_model,
    table_panel,
    vectorized_rproc,
)


def xla_fns(model):
    return model.rproc.mechanics_pf_interp, model.rproc.mechanics_per_interp


def test_check_accepts_a_kernel_that_matches_xla():
    model = sde_model()
    assert _check(model, *xla_fns(model), False, "threefry") is None


def test_check_rejects_a_kernel_that_differs_from_xla():
    model = sde_model()
    pf, per = xla_fns(model)

    def off(X_, *args):
        out, t_idx = pf(X_, *args)
        return out * 1.01 + 0.1, t_idx

    assert "differ" in (_check(model, off, per, False, "threefry") or "")


def test_check_sees_the_steps_when_the_first_interval_has_no_length():
    def drifting(X_, theta_, key, covars, t, dt):
        out = sde_rproc(X_, theta_, key, covars, t, dt)
        return {"x": out["x"] + 5.0 * dt}

    times = np.arange(0.0, 20.0)
    model = sde_model(times=times)
    assert _interval(model, "threefry").obs == 1
    assert _check(model, *xla_fns(model), False, "threefry") is None
    wrong = xla_fns(sde_model(drifting, times=times))
    assert "differ" in (_check(model, *wrong, False, "threefry") or "")


def test_check_rejects_a_per_particle_kernel_that_differs_from_xla():
    model = sde_model()
    pf, per = xla_fns(model)

    def off(X_, *args):
        out, t_idx = per(X_, *args)
        return out * 1.01 + 0.1, t_idx

    assert "estimation scale" in (_check(model, pf, off, False, "threefry") or "")


def test_check_rejects_gradients_that_differ_from_xla():
    model = sde_model()
    pf, per = xla_fns(model)

    def off(X_, *args):  # the same particles, with a different derivative
        out, t_idx = per(X_, *args)
        return out + 0.5 * (X_ - jax.lax.stop_gradient(X_)), t_idx

    assert _check(model, pf, off, False, "threefry") is None
    assert "gradients differ" in (_check_grads(model, pf, off, "threefry") or "")


def test_check_grads_accepts_xlas_own_gradients():
    model = sde_model()
    assert _check_grads(model, *xla_fns(model), "threefry") is None


def test_check_grads_rejects_shared_parameter_gradients_that_differ():
    model = sde_model()
    pf, per = xla_fns(model)

    def off(X_, theta_, *args):  # the same particles, with a different derivative
        out, t_idx = pf(X_, theta_, *args)
        return out + 100.0 * jnp.sum(theta_ - jax.lax.stop_gradient(theta_)), t_idx

    assert _check(model, off, per, False, "threefry") is None
    why = _check_grads(model, off, per, "threefry") or ""
    assert "parameter gradient differs" in why and "one parameter vector" in why


def with_wrong_grads(f):
    """Return `f`, shifted by 1 undifferentiated and with a wrong derivative."""

    def g(X_, *args):
        out, *rest = f(X_, *args)
        return out + 1.0 + 0.5 * (X_ - jax.lax.stop_gradient(X_)), *rest

    return g


def test_xla_grads_keep_the_kernel_undifferentiated():
    model = sde_model()
    per = model.rproc.mechanics_per_interp
    kr = _with_xla_grads(Kernels(with_wrong_grads(per), per))
    iv = _interval(model, "threefry")
    np.testing.assert_allclose(
        _forward(iv, kr.rproc, iv.th_per, True)(iv.X0),
        _forward(iv, per, iv.th_per, True)(iv.X0) + 1.0,
        rtol=1e-6,
    )
    np.testing.assert_allclose(_grads(iv, kr.rproc), _grads(iv, per), rtol=1e-5)


def test_xla_grads_cover_the_fused_density():
    model = sde_model()
    assert model.dmeas is not None
    per, dmeas = model.rproc.mechanics_per_interp, model.dmeas.mechanics_per

    def fused(X_, theta_, keys, cov, *rest):
        out, t_end = per(X_, theta_, keys, cov, *rest[:6])
        trans, y, t_obs = rest[5:]
        c = None if cov is None else cov[t_end]
        return out, t_end, dmeas(y, out, theta_, c, t_obs, trans)

    kr = _with_xla_grads(
        Kernels(per, per, with_dmeasure=with_wrong_grads(fused), dmeasure=dmeas)
    )
    assert kr.with_dmeasure is not None
    iv = _interval(model, "threefry")
    y = jnp.asarray(np.asarray(model.ys, dtype=np.float32)[0])
    t_obs = jnp.float32(model.ys.index[0])

    def loss(f):
        def run(X, th):
            out, _, w = f(X, th, *iv.rest, True, y, t_obs)
            return jnp.sum(out) + jnp.sum(w)

        return jax.jit(jax.grad(run, (0, 1)))(iv.X0, iv.th_per)

    for a, b in zip(loss(kr.with_dmeasure), loss(fused), strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5)


def test_coupling_probe():
    coupled = sde_model(particle_mean)
    assert "other particles" in (
        _check(coupled, *xla_fns(coupled), True, "threefry") or ""
    )
    alone = sde_model(vectorized_rproc)
    assert _check(alone, *xla_fns(alone), True, "philox") is None


def test_backend_is_chosen_again_once_theta_is_set(monkeypatch):
    calls = []
    monkeypatch.setattr(
        pypomp.cute, "_build", lambda model, quiet, rng, check: calls.append(1)
    )
    model = sde_model()
    theta = model.theta
    model.theta = None
    model.to_struct()
    model.to_struct()
    assert len(calls) == 2
    model.theta = theta
    model.to_struct()
    model.to_struct()
    assert len(calls) == 3


def test_rng_must_be_threefry_or_philox():
    model = sde_model()
    with pytest.raises(ValueError, match="rng"):
        model.set_backend("cute", rng="rbg")
    with pytest.raises(ValueError, match="rng"):
        model.to_struct(rng="philox4x32")


def test_built_kernels_are_kept_per_generator(monkeypatch):
    calls = []
    monkeypatch.setattr(
        pypomp.cute, "_build", lambda model, quiet, rng, check: calls.append(rng)
    )
    model = sde_model()
    model.set_backend("cute")
    model.to_struct()
    model.to_struct(rng="philox")
    model.to_struct()
    model.set_backend("cute", rng="philox")
    model.to_struct()
    assert calls == ["threefry", "philox"]


def test_unchecked_kernels_need_no_theta_and_are_kept_apart(monkeypatch):
    calls = []
    monkeypatch.setattr(
        pypomp.cute, "_build", lambda model, quiet, rng, check: calls.append(check)
    )
    model = sde_model()
    theta = model.theta
    model.theta = None
    model.set_backend("cute", check=False)
    model.to_struct()
    model.to_struct()
    assert calls == [False]
    model.theta = theta
    model.set_backend("cute")
    model.to_struct()
    assert calls == [False, True]


def test_panel_simulates_and_probes_with_the_first_units_kernel(monkeypatch):
    built = []
    monkeypatch.setattr(
        pypomp.cute, "_build", lambda model, quiet, rng, check: built.append(model)
    )
    panel, _, _ = table_panel()
    panel.set_backend("cute")
    panel.simulate(nsim=2, key=jax.random.key(0))
    panel.probe(
        {"mean": lambda y: float(jnp.mean(y["y"]))}, nsim=2, key=jax.random.key(1)
    )
    assert len(built) == 1
    assert built[0] is panel.unit_objects["u1"]


def test_panel_simulation_keeps_each_units_data_and_parameters():
    panel, _, _ = table_panel(reverse_last=True)
    panel.set_backend("xla")
    key = jax.random.key(3)
    X, Y = panel.simulate(nsim=2, key=key)
    active, _ = panel._update_fresh_key(key)
    for unit, obj in panel.unit_objects.items():
        active, subkey = jax.random.split(active)
        theta = pp.PompParameters(panel.get_unit_parameters(unit))
        X_u, Y_u = obj.simulate(nsim=2, theta=theta, key=subkey)
        for got, want in ((X, X_u), (Y, Y_u)):
            got = got[got["unit"] == unit].drop(columns="unit")
            np.testing.assert_allclose(got.to_numpy(float), want.to_numpy(float))


def test_panel_takes_parameters_only_to_check_a_new_kernel(monkeypatch):
    monkeypatch.setattr(pypomp.cute, "_build", lambda model, quiet, rng, check: None)
    panel, _, _ = table_panel()
    panel.theta = pp.PanelParameters.merge(panel.theta, panel.theta)
    replicates = []
    get = pp.PanelPomp.get_unit_parameters

    def spy(self, unit, theta=None):
        replicates.append(self._prepare_theta_input(theta).num_replicates())
        return get(self, unit, theta)

    monkeypatch.setattr(pp.PanelPomp, "get_unit_parameters", spy)
    panel.set_backend("xla")
    panel.to_struct()
    panel.set_backend("cute", check=False)
    panel.to_struct()
    assert replicates == []
    panel.set_backend("cute")
    panel.to_struct()
    panel.to_struct()
    assert replicates == [1]
    assert all(u._theta is None for u in panel.unit_objects.values())


def test_translated_threefry_is_jaxs():
    pytest.importorskip("cutlass")
    from pypomp.cute import translate as ct

    key = jax.random.key(123, impl="threefry2x32")
    k0, k1 = (int(v) for v in jax.random.key_data(key))
    ctx = ct._Ctx(vectorized_J=None, particle=None, rng="threefry")
    bits = np.asarray(jax.random.bits(key, (2, 3), "uint32"))
    np.testing.assert_array_equal(ct._draw(ctx, k0, k1, (2, 3)).astype(np.uint32), bits)
    for i in (0, 1, 4999):
        derived = [int(v) for v in ct._key_derive("threefry", k0, k1, i)]
        split = jax.random.split(key, 5000)[i]
        assert derived == list(np.asarray(jax.random.key_data(split)))
        folded = jax.random.fold_in(key, i)
        assert derived == list(np.asarray(jax.random.key_data(folded)))


@pytest.mark.parametrize("make", [table_model, he10_model], ids=["tables", "he10"])
@pytest.mark.parametrize("per_particle", [False, True], ids=["pf", "per"])
def test_hoisted_step_computes_what_the_step_does(make, per_particle):
    pytest.importorskip("cutlass")
    from jax.extend.core import jaxpr_as_fun

    from pypomp.cute import translate as ct
    from pypomp.cute.analysis import _trace_step

    model = make()
    step_j = _trace_step(model)[0]
    shared = [False, not per_particle, False, True, True, True]
    h = ct._hoist(step_j, shared)
    assert h is not None
    ns, d = len(model.rproc.statenames), len(model.rproc.param_names)
    th = model.theta.to_jax_array(model.canonical_param_names)[0]
    cov = jnp.asarray(model._covars_extended)[7]
    args = [
        jnp.linspace(-1.0, 1.0, ns, dtype=jnp.float32) + 3.0,
        jnp.asarray(th, jnp.float32).reshape(d),
        jnp.array([5, 6], jnp.uint32),
        jnp.asarray(cov, jnp.float32),
        jnp.float32(7.3),
        jnp.float32(0.01),
    ]
    res = jaxpr_as_fun(h.shared)(*[a for a, s in zip(args, shared, strict=True) if s])
    want = jaxpr_as_fun(step_j)(*args)
    got = jaxpr_as_fun(h.rest)(*args, *res)
    for a, b in zip(want, got, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    if make is table_model:
        # The weekly table's reads depend only on the covariate and the time; the
        # grid's, at the state, stay in the kernel.
        assert len(ct._runtime_tables([h.rest])) < len(ct._runtime_tables([step_j]))


def test_hessian_optimizers_train_on_the_xla_rproc():
    model = sde_model()
    s = model.to_struct(backend="xla")

    def kernel(*args):
        raise AssertionError("Hessians can't go through the kernel")

    s = s._replace(
        backend=CuteBackend(Kernels(kernel, s.rproc_pf), Kernels(kernel, s.rproc_per))
    )
    th = model.theta.to_jax_array(model.canonical_param_names)
    eta = pp.LearningRate({p: 0.01 for p in model.canonical_param_names})
    keys = jax.random.split(jax.random.key(0), 1)
    _, traces = F.train(s, th, 32, 1, eta, keys, optimizer=pp.Newton())
    assert np.isfinite(np.asarray(traces)).all()


def with_xla_kernels(s, max_steps=None):
    """Return `s` with a CuTe backend whose kernels are XLA code computing what
    CuTe's compute, so iterated filtering takes the kernels' path on CPU."""
    pf = Kernels(s.rproc_pf, s.rproc_pf, max_steps=max_steps)
    return s._replace(backend=CuteBackend(pf, xla_iterated_filtering_kernels(s)))


def xla_iterated_filtering_kernels(s):
    """Kernels for `s.rproc_per` that are XLA code computing what CuTe's compute."""
    rproc, dmeas = s.rproc_per, s.dmeas_per
    assert dmeas is not None

    def with_dmeasure(X, th, keys, cov, dt, t, t_idx, nstep, acc, trans, y, t_obs):
        X, t_end = rproc(X, th, keys, cov, dt, t, t_idx, nstep, acc, trans)
        c = None if cov is None else cov[t_end]
        return X, t_end, dmeas(y, X, th, c, t_obs, trans)

    def perturb_step(
        X,
        th,
        anc,
        key_perturb,
        sigmas,
        keys,
        cov,
        dt,
        t,
        t_idx,
        nstep,
        acc,
        trans,
        y,
        t_obs,
        scratch,
    ):
        assert scratch[0].shape == X.shape and scratch[1].shape == th.shape
        th = th[anc] + sigmas * jax.random.normal(key_perturb, th.shape)
        iv = (keys, cov, dt, t, t_idx, nstep, acc, trans)
        X, t_end, w = with_dmeasure(X[anc], th, *iv, y, t_obs)
        return X, th, t_end, w

    return Kernels(
        rproc,
        rproc,
        with_dmeasure=with_dmeasure,
        dmeasure=dmeas,
        perturb_step=perturb_step,
    )


def test_engine_for_runs_kernels_only_on_their_own_density():
    from pypomp.core.algorithms.contexts import ModelFns
    from pypomp.core.algorithms.engine import XlaEngine
    from pypomp.cute.engine import CuteEngine

    s = sde_model().to_struct(backend="xla")
    assert type(ModelFns.per(s).engine) is XlaEngine
    kernels = with_xla_kernels(s)
    engine = ModelFns.per(kernels).engine
    assert isinstance(engine, CuteEngine) and engine._perturb_step is not None
    # A density the kernels don't evaluate runs separately, on XLA.
    dmeas = s.dmeas_per
    assert dmeas is not None
    other = kernels._replace(dmeas_per=lambda *a: dmeas(*a))
    engine = ModelFns.per(other).engine
    assert isinstance(engine, CuteEngine) and engine._perturb_step is None
    # A process model the kernels weren't built from runs on XLA.
    rproc = s.rproc_per
    other = kernels._replace(rproc_per=lambda *a: rproc(*a))
    assert type(ModelFns.per(other).engine) is XlaEngine
    assert isinstance(ModelFns.pf(other).engine, CuteEngine)


def test_kernels_reject_a_grid_with_more_steps_than_they_were_built_for():
    from pypomp.core.algorithms.contexts import ModelFns

    s = sde_model().to_struct(backend="xla")
    n = int(np.max(np.asarray(s.nstep_array)))
    ModelFns.pf(with_xla_kernels(s, max_steps=n))
    with pytest.raises(ValueError, match="built for at most"):
        ModelFns.pf(with_xla_kernels(s, max_steps=n - 1))


@pytest.mark.parametrize("n_obs, thresh", [(40, 0.0), (41, 1e3)])
def test_iterated_filtering_on_kernels_that_resample_on_load_matches_xla(n_obs, thresh):
    # The kernels' path defers each resampling to the next interval's load and
    # scans two intervals at a time; with XLA standing in for the kernels, it must
    # give XLA's results for the same keys.
    model = sde_model(times=np.arange(1, n_obs + 1, dtype=float))
    s = model.to_struct(backend="xla")
    kernels = with_xla_kernels(s)
    th = model.theta.to_jax_array(model.canonical_param_names)
    keys = jax.random.split(jax.random.key(3), th.shape[0])
    rw = pp.RWSigma(
        sigmas={"r": 0.02, "sig": 0.02, "tau": 0.02, "z0m": 0.0, "z0s": 0.0},
        init_names=[],
    ).geometric_cooling(0.5)
    want = F.mif(s, th, 64, 2, rw, keys, thresh=thresh)
    got = F.mif(kernels, th, 64, 2, rw, keys, thresh=thresh)
    for a, b in zip(want, got, strict=True):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-6)
