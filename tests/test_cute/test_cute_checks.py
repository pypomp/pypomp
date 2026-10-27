"""The checks and fallbacks around the CuTe backend that run on XLA, so need no GPU."""

import jax
import numpy as np
import pytest

import pypomp as pp
import pypomp.cute
import pypomp.functional as F
from pypomp.cute import KernelRProc
from pypomp.cute.verify import _check

from .test_cute_backend import particle_mean, sde_model, vectorized_rproc


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

    assert "gradients differ" in (_check(model, pf, off, False, "threefry") or "")


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
        model.set_rproc_backend("cute", rng="rbg")
    with pytest.raises(ValueError, match="rng"):
        model.to_struct(rng="philox4x32")


def test_built_kernels_are_kept_per_generator(monkeypatch):
    calls = []
    monkeypatch.setattr(
        pypomp.cute, "_build", lambda model, quiet, rng, check: calls.append(rng)
    )
    model = sde_model()
    model.set_rproc_backend("cute")
    model.to_struct()
    model.to_struct(rng="philox")
    model.to_struct()
    model.set_rproc_backend("cute", rng="philox")
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
    model.set_rproc_backend("cute", check=False)
    model.to_struct()
    model.to_struct()
    assert calls == [False]
    model.theta = theta
    model.set_rproc_backend("cute")
    model.to_struct()
    assert calls == [False, True]


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


def test_hessian_optimizers_train_on_the_xla_rproc():
    model = sde_model()
    s = model.to_struct(rproc_backend="xla")

    def kernel(*args):
        raise AssertionError("Hessians can't go through the kernel")

    s = s._replace(
        rproc_pf=KernelRProc(kernel, s.rproc_pf),
        rproc_per=KernelRProc(kernel, s.rproc_per),
    )
    th = model.theta.to_jax_array(model.canonical_param_names)
    eta = pp.LearningRate({p: 0.01 for p in model.canonical_param_names})
    keys = jax.random.split(jax.random.key(0), 1)
    _, traces = F.train(s, th, 32, 1, eta, keys, optimizer=pp.Newton())
    assert np.isfinite(np.asarray(traces)).all()
