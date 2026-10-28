"""Pomp.backend_report where CuTe can't run, as on a machine without a GPU."""

import jax
import jax.numpy as jnp
import pytest

import pypomp as pp
import pypomp.cute

from .test_cute_backend import normal_dmeas, sde_model, uneven_model


@pytest.fixture(autouse=True)
def no_cute(monkeypatch):
    monkeypatch.setattr(pypomp.cute, "available", lambda: (False, "no GPU here"))


def sorts(X_, theta_, key, covars, t, dt):
    z = X_["x"]
    return {"x": z + jnp.sort(jnp.stack([z, -z]))[0] * dt}


def dmeas_at_runtime_index(Y_, X_, theta_, covars, t):
    v = jnp.stack([X_["x"], 2 * X_["x"]])
    i = jnp.asarray(X_["x"] > 0, jnp.int32)
    return normal_dmeas(Y_, {"x": v[i]}, theta_, covars, t)


@pp.vectorized
def particle_mean(X_, theta_, key, covars, t, dt):
    z = X_["x"]
    return {"x": z + jnp.mean(z) * dt + 0.1 * jax.random.normal(key, z.shape)}


def test_a_model_that_translates():
    report = sde_model().backend_report()
    assert not report.available and "no GPU here" in report.reason
    assert report.translates
    assert all(p.translates for p in report.parts if p.translates is not None)
    assert not any(p.cute for p in report.parts)
    text = str(report)
    assert "no GPU here" in text and "doesn't translate" not in text


def test_intervals_of_different_lengths_translate():
    model = uneven_model()
    assert model.rproc.nstep is None
    assert model.backend_report().translates


def test_rproc_problems_point_at_the_model_line():
    report = sde_model(sorts).backend_report()
    assert not report.translates
    (problem,) = report["rproc"].problems
    assert problem.what == "sort"
    assert any("test_cute_report.py" in w and "(sorts)" in w for w in problem.where)
    # Without the process model on the kernel, its gradients don't apply.
    assert report["gradients"].translates is None
    assert "sort" in str(report)


def test_dmeasure_problems_point_at_the_model_line():
    report = sde_model(dmeas=dmeas_at_runtime_index).backend_report()
    assert report.translates and not report["dmeasure"].translates
    (problem,) = report["dmeasure"].problems
    assert "runtime index" in problem.what
    assert any("(dmeas_at_runtime_index)" in w for w in problem.where), problem


def test_particle_coupling_points_at_the_model_line():
    report = sde_model(particle_mean).backend_report()
    whats = {p.what: p.where for p in report["rproc"].problems}
    assert "reduce_sum over particles" in whats
    assert any("(particle_mean)" in w for w in whats["reduce_sum over particles"])


def test_xla_backend_reports_what_would_translate():
    report = sde_model(sorts).backend_report(backend="xla")
    assert report.backend == "xla" and report.reason == "backend='xla'"
    assert not report.translates


def test_report_parts_are_named():
    report = sde_model().backend_report()
    assert [p.name for p in report.parts] == list(pypomp.cute.PARTS)
    with pytest.raises(KeyError):
        report["nothing"]
