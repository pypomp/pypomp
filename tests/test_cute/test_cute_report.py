"""Pomp.backend_report where CuTe can't run, as on a machine without a GPU."""

import jax
import jax.numpy as jnp
import pytest

import pypomp as pp
import pypomp.cute
import pypomp.cute.analysis
from pypomp.cute import PARTS, BackendReport, Part

from .test_cute_backend import (
    control_rproc,
    particlewise_rproc,
    scan_rproc,
    sde_dmeas,
    sde_model,
    sde_rproc,
    table_panel,
    uneven_model,
)


@pytest.fixture(autouse=True)
def no_cute(monkeypatch):
    monkeypatch.setattr(pypomp.cute, "available", lambda: (False, "no GPU here"))


def sorts(X_, theta_, key, covars, t, dt):
    z = X_["x"]
    return {"x": z + jnp.sort(jnp.stack([z, -z]))[0] * dt}


def dmeas_at_runtime_index(Y_, X_, theta_, covars, t):
    # Too many runtime values to select among in registers.
    v = X_["x"] * jnp.arange(100.0)
    i = jnp.clip(jnp.round(10 * X_["x"]).astype(jnp.int32), 0, 99)
    return sde_dmeas(Y_, {"x": v[i]}, theta_, covars, t)


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


def constant_nextafter(X_, theta_, key, covars, t, dt):
    # An unsupported primitive of constants only, which the translator folds.
    top = jnp.nextafter(jnp.float32(1.0), jnp.float32(0.0))
    return {"x": jnp.minimum(X_["x"], top) + 0.1 * jax.random.normal(key) * dt}


def test_unsupported_operations_of_constants_translate():
    assert sde_model(constant_nextafter).backend_report()["rproc"].translates


def test_intervals_of_different_lengths_translate():
    model = uneven_model()
    assert model.rproc.nstep is None
    assert model.backend_report().translates


def test_a_scan_translates():
    report = sde_model(scan_rproc).backend_report()
    assert report.translates, str(report)
    assert report["gradients"].translates, str(report)


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


def test_particlewise_shapes_translate():
    report = sde_model(particlewise_rproc).backend_report()
    assert report.translates and report["gradients"].translates, str(report)


def test_a_while_loop_keeps_only_the_gradients_off_the_kernel():
    # Reverse mode can't trace a while loop, even one the kernel runs forward.
    report = sde_model(control_rproc).backend_report()
    assert report.translates, str(report)
    assert not report["gradients"].translates
    (problem,) = report["gradients"].problems
    assert problem.what.startswith("tracing the gradients failed (ValueError: ")
    assert "while_loop" in problem.what


def runtime_while(X_, theta_, key, covars, t, dt):
    z = X_["x"]
    n = jnp.clip(jnp.round(10 * z), 0, 5).astype(jnp.int32)
    _, w = jax.lax.while_loop(
        lambda c: c[0] < n, lambda c: (c[0] + 1, 0.9 * c[1]), (0, z)
    )
    return sde_rproc({"x": w}, theta_, key, covars, t, dt)


def test_a_while_loop_with_a_runtime_trip_count_doesnt_translate():
    report = sde_model(runtime_while).backend_report()
    assert not report.translates
    whats = [p.what for p in report["rproc"].problems]
    assert "while loop with a runtime trip count" in whats


def _vec(f):
    """A @vectorized rproc that adds `f` of the particles to the SDE's step."""

    @pp.vectorized
    def rproc(X_, theta_, key, covars, t, dt):
        z = X_["x"]
        eps = jax.random.normal(key, z.shape)
        return {"x": z + 0.01 * f(z, key) * dt + 0.1 * jnp.sqrt(dt) * eps}

    return rproc


def _scan_over(z, key):
    return jax.lax.scan(lambda c, x: (c + x, c), 0.0, z)[1]


def _python_loop(z, key):
    return sum(z[i] for i in range(z.shape[0])) * jnp.ones_like(z)


@pytest.mark.parametrize(
    "f, what",
    [
        (lambda z, key: z + jnp.arange(z.shape[0]), "iota over particles"),
        (lambda z, key: z * (z @ z), "dot_general over particles"),
        (lambda z, key: z / z.shape[0], "constants that depend on the number"),
        (
            lambda z, key: jax.random.normal(key, (z.shape[0], 2))[:, 0],
            "random_bits with particles on a non-trailing axis",
        ),
        (
            lambda z, key: z + jax.random.normal(key, (1,)),
            "random_bits with a trailing axis of length 1",
        ),
        (lambda z, key: jnp.outer(z, z)[0], "several particle axes"),
        (_scan_over, "scan over particles"),
        (lambda z, key: jnp.concatenate([z[1:], z[:1]]), "across particles"),
        (_python_loop, "operations that depend on the number of particles"),
        (
            lambda z, key: z * 2.0 if z.shape[0] < 3 else z + 2.0,
            "operations that depend on the number of particles",
        ),
        (lambda z, key: jnp.stack([z, z]).reshape(-1)[: z.shape[0]], "reshape across"),
    ],
    ids=[
        "iota",
        "dot",
        "J-constant",
        "draw-axis",
        "draw-trailing-1",
        "outer",
        "scan",
        "roll",
        "python-loop",
        "J-branch",
        "flatten",
    ],
)
def test_particle_coupling_is_named(f, what):
    report = sde_model(_vec(f)).backend_report()
    assert not report.translates
    whats = [p.what for p in report["rproc"].problems]
    assert any(what in w for w in whats), whats


def _branches_and_loops(z, key):
    # A cond and a while loop over every particle at once, each kept to itself.
    v = jax.lax.cond(jax.random.uniform(key) > 0.5, jnp.sin, lambda v: v * v, z)
    return jax.lax.while_loop(
        lambda c: c[0] < 2, lambda c: (c[0] + 1, 0.9 * c[1]), (0, v)
    )[1]


def test_vectorized_conds_and_loops_keep_particles_apart():
    report = sde_model(_vec(_branches_and_loops)).backend_report()
    assert report.translates, str(report)


def batched(X_, theta_, key, covars, t, dt):
    z = X_["x"]
    a = jnp.stack([jnp.stack([z, 2 * z, 3 * z]), jnp.stack([-z, z, z * z])])
    i = jnp.clip(jnp.round(z), 0, 2).astype(jnp.int32)
    picked = jax.vmap(lambda r, j: r[j])(a, jnp.stack([i, 2 - i])).sum()
    dots = jnp.einsum("bi,bi->b", a, a).sum()
    return sde_rproc({"x": z + 0.01 * (picked + dots) * dt}, theta_, key, covars, t, dt)


def test_batched_gathers_and_dots_dont_translate():
    report = sde_model(batched).backend_report()
    whats = [p.what for p in report["rproc"].problems]
    assert "gather with batching dimensions" in whats
    assert "batched dot_general" in whats


def test_a_model_without_dmeas_reports_it():
    report = sde_model(dmeas=None).backend_report()
    assert report.translates
    assert report["dmeasure"].translates is None
    assert report["dmeasure"].note == "the model has no dmeas"


def failing_dmeas(Y_, X_, theta_, covars, t):
    # Passes the model's check, which calls it with floats, and fails when traced.
    if isinstance(X_["x"], jax.Array):
        raise RuntimeError("no density here")
    return sde_dmeas(Y_, X_, theta_, covars, t)


def test_a_dmeas_that_fails_to_trace_is_named():
    report = sde_model(dmeas=failing_dmeas).backend_report()
    assert report.translates
    (problem,) = report["dmeasure"].problems
    assert problem.what == "tracing dmeas failed (RuntimeError: no density here)"


def test_a_failed_analysis_is_reported(monkeypatch):
    def fail(model):
        raise RuntimeError("no jaxpr")

    monkeypatch.setattr(pypomp.cute.analysis, "_analyze", fail)
    report = sde_model().backend_report(backend="xla")
    assert not report.translates
    (problem,) = report["rproc"].problems
    assert problem.what == "tracing rproc failed (RuntimeError: no jaxpr)"


@pytest.mark.parametrize("rng", ["threefry", "philox"])
def test_report_of_built_kernels_says_what_still_runs_on_xla(rng):
    parts = tuple(
        Part(n, n != "dmeasure", True, note="" if n != "dmeasure" else "kept")
        for n in PARTS
    )
    text = str(BackendReport("cute", rng, True, "", False, parts))
    assert "  rproc         CuTe" in text
    assert "  dmeasure      XLA   kept" in text
    assert "float64 particles" in text
    assert ("keys of another generator" in text) == (rng == "threefry")
    assert "not checked against XLA (check=False)" in text


def test_panel_reports_its_first_units_model():
    panel, _, _ = table_panel()
    report = panel.backend_report()
    assert report.translates
    first = panel.unit_objects["u1"].backend_report()
    assert str(report) == str(first)
