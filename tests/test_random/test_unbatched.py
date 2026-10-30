"""The samplers' one-element paths, which branch instead of selecting inside
branching_traces, match their batched paths bit for bit, and appear nowhere else."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import ndtri as jax_ndtri

from pypomp.random import binominv, gammainv, poissoninv
from pypomp.random._branch import branching_traces
from pypomp.random._ndtri import ndtri

F32 = np.float32


def _same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


def _one_at_a_time(f, *args):
    # A function of its own, so this jit shares no trace with jits outside it.
    with branching_traces():
        g = jax.jit(lambda *a: f(*a))
        return np.array(
            [
                np.asarray(g(*(a[i : i + 1] for a in args)))[0]
                for i in range(len(args[0]))
            ]
        )


def _conds(jaxpr) -> int:
    """Count the cond equations in `jaxpr` and in every jaxpr it calls."""
    n = 0
    for eqn in jaxpr.eqns:
        n += eqn.primitive.name == "cond"
        for v in eqn.params.values():
            for sub in v if isinstance(v, (list, tuple)) else [v]:
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    n += _conds(inner)
    return n


_SCALARS = (F32(0.3), F32(20.0), F32(0.1))


def _uniforms(rng, n):
    u = rng.uniform(size=n).astype(F32)
    u[:8] = [0, 1, 1e-38, 1e-7, 0.5, 1 - 6e-8, 0.01, 0.99]
    return u


def _boundary_points():
    """Float32 neighbourhoods of ndtri's piece boundaries, w = 3.5 and 18.0625."""
    pts = []
    for w in (3.5, 18.0625):
        a = np.exp(-w) / 2
        for c in (F32(a), F32(1 - a)):
            pts.append((c.view(np.int32) + np.arange(-50, 50)).astype(np.int32))
    return np.concatenate(pts).view(F32)


def test_ndtri_matches_jax():
    rng = np.random.default_rng(0)
    p = np.concatenate(
        [
            rng.uniform(size=20000).astype(F32),
            np.exp(rng.uniform(np.log(1e-45), 0, size=20000)).astype(F32),
            _boundary_points(),
            np.array([0, 1, 0.5, -1, 2, np.nan, np.finfo(F32).tiny, 1e-45], F32),
        ]
    )
    expected = jax.jit(jax_ndtri)(jnp.asarray(p))
    assert _same(jax.jit(jax.vmap(ndtri))(jnp.asarray(p)), expected)
    idx = np.r_[len(p) - 408 : len(p), 0:100]
    assert _same(_one_at_a_time(ndtri, p[idx]), np.asarray(expected)[idx])


@pytest.mark.parametrize("p", [1e-30, 1e-6, 0.01, 0.3, 0.5, 0.9, 0.999999])
def test_ndtri_grad_matches_jax(p):
    assert jax.grad(ndtri)(F32(p)) == jax.grad(jax_ndtri)(F32(p))


def test_binominv_one_at_a_time():
    rng = np.random.default_rng(1)
    n_pts = 400
    u = _uniforms(rng, n_pts)
    n = np.floor(np.exp(rng.uniform(0, np.log(2e6), size=n_pts))).astype(F32)
    n[8:120] = rng.integers(0, 12, size=112)
    p = np.exp(rng.uniform(np.log(1e-7), 0, size=n_pts)).astype(F32)
    p[120:160] = rng.uniform(0.45, 1, size=40)
    p[160:166] = [0, 1, 0.5, 1e-30, 0.999999, 0.3]
    batched = binominv(jnp.asarray(u), jnp.asarray(n), jnp.asarray(p))
    assert _same(_one_at_a_time(binominv, u, n, p), batched)


def test_poissoninv_one_at_a_time():
    rng = np.random.default_rng(2)
    n_pts = 400
    u = _uniforms(rng, n_pts)
    lam = np.exp(rng.uniform(np.log(1e-3), np.log(1e4), size=n_pts)).astype(F32)
    lam[8:100] = rng.uniform(0, 12, size=92)
    lam[100:105] = [0, -1, 4, 4.0001, 1e-30]
    batched = poissoninv(jnp.asarray(u), jnp.asarray(lam))
    assert _same(_one_at_a_time(poissoninv, u, lam), batched)


def test_gammainv_one_at_a_time():
    rng = np.random.default_rng(3)
    n_pts = 300
    u = _uniforms(rng, n_pts)
    alpha = np.exp(rng.uniform(np.log(1e-2), np.log(1e3), size=n_pts)).astype(F32)
    batched = gammainv(jnp.asarray(u), jnp.asarray(alpha))
    assert _same(_one_at_a_time(gammainv, u, alpha), batched)


def test_shortcuts_only_while_branching():
    with branching_traces():
        assert _conds(jax.make_jaxpr(lambda p: ndtri(p))(F32(0.3)).jaxpr) > 0
        jaxpr = jax.make_jaxpr(lambda *a: binominv(*a))(*_SCALARS).jaxpr
        assert _conds(jaxpr) > 0
        # A jit call, which the CuTe translator emits once as a device function.
        assert [e.primitive.name for e in jaxpr.eqns] in (["jit"], ["pjit"])
    assert _conds(jax.make_jaxpr(lambda p: ndtri(p))(F32(0.3)).jaxpr) == 0
    assert _conds(jax.make_jaxpr(lambda *a: binominv(*a))(*_SCALARS).jaxpr) == 0
    # XLA's filters vmap the per-particle rproc over particles.
    arrays = [jnp.full(4, s) for s in _SCALARS]
    assert _conds(jax.make_jaxpr(jax.vmap(binominv))(*arrays).jaxpr) == 0


@pytest.mark.parametrize("branching_first", [True, False])
def test_jit_cache_keeps_the_paths_apart(branching_first):
    # An exact_max of its own, so no other test has traced these entries.
    def trace():
        return _conds(
            jax.make_jaxpr(lambda *a: binominv(*a, exact_max=9 + branching_first))(
                *_SCALARS
            ).jaxpr
        )

    def branching_trace():
        with branching_traces():
            return trace()

    if branching_first:
        assert branching_trace() > 0 and trace() == 0
    else:
        assert trace() == 0 and branching_trace() > 0
