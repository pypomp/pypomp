"""The CuTe kernels for the filter outside rproc: resampling and the fused density."""

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pypomp as pp
import pypomp.functional as F
from pypomp.core.algorithms.helpers import _resample
from pypomp.cute import KernelRProc, available

from .test_cute_backend import he10_model, sde_model, xla_and_cute

ok, why = available()
pytestmark = pytest.mark.skipif(not ok, reason=f"CuTe backend unavailable: {why}")


def log_weights(key, R, J):
    """Normalized log-weights from near-uniform to degenerate, some with dead runs."""
    ks = jax.random.split(key, 2)
    z = jax.random.normal(ks[0], (R, J)) * jnp.linspace(0.1, 8.0, R)[:, None]
    dead = jax.random.uniform(ks[1], (R, J)) < jnp.linspace(0.0, 0.9, R)[:, None]
    runs = (jnp.arange(J) // 37) % 3 == 0
    w = jnp.where(dead | (runs & (jnp.arange(R) % 2 == 0)[:, None]), -jnp.inf, z)
    w = w.at[:, J // 2].set(z[:, J // 2])  # at least one live particle
    return w - jax.scipy.special.logsumexp(w, axis=-1, keepdims=True)


@pytest.mark.parametrize("J", [1, 7, 256, 5003])
def test_resample_kernel_keeps_systematic_resampling_guarantees(J):
    rp = sde_model().to_struct(rproc_backend="cute").rproc_pf
    assert isinstance(rp, KernelRProc) and rp.resample is not None
    resample = rp.resample
    R = 64
    W = log_weights(jax.random.key(J), R, J)
    K = jax.random.split(jax.random.key(J + 1), R)
    got = np.asarray(jax.jit(jax.vmap(resample))(W, K))
    ref = np.asarray(jax.jit(jax.vmap(_resample))(W, K))
    w = np.exp(np.asarray(W, np.float64))
    assert ((got >= 0) & (got < J)).all()
    assert not (w[np.arange(R)[:, None], got] == 0).any()
    counts = np.stack([np.bincount(g, minlength=J) for g in got])
    jw = J * w / w.sum(-1, keepdims=True)
    # floor or ceil of J w, give or take the rounding of the cumulative weights
    assert ((counts >= np.floor(jw) - 1) & (counts <= np.ceil(jw) + 1)).all()
    # Only the summation order differs from XLA, so nearly every index agrees.
    assert (got != ref).mean() < 1e-3


def test_few_filters_of_many_particles_resample_on_xla():
    rp = sde_model().to_struct(rproc_backend="cute").rproc_pf
    assert isinstance(rp, KernelRProc) and rp.resample is not None
    resample = rp.resample
    R, J = 2, 20011
    W = log_weights(jax.random.key(3), R, J)
    K = jax.random.split(jax.random.key(4), R)
    got = jax.jit(jax.vmap(resample))(W, K)
    np.testing.assert_array_equal(got, jax.jit(jax.vmap(_resample))(W, K))


def unfused(struct):
    """`struct` with its density wrapped, so the kernel can't evaluate it."""
    dm_pf, dm_per = struct.dmeas_pf, struct.dmeas_per
    return struct._replace(
        dmeas_pf=lambda *a: dm_pf(*a), dmeas_per=lambda *a: dm_per(*a)
    )


# A density evaluated in the kernel rounds differently from XLA's, which moves the
# odd systematic resampling index, so whole filters agree only in distribution.


@pytest.mark.parametrize(
    ("make", "rng"),
    [(sde_model, "threefry"), (he10_model, "threefry"), (sde_model, "philox")],
    ids=["sde", "he10", "sde-philox"],
)
def test_fused_density_matches_the_separate_one(make, rng):
    model = make()
    _, cute = xla_and_cute(model, rng)
    for rp in (cute.rproc_pf, cute.rproc_per):
        assert isinstance(rp, KernelRProc) and rp.with_dmeasure is not None
    # Built only when its check against XLA passes.
    assert cute.rproc_per.perturb_step is not None
    assert cute.rproc_pf.perturb_step is None
    th = jnp.tile(model.theta.to_jax_array(model.canonical_param_names), (4, 1))
    keys = jax.random.split(jax.random.key(0), 4)[:, None]
    a = F.pfilter(cute, th, 300, keys, CLL=True)["CLL"]
    b = F.pfilter(unfused(cute), th, 300, keys, CLL=True)["CLL"]
    # The first observation follows the same particles either way.
    np.testing.assert_allclose(a[..., 0], b[..., 0], rtol=1e-5, atol=1e-4)


def test_filters_with_fused_density_agree_in_distribution():
    model = sde_model()
    _, cute = xla_and_cute(model)
    th = jnp.tile(model.theta.to_jax_array(model.canonical_param_names), (200, 1))
    keys = jax.random.split(jax.random.key(0), 200)[:, None]
    a = np.asarray(F.pfilter(cute, th, 500, keys)["logLik"]).ravel()
    b = np.asarray(F.pfilter(unfused(cute), th, 500, keys)["logLik"]).ravel()
    se = np.sqrt(a.var() / a.size + b.var() / b.size)
    assert abs(a.mean() - b.mean()) < 4 * se

    rw = pp.RWSigma(
        sigmas={"r": 0.02, "sig": 0.02, "tau": 0.02, "z0m": 0.0, "z0s": 0.0},
        init_names=[],
    ).geometric_cooling(0.5)
    keys = jax.random.split(jax.random.key(4), 200)
    a = np.asarray(F.mif(cute, th, 256, 1, rw, keys, return_swarm=False)[0]).ravel()
    b = np.asarray(
        F.mif(unfused(cute), th, 256, 1, rw, keys, return_swarm=False)[0]
    ).ravel()
    se = np.sqrt(a.var() / a.size + b.var() / b.size)
    assert abs(a.mean() - b.mean()) < 4 * se


@pytest.mark.parametrize("should_trans", [False, True], ids=["natural", "estimation"])
def test_fused_density_gradients_match_the_separate_one(should_trans):
    model = he10_model()
    _, cute = xla_and_cute(model)
    rp = cute.rproc_per if should_trans else cute.rproc_pf
    assert isinstance(rp, KernelRProc)
    with_dmeasure, dmeasure = rp.with_dmeasure, rp.dmeasure
    assert with_dmeasure is not None and dmeasure is not None
    names = model.canonical_param_names
    th_nat = model.theta.to_jax_array(names)[0]
    th = th_nat
    if should_trans:
        th = model.par_trans._transform_array(th[None], names, direction="to_est")[0]
        th = jnp.tile(th, (128, 1))
    cov0 = None if cute.covars_extended is None else cute.covars_extended[0]
    X0 = cute.rinit_pf(
        th_nat, jax.random.split(jax.random.key(9), 128), cov0, cute.t0, False
    )
    nstep = int(np.asarray(cute.nstep_array)[0])
    rest = (cute.covars_extended, cute.dt_array_extended, cute.t0, 0, nstep)
    rest = rest + (cute.accumvars, should_trans)
    y, t_obs = cute.ys[0], cute.times[0]
    c = jnp.cos(0.1 * jnp.arange(X0.size, dtype=jnp.float32)).reshape(X0.shape)
    key = jax.random.key(5)

    def fused(X, th):
        out, _, w = with_dmeasure(X, th, key, *rest, y, t_obs)
        return jnp.sum(out * c) + jnp.sum(jnp.where(jnp.isfinite(w), w, 0.0))

    def separate(X, th):
        out, t_end = rp(X, th, key, *rest)
        cov = None if cute.covars_extended is None else cute.covars_extended[t_end]
        w = dmeasure(y, out, th, cov, t_obs, should_trans)
        return jnp.sum(out * c) + jnp.sum(jnp.where(jnp.isfinite(w), w, 0.0))

    for a, b in zip(
        jax.jit(jax.grad(fused, argnums=(0, 1)))(X0, th),
        jax.jit(jax.grad(separate, argnums=(0, 1)))(X0, th),
        strict=True,
    ):
        a, b = np.asarray(a), np.asarray(b)
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6 * np.abs(b).max())


def test_mif_kernel_returns_the_resampled_swarm():
    model = sde_model()
    _, cute = xla_and_cute(model)
    rw = pp.RWSigma(
        sigmas={"r": 0.02, "sig": 0.02, "tau": 0.02, "z0m": 0.0, "z0s": 0.0},
        init_names=[],
    ).geometric_cooling(0.5)
    th = jnp.tile(model.theta.to_jax_array(model.canonical_param_names), (3, 1))
    keys = jax.random.split(jax.random.key(7), 3)
    logliks, traces, swarm = F.mif(cute, th, 256, 2, rw, keys)
    assert np.isfinite(np.asarray(logliks)).all()
    # The traces' last row is the mean of the swarm the kernel path gathers at the end.
    np.testing.assert_allclose(
        np.asarray(traces)[:, -1], np.asarray(swarm).mean(axis=1), rtol=1e-5, atol=1e-6
    )
    # Parameters without a random walk stay where they started.
    names = model.canonical_param_names
    swarm = np.asarray(swarm)
    for n in ("z0m", "z0s"):
        k = names.index(n)
        start = np.broadcast_to(np.asarray(th)[:, None, k], swarm.shape[:2])
        np.testing.assert_allclose(swarm[..., k], start)


def test_resample_kernel_leaves_float64_on_xla():
    from pypomp.cute.resample import resample

    jax.config.update("jax_enable_x64", True)
    try:
        W = jnp.log(jnp.full(10, 0.1, jnp.float64))
        k = jax.random.key(0)
        np.testing.assert_array_equal(resample(W, k), _resample(W, k))
    finally:
        jax.config.update("jax_enable_x64", False)


def test_deepcopy_keeps_the_filter_kernels():
    model = sde_model()
    model.set_rproc_backend("cute")
    s = deepcopy(model).to_struct()
    assert isinstance(s.rproc_pf, KernelRProc)
    assert s.rproc_pf.resample is not None
    assert s.rproc_pf.with_dmeasure is not None
