"""One CuTe kernel per observation interval, in place of pypomp's ``rprocess_interp``.

Each CUDA thread owns one particle of one filter, keeps its state in registers and
runs the interval's Euler steps with the user's ``rproc``, translated from its jaxpr
by ``translate.py``. Every filter of a vmapped call (parameter sets, replicates,
iterated filtering searches) goes into the same launch: a ``custom_vmap`` rule folds
each batch dimension into the kernel's particle axis.

The rest of the particle filter is pypomp's own. The substep keys are split as
``_time_interp`` splits them, and the kernel draws from them with the generator that
``rng`` names: JAX's threefry, so its random numbers are XLA's, or pypomp's
Philox4x32-10 (:mod:`pypomp.random.philox`), which costs fewer instructions.
Constant tables that the step reads at runtime indices reach every kernel in one
buffer argument (see ``translate.py``). What every particle of a filter shares, the
part of the step and density that depends only on the time, covariates, observation
and, unless each particle has its own, the parameters, runs once per filter on XLA,
and the kernel reads the results from each step's row of the step table.

Gradients go through ``jax.custom_vjp``. When JAX differentiates through an interval,
the forward kernel also stores each Euler step's input state (nstep x ns floats per
particle, for one interval at a time, since pypomp checkpoints each observation). A
second kernel runs the vector-Jacobian product (VJP) of the user's step, from
``jax.vjp`` through the translator, over the steps in reverse. It regenerates the
random numbers instead of storing them.
"""

from __future__ import annotations

import atexit
import functools
import hashlib
import importlib.util
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import cutlass  # pyright: ignore[reportMissingImports]
from cutlass.jax import cutlass_call  # pyright: ignore[reportMissingImports]
from jax.extend.core import jaxpr_as_fun

from . import KernelRProc
from . import translate as ct
from .resample import resample as _resample_kernel

if TYPE_CHECKING:
    from ..core.pomp import Pomp

# A cute.Tensor inside a kernel, and a CuTe DSL scalar or constant.
_Tensor = Any
_Scalar = ct._Scalar
# CUDA threads per block.
_BLOCK = 128
# Particles in the trace that finds a @vectorized rproc's particle axis: a prime that
# no other axis length is likely to equal.
_PROBE_J = 1031
# _trace_step's output: step, transform, whether vectorized, and their VJPs.
_Traced = tuple[ct.ClosedJaxpr, ct.ClosedJaxpr, bool, ct.ClosedJaxpr, ct.ClosedJaxpr]
# The process kernel, the one with the fused density and iterated filtering's step;
# the last two are None when the model's density doesn't translate.
_Cores = tuple[Callable, Callable | None, Callable | None]


class _Spec(NamedTuple):
    """What the generated kernels need to know about the model."""

    step_j: ct.ClosedJaxpr
    prep_j: ct.ClosedJaxpr
    step_vjp_j: ct.ClosedJaxpr
    prep_vjp_j: ct.ClosedJaxpr
    vec: bool
    ns: int
    nth: int
    nc: int
    nstep: int
    acc: tuple[int, ...]
    per_particle: bool
    rng: str
    dm_j: ct.ClosedJaxpr | None
    ny: int
    ncd: int
    noise_j: ct.ClosedJaxpr
    offsets: dict[tuple, int]
    """Where each table the step reads at runtime indices starts in the kernels'
    table buffer (see :func:`translate._runtime_tables`)."""
    res: tuple[tuple[tuple[int, ...], np.dtype], ...]
    """Shapes and dtypes of the values hoisted out of the step (see
    :func:`translate._hoist`), which follow the covariates in each step's row."""
    dm_res: tuple[tuple[tuple[int, ...], np.dtype], ...]
    """The same for the density, whose values end each observation's row."""


def _trace_step(model: Pomp) -> _Traced:
    """Return per-particle jaxprs of the Euler step, the parameter transform and VJPs.

    The step maps (x, theta, key data, covariates, t, dt) to the next x, with theta on
    the natural scale; the transform maps theta from the estimation scale to the
    natural scale. The third output is whether the rproc is ``@vectorized``.
    """
    rp = model.rproc
    snames = list(rp.statenames)
    pnames = list(rp.param_names)
    vec = bool(rp._is_vectorized)
    ns, d, nc = len(snames), len(pnames), len(rp.covar_names)

    def step(x, th, kd, cov, t, dt):
        X = {n: (x[i][None] if vec else x[i]) for i, n in enumerate(snames)}
        theta = rp._prepare_theta(th, False)
        covd = rp._unpack_covars(cov) if nc else {}
        res = rp._call_original(
            X_=X,
            theta_=theta,
            key=jax.random.wrap_key_data(kd),
            covars=covd,
            t=t,
            dt=dt,
        )
        return jnp.stack([jnp.asarray(res[n], jnp.float32).reshape(()) for n in snames])

    def prep(th):
        theta = rp._prepare_theta(th, True)
        return jnp.stack(
            [jnp.asarray(theta[n], jnp.float32).reshape(()) for n in pnames]
        )

    def step_vjp(x, th, kd, cov, t, dt, gx):
        _, pull = jax.vjp(lambda x, th: step(x, th, kd, cov, t, dt), x, th)
        return pull(gx)

    def prep_vjp(th, gth):
        return jax.vjp(prep, th)[1](gth)[0]

    f32 = jnp.float32
    args = (
        jnp.zeros(ns, f32),
        jnp.ones(d, f32),
        jnp.zeros(2, jnp.uint32),
        jnp.ones(max(nc, 1), f32)[:nc],
        f32(0.0),
        f32(0.01),
    )
    step_j = jax.make_jaxpr(step)(*args)
    prep_j = jax.make_jaxpr(prep)(jnp.zeros(d, f32))
    step_vjp_j = jax.make_jaxpr(step_vjp)(*args, jnp.zeros(ns, f32))
    prep_vjp_j = jax.make_jaxpr(prep_vjp)(jnp.zeros(d, f32), jnp.zeros(d, f32))
    return step_j, prep_j, vec, step_vjp_j, prep_vjp_j


def _trace_noise(d: int) -> ct.ClosedJaxpr:
    """Return the jaxpr of d standard normal draws from a key's data."""
    return jax.make_jaxpr(
        lambda kd: jax.random.normal(jax.random.wrap_key_data(kd), (d,), jnp.float32)
    )(jnp.zeros(2, jnp.uint32))


def _trace_dmeas(model: Pomp) -> ct.ClosedJaxpr | None:
    """Return the per-particle jaxpr of the measurement density, or None without one.

    It maps (y, x, theta, covariates, t) to the log-density, with theta on the natural
    scale.
    """
    dm = model.dmeas
    if dm is None:
        return None
    ny, ns = len(dm.y_names), len(dm.statenames)
    d, nc = len(dm.param_names), len(dm.covar_names)

    def dmeas(y, x, th, cov, t):
        out = dm.mechanics(y, x, th, cov if nc else None, t, False)
        return jnp.asarray(out, jnp.float32).reshape(())

    f32 = jnp.float32
    return jax.make_jaxpr(dmeas)(
        jnp.zeros(ny, f32),
        jnp.zeros(ns, f32),
        jnp.ones(d, f32),
        jnp.ones(max(nc, 1), f32)[:nc],
        f32(0.0),
    )


def _vectorized_jaxpr(model: Pomp, J: int) -> ct.ClosedJaxpr:
    """Return the jaxpr of a ``@vectorized`` rproc's Euler step on J particles."""
    rp = model.rproc
    snames = list(rp.statenames)
    ns, d, nc = len(snames), len(rp.param_names), len(rp.covar_names)

    def step(x, th, kd, cov, t, dt):
        res = rp._call_original(
            X_={n: x[i] for i, n in enumerate(snames)},
            theta_=rp._prepare_theta(th, False),
            key=jax.random.wrap_key_data(kd),
            covars=rp._unpack_covars(cov) if nc else {},
            t=t,
            dt=dt,
        )
        return jnp.stack([jnp.broadcast_to(res[n], (J,)) for n in snames])

    f32 = jnp.float32
    return jax.make_jaxpr(step)(
        jnp.zeros((ns, J), f32),
        jnp.ones(d, f32),
        jnp.zeros(2, jnp.uint32),
        jnp.ones(max(nc, 1), f32)[:nc],
        f32(0.0),
        f32(0.01),
    )


def _particle_mixing(model: Pomp) -> set[str]:
    """Return what in a ``@vectorized`` rproc couples particles, which the kernel
    can't run with one particle per thread."""
    two = _vectorized_jaxpr(model, 2).jaxpr
    probe = _vectorized_jaxpr(model, _PROBE_J).jaxpr
    return ct._particle_mixing(two, probe, _PROBE_J)


# The functions below run at trace time inside a kernel, for one thread's particle.


def _load_th(
    TH: _Tensor, g: _Scalar, b: _Scalar, nth: int, per_particle: bool
) -> np.ndarray:
    """Return the parameters of particle g, or of its filter b."""
    row = g if per_particle else b
    th = np.empty(nth, dtype=object)
    for k in range(nth):
        th[k] = TH[row, k]
    return th


def _load(
    X: _Tensor,
    TH: _Tensor,
    g: _Scalar,
    b: _Scalar,
    ns: int,
    nth: int,
    acc: tuple[int, ...],
    per_particle: bool,
) -> tuple[tuple[_Scalar, ...], np.ndarray]:
    """Return particle g's state, with accumulators reset, and its parameters."""
    xs = [X[g, s] for s in range(ns)]
    for a in acc:
        xs[a] = cutlass.Float32(0.0)
    return tuple(xs), _load_th(TH, g, b, nth, per_particle)


def _tables(TB: _Tensor, offsets: dict[tuple, int]) -> ct._Tables:
    """Return the kernel's table buffer `TB` for the translator."""
    return ct._Tables(TB, offsets)


def _prep(
    prep_j: ct.ClosedJaxpr, th: np.ndarray, rng: str, tb: ct._Tables
) -> np.ndarray:
    """Return the parameters `th` moved from the estimation to the natural scale."""
    ctx = ct._Ctx(vectorized_J=None, particle=None, rng=rng, tables=tb)
    (out,) = ct._interpret(prep_j, [th], ctx)
    return np.array([ct._as_dsl(v, np.float32) for v in out], dtype=object)


def _step_keys(
    vec: bool, rng: str, KEYS: _Tensor, b: _Scalar, j: _Scalar, k: _Scalar
) -> tuple[_Scalar, _Scalar]:
    """Return particle j's key for step k of filter b, as pypomp derives it."""
    k0, k1 = KEYS[b, k, 0], KEYS[b, k, 1]
    if vec:
        return k0, k1
    return ct._key_derive(rng, k0, k1, j)  # split(substep_key, J)[j]


def _step_once(
    step_j: ct.ClosedJaxpr,
    vec: bool,
    rng: str,
    J: int,
    ns: int,
    nc: int,
    xs: tuple[_Scalar, ...],
    th: np.ndarray,
    KEYS: _Tensor,
    TAB: _Tensor,
    tb: ct._Tables,
    b: _Scalar,
    j: _Scalar,
    k: _Scalar,
    res: tuple = (),
) -> tuple[_Scalar, ...]:
    """Return the state after Euler step k; k may be a runtime loop index. The
    values `res` describes, hoisted out of the step, follow the covariates."""
    ctx = ct._Ctx(vectorized_J=J if vec else None, particle=j, rng=rng, tables=tb)
    k0, k1 = _step_keys(vec, rng, KEYS, b, j, k)
    cov = np.empty(nc, dtype=object)
    for c in range(nc):
        cov[c] = TAB[b, k, 2 + c]
    (out,) = ct._interpret(
        step_j,
        [
            ct._obj(list(xs)),
            th,
            ct._obj([k0, k1]),
            cov,
            ct._obj(TAB[b, k, 0]),
            ct._obj(TAB[b, k, 1]),
            *_read_hoisted(lambda c: TAB[b, k, 2 + nc + c], res),
        ],
        ctx,
    )
    return tuple(ct._as_dsl(out[s], np.float32) for s in range(ns))


def _dmeas_once(
    dm_j: ct.ClosedJaxpr,
    rng: str,
    ny: int,
    ncd: int,
    xs: tuple[_Scalar, ...],
    th: np.ndarray,
    OBS: _Tensor,
    tb: ct._Tables,
    b: _Scalar,
    res: tuple = (),
) -> _Scalar:
    """Return the measurement log-density of state `xs`; OBS[b] holds filter b's
    observation, then the observation time, its covariates and the values `res`
    describes, hoisted out of the density."""
    y = np.empty(ny, dtype=object)
    for i in range(ny):
        y[i] = OBS[b, i]
    cov = np.empty(ncd, dtype=object)
    for c in range(ncd):
        cov[c] = OBS[b, ny + 1 + c]
    ctx = ct._Ctx(vectorized_J=None, particle=None, rng=rng, tables=tb)
    hoisted = _read_hoisted(lambda c: OBS[b, ny + 1 + ncd + c], res)
    (out,) = ct._interpret(
        dm_j, [y, ct._obj(list(xs)), th, cov, ct._obj(OBS[b, ny]), *hoisted], ctx
    )
    return ct._as_dsl(np.asarray(out, dtype=object).reshape(-1)[0], np.float32)


def _read_hoisted(col: Callable[[int], _Scalar], res: tuple) -> list[np.ndarray]:
    """Return the hoisted values `res` describes, from consecutive float32 columns
    ``col(0), col(1), ...`` that hold their bits (see :func:`_as_columns`)."""
    out, c = [], 0
    for shape, dtype in res:
        a = np.empty(shape, dtype=object)
        for idx in np.ndindex(*shape) if shape else [()]:
            v = col(c)
            if dtype == np.bool_:
                v = v != cutlass.Float32(0.0)
            elif dtype != np.float32:
                v = v.bitcast(ct._DSL[dtype])
            a[idx] = v
            c += 1
        out.append(a)
    return out


def _as_columns(vals: list[jax.Array], lead: int) -> jax.Array:
    """Return hoisted values, each with `lead` leading axes, as float32 columns
    holding their bits, or booleans as 0 and 1."""
    cols = []
    for v in vals:
        v = v.reshape(v.shape[:lead] + (-1,))
        if v.dtype == jnp.bool_:
            v = v.astype(jnp.float32)
        elif v.dtype != jnp.float32:
            v = jax.lax.bitcast_convert_type(v, jnp.float32)
        cols.append(v)
    return jnp.concatenate(cols, axis=-1)


def _perturb(
    noise_j: ct.ClosedJaxpr,
    rng: str,
    nth: int,
    th: np.ndarray,
    PK: _Tensor,
    SIG: _Tensor,
    THO: _Tensor,
    b: _Scalar,
    j: _Scalar,
    g: _Scalar,
) -> np.ndarray:
    """Return parameters `th` plus SIG[b] times standard normal draws, and write
    them to row g of THO.

    With threefry the draws are row j of ``normal(key, (J, nth))`` for key PK[b], as
    iterated filtering on XLA draws them. Philox can't place a row of that draw in
    one thread, so it draws ``normal(split(key, J)[j], (nth,))``.
    """
    if rng == "threefry":
        k0, k1 = PK[b, 0], PK[b, 1]
        offset = j * cutlass.Uint32(nth)
        ctx = ct._Ctx(vectorized_J=None, particle=None, rng=rng, counter_offset=offset)
    else:
        k0, k1 = ct._key_derive(rng, PK[b, 0], PK[b, 1], j)
        ctx = ct._Ctx(vectorized_J=None, particle=None, rng=rng)
    (z,) = ct._interpret(noise_j, [ct._obj([k0, k1])], ctx)
    out = np.empty(nth, dtype=object)
    for k in range(nth):
        out[k] = th[k] + SIG[b, k] * ct._as_dsl(z[k], np.float32)
        THO[g, k] = out[k]
    return out


def _store(OUT: _Tensor, g: _Scalar, xs: tuple[_Scalar, ...]) -> None:
    """Write particle g's state."""
    for s, x in enumerate(xs):
        OUT[g, s] = x


def _save(H: _Tensor, g: _Scalar, k: _Scalar, xs: tuple[_Scalar, ...]) -> None:
    """Write step k's input state to particle g's row of the state buffer."""
    ns = len(xs)
    for s, x in enumerate(xs):
        H[g, k * ns + s] = x


def _hist(H: _Tensor, g: _Scalar, k: _Scalar, ns: int) -> tuple[_Scalar, ...]:
    """Return step k's input state from particle g's row of the state buffer."""
    return tuple(H[g, k * ns + s] for s in range(ns))


def _load_g(G: _Tensor, g: _Scalar, ns: int) -> tuple[_Scalar, ...]:
    """Return the cotangent of particle g's output state."""
    return tuple(G[g, s] for s in range(ns))


def _zeros(n: int) -> tuple[_Scalar, ...]:
    """Return n float32 zeros."""
    return tuple(cutlass.Float32(0.0) for _ in range(n))


def _vjp_once(
    step_vjp_j: ct.ClosedJaxpr,
    vec: bool,
    rng: str,
    J: int,
    ns: int,
    nth: int,
    nc: int,
    xs: tuple[_Scalar, ...],
    th: np.ndarray,
    KEYS: _Tensor,
    TAB: _Tensor,
    tb: ct._Tables,
    b: _Scalar,
    j: _Scalar,
    k: _Scalar,
    gs: tuple[_Scalar, ...],
    acc_th: tuple[_Scalar, ...],
) -> tuple[_Scalar, ...]:
    """Return the state cotangent before step k and the summed theta cotangent."""
    ctx = ct._Ctx(vectorized_J=J if vec else None, particle=j, rng=rng, tables=tb)
    k0, k1 = _step_keys(vec, rng, KEYS, b, j, k)
    cov = np.empty(nc, dtype=object)
    for c in range(nc):
        cov[c] = TAB[b, k, 2 + c]
    gx, gth = ct._interpret(
        step_vjp_j,
        [
            ct._obj(list(xs)),
            th,
            ct._obj([k0, k1]),
            cov,
            ct._obj(TAB[b, k, 0]),
            ct._obj(TAB[b, k, 1]),
            ct._obj(list(gs)),
        ],
        ctx,
    )
    gx = tuple(ct._as_dsl(gx[s], np.float32) for s in range(ns))
    acc_th = tuple(a + ct._as_dsl(gth[i], np.float32) for i, a in enumerate(acc_th))
    return gx + acc_th


def _store_bwd(
    DX: _Tensor,
    DTH: _Tensor,
    g: _Scalar,
    gx: tuple[_Scalar, ...],
    gth: tuple[_Scalar, ...],
    acc: tuple[int, ...],
    prep_vjp_j: ct.ClosedJaxpr,
    th_raw: np.ndarray,
    transform: bool,
    rng: str,
    tb: ct._Tables,
) -> None:
    """Write particle g's input-state and parameter cotangents."""
    for s, v in enumerate(gx):
        # Accumulators are reset at the start of the interval, so get no gradient.
        DX[g, s] = cutlass.Float32(0.0) if s in acc else v
    if transform:
        ctx = ct._Ctx(vectorized_J=None, particle=None, rng=rng, tables=tb)
        (out,) = ct._interpret(prep_vjp_j, [th_raw, ct._obj(list(gth))], ctx)
        gth = tuple(ct._as_dsl(v, np.float32) for v in out)
    for i, v in enumerate(gth):
        DTH[g, i] = v


# CuTe's AST preprocessor turns `for k in range(NSTEP)` into a device loop with the
# state in loop-carried registers, so the step's code is emitted once, not NSTEP
# times. That needs one named variable per state, hence generated source per model.
_SRC = """
import cutlass
import cutlass.cute as cute
from pypomp.cute.kernel import _load, _prep, _step_once, _store, _tables


def make(spec, J, transform, block):
    step_j = spec.step_j
    prep_j = spec.prep_j
    vec = spec.vec
    ns = spec.ns
    nth = spec.nth
    nc = spec.nc
    NSTEP = spec.nstep
    acc = spec.acc
    per_particle = spec.per_particle
    rng = spec.rng
    offsets = spec.offsets
    res = spec.res
    @cute.kernel
    def kernel(X: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor, TB: cute.Tensor,
               OUT: cute.Tensor, N: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        g = bidx * block + tidx
        if g < N:
            b = g // J
            j = (g - b * J).to(cutlass.Uint32)
            tb = _tables(TB, offsets)
            xs0, th = _load(X, TH, g, b, ns, nth, acc, per_particle)
            if cutlass.const_expr(transform):
                th = _prep(prep_j, th, rng, tb)
            {xs} = xs0
            for k in range(NSTEP):
                {xs} = _step_once(step_j, vec, rng, J, ns, nc, ({xs}), th, KEYS, TAB, tb, b, j, k, res)
            _store(OUT, g, ({xs}))

    @cute.jit
    def launch(stream, X: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor,
               TB: cute.Tensor, OUT: cute.Tensor):
        N = cute.size(X, mode=[0])
        kernel(X, TH, KEYS, TAB, TB, OUT, N).launch(grid=[(N + block - 1) // block, 1, 1],
                                                block=[block, 1, 1], stream=stream)

    return launch
"""


_SRC_SAVE = """
import cutlass
import cutlass.cute as cute
from pypomp.cute.kernel import _load, _prep, _step_once, _store, _save, _tables


def make(spec, J, transform, block):
    step_j = spec.step_j
    prep_j = spec.prep_j
    vec = spec.vec
    ns = spec.ns
    nth = spec.nth
    nc = spec.nc
    NSTEP = spec.nstep
    acc = spec.acc
    per_particle = spec.per_particle
    rng = spec.rng
    offsets = spec.offsets
    res = spec.res
    @cute.kernel
    def kernel(X: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor, TB: cute.Tensor,
               OUT: cute.Tensor, H: cute.Tensor, N: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        g = bidx * block + tidx
        if g < N:
            b = g // J
            j = (g - b * J).to(cutlass.Uint32)
            tb = _tables(TB, offsets)
            xs0, th = _load(X, TH, g, b, ns, nth, acc, per_particle)
            if cutlass.const_expr(transform):
                th = _prep(prep_j, th, rng, tb)
            {xs} = xs0
            for k in range(NSTEP):
                _save(H, g, k, ({xs}))
                {xs} = _step_once(step_j, vec, rng, J, ns, nc, ({xs}), th, KEYS, TAB, tb, b, j, k, res)
            _store(OUT, g, ({xs}))

    @cute.jit
    def launch(stream, X: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor,
               TB: cute.Tensor, OUT: cute.Tensor, H: cute.Tensor):
        N = cute.size(X, mode=[0])
        kernel(X, TH, KEYS, TAB, TB, OUT, H, N).launch(grid=[(N + block - 1) // block, 1, 1],
                                                   block=[block, 1, 1], stream=stream)

    return launch
"""


_SRC_DM = """
import cutlass
import cutlass.cute as cute
from pypomp.cute.kernel import _load, _prep, _step_once, _store, _dmeas_once, _tables


def make(spec, J, transform, block):
    step_j = spec.step_j
    prep_j = spec.prep_j
    vec = spec.vec
    ns = spec.ns
    nth = spec.nth
    nc = spec.nc
    NSTEP = spec.nstep
    acc = spec.acc
    per_particle = spec.per_particle
    rng = spec.rng
    dm_j = spec.dm_j
    ny = spec.ny
    ncd = spec.ncd
    offsets = spec.offsets
    res = spec.res
    dm_res = spec.dm_res
    @cute.kernel
    def kernel(X: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor, TB: cute.Tensor,
               OBS: cute.Tensor, OUT: cute.Tensor, W: cute.Tensor, N: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        g = bidx * block + tidx
        if g < N:
            b = g // J
            j = (g - b * J).to(cutlass.Uint32)
            tb = _tables(TB, offsets)
            xs0, th = _load(X, TH, g, b, ns, nth, acc, per_particle)
            if cutlass.const_expr(transform):
                th = _prep(prep_j, th, rng, tb)
            {xs} = xs0
            for k in range(NSTEP):
                {xs} = _step_once(step_j, vec, rng, J, ns, nc, ({xs}), th, KEYS, TAB, tb, b, j, k, res)
            _store(OUT, g, ({xs}))
            W[g] = _dmeas_once(dm_j, rng, ny, ncd, ({xs}), th, OBS, tb, b, dm_res)

    @cute.jit
    def launch(stream, X: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor,
               TB: cute.Tensor, OBS: cute.Tensor, OUT: cute.Tensor, W: cute.Tensor):
        N = cute.size(X, mode=[0])
        kernel(X, TH, KEYS, TAB, TB, OBS, OUT, W, N).launch(grid=[(N + block - 1) // block, 1, 1],
                                                        block=[block, 1, 1], stream=stream)

    return launch
"""


_SRC_PER = """
import cutlass
import cutlass.cute as cute
from pypomp.cute.kernel import _load, _prep, _step_once, _store, _dmeas_once, _perturb, _tables


def make(spec, J, transform, block):
    step_j = spec.step_j
    prep_j = spec.prep_j
    vec = spec.vec
    ns = spec.ns
    nth = spec.nth
    nc = spec.nc
    NSTEP = spec.nstep
    acc = spec.acc
    per_particle = spec.per_particle
    rng = spec.rng
    dm_j = spec.dm_j
    ny = spec.ny
    ncd = spec.ncd
    noise_j = spec.noise_j
    offsets = spec.offsets
    res = spec.res
    dm_res = spec.dm_res
    @cute.kernel
    def kernel(X: cute.Tensor, TH: cute.Tensor, ANC: cute.Tensor, PK: cute.Tensor, SIG: cute.Tensor,
               KEYS: cute.Tensor, TAB: cute.Tensor, TB: cute.Tensor, OBS: cute.Tensor, OUT: cute.Tensor,
               THO: cute.Tensor, W: cute.Tensor, N: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        g = bidx * block + tidx
        if g < N:
            b = g // J
            j = (g - b * J).to(cutlass.Uint32)
            tb = _tables(TB, offsets)
            xs0, th = _load(X, TH, b * J + ANC[g], b, ns, nth, acc, per_particle)
            th = _perturb(noise_j, rng, nth, th, PK, SIG, THO, b, j, g)
            if cutlass.const_expr(transform):
                th = _prep(prep_j, th, rng, tb)
            {xs} = xs0
            for k in range(NSTEP):
                {xs} = _step_once(step_j, vec, rng, J, ns, nc, ({xs}), th, KEYS, TAB, tb, b, j, k, res)
            _store(OUT, g, ({xs}))
            W[g] = _dmeas_once(dm_j, rng, ny, ncd, ({xs}), th, OBS, tb, b, dm_res)

    @cute.jit
    def launch(stream, X: cute.Tensor, TH: cute.Tensor, ANC: cute.Tensor, PK: cute.Tensor,
               SIG: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor, TB: cute.Tensor,
               OBS: cute.Tensor, OUT: cute.Tensor, THO: cute.Tensor, W: cute.Tensor):
        # OUT and THO are inputs aliased to outputs, so cutlass_call passes them once.
        N = cute.size(X, mode=[0])
        kernel(X, TH, ANC, PK, SIG, KEYS, TAB, TB, OBS, OUT, THO, W, N).launch(
            grid=[(N + block - 1) // block, 1, 1], block=[block, 1, 1], stream=stream)

    return launch
"""


_SRC_BWD = """
import cutlass
import cutlass.cute as cute
from pypomp.cute.kernel import _load_th, _prep, _hist, _load_g, _zeros, _vjp_once, _store_bwd, _tables


def make(spec, J, transform, block):
    prep_j = spec.prep_j
    step_vjp_j = spec.step_vjp_j
    prep_vjp_j = spec.prep_vjp_j
    vec = spec.vec
    ns = spec.ns
    nth = spec.nth
    nc = spec.nc
    NSTEP = spec.nstep
    acc = spec.acc
    per_particle = spec.per_particle
    rng = spec.rng
    offsets = spec.offsets
    @cute.kernel
    def kernel(H: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor, TB: cute.Tensor,
               G: cute.Tensor, DX: cute.Tensor, DTH: cute.Tensor, N: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        g = bidx * block + tidx
        if g < N:
            b = g // J
            j = (g - b * J).to(cutlass.Uint32)
            tb = _tables(TB, offsets)
            th_raw = _load_th(TH, g, b, nth, per_particle)
            th = th_raw
            if cutlass.const_expr(transform):
                th = _prep(prep_j, th_raw, rng, tb)
            {gs} = _load_g(G, g, ns)
            {ths} = _zeros(nth)
            for kk in range(NSTEP):
                k = NSTEP - 1 - kk
                {gs} {ths} = _vjp_once(step_vjp_j, vec, rng, J, ns, nth, nc, _hist(H, g, k, ns), th, KEYS, TAB,
                                       tb, b, j, k, ({gs}), ({ths}))
            _store_bwd(DX, DTH, g, ({gs}), ({ths}), acc, prep_vjp_j, th_raw, transform, rng, tb)

    @cute.jit
    def launch(stream, H: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor,
               TB: cute.Tensor, G: cute.Tensor, DX: cute.Tensor, DTH: cute.Tensor):
        N = cute.size(G, mode=[0])
        kernel(H, TH, KEYS, TAB, TB, G, DX, DTH, N).launch(grid=[(N + block - 1) // block, 1, 1],
                                                       block=[block, 1, 1], stream=stream)

    return launch
"""


@functools.cache
def _generated_dir() -> str:
    """Return a directory that only this user can write, removed at exit."""
    path = tempfile.mkdtemp(prefix="pypomp_cute_")
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    return path


def _import_generated(src: str) -> Any:
    """Import generated kernel source `src` as a module, cached by its hash."""
    path = os.path.join(
        _generated_dir(), f"pypomp_cute_{hashlib.md5(src.encode()).hexdigest()[:12]}.py"
    )
    if not os.path.exists(path):
        with open(path, "w") as fh:
            fh.write(src)
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _state_names(prefix: str, n: int) -> str:
    """Return "p0, p1, ..., " for n register names in generated kernel code."""
    return "".join(f"{prefix}{i}, " for i in range(n))


def _build_launcher(kind: str, spec: _Spec, J: int, transform: bool) -> Any:
    """Return the launcher of the "fwd", "save", "bwd", "dm" or "per" kernel for J
    particles."""
    gs, ths = _state_names("g", spec.ns), _state_names("a", spec.nth)
    src = (
        {
            "fwd": _SRC,
            "save": _SRC_SAVE,
            "bwd": _SRC_BWD,
            "dm": _SRC_DM,
            "per": _SRC_PER,
        }[kind]
        .replace("{xs}", _state_names("x", spec.ns).rstrip())
        .replace("{gs} {ths}", gs + ths.rstrip())
        .replace("{gs}", gs.rstrip())
        .replace("{ths}", ths.rstrip())
    )
    return _import_generated(src).make(spec, J, transform, _BLOCK)


def _broadcast_unbatched(
    axis_size: int, in_batched: Sequence[bool], args: Sequence[jax.Array]
) -> list[jax.Array]:
    """Give each unbatched argument of a ``custom_vmap`` rule the batch axis."""
    return [
        a if bat else jnp.broadcast_to(a, (axis_size,) + a.shape)
        for a, bat in zip(args, in_batched, strict=True)
    ]


def _is_threefry(key: jax.Array) -> bool:
    """Return whether `key` draws as the kernel's "threefry" generator does: with
    JAX's threefry, in its partitionable layout."""
    threefry = jax.random.key_impl(jax.random.key(0, impl="threefry2x32"))
    return bool(jax.config.jax_threefry_partitionable) and (
        jax.random.key_impl(key) == threefry
    )


def _perturbation(
    rng: str, key: jax.Array, J: int, d: int, dtype: Any = jnp.float32
) -> jax.Array:
    """Return the (J, d) standard normal draws the iterated filtering kernel adds,
    scaled, to the parameters: with threefry, XLA's ``normal(key, (J, d))``."""
    if rng == "threefry":
        return jax.random.normal(key, (J, d), dtype)
    return jax.vmap(lambda k: jax.random.normal(k, (d,), dtype))(
        jax.random.split(key, J)
    )


def _hoist(cj: ct.ClosedJaxpr, shared: list[bool]) -> ct._Hoisted | None:
    """Return `cj` split by :func:`translate._hoist`, if the kernel's part
    translates, else None."""
    h = ct._hoist(cj, shared)
    return None if h is None or ct._untranslatable(h.rest) else h


def _make_rprocess_interp(
    model: Pomp,
    traced: _Traced,
    per_particle: bool,
    rng: str,
    dm_j: ct.ClosedJaxpr | None,
) -> KernelRProc:
    """Return a CuTe replacement for the model's ``rprocess_interp``.

    With `per_particle`, every particle has its own parameters, as in iterated
    filtering (``mechanics_per_interp``); otherwise every filter has one parameter
    vector (``mechanics_pf_interp``). The kernel draws with `rng`, "threefry" or
    "philox". Given the measurement density's jaxpr `dm_j`, it also carries
    ``with_dmeasure``, which evaluates the density at the end of the interval in
    the same kernel, while each particle's state is still in registers.
    """
    rp, dm = model.rproc, model.dmeas
    assert rp.nstep is not None
    xla = rp.mechanics_per_interp if per_particle else rp.mechanics_pf_interp
    dmeas_xla = None
    if dm is not None:
        dmeas_xla = dm.mechanics_per if per_particle else dm.mechanics_pf
    step_j, prep_j, vec, step_vjp_j, prep_vjp_j = traced
    # What a filter's particles share runs once per filter and step on XLA: the
    # covariates, times and observation, and the parameters unless every particle
    # has its own.
    hs = _hoist(step_j, [False, not per_particle, False, True, True, True])
    hd = None
    if dm_j is not None:
        hd = _hoist(dm_j, [True, False, not per_particle, True, True])
    jaxprs = [step_j, prep_j, step_vjp_j, prep_vjp_j]
    jaxprs += [h.rest for h in (hs, hd) if h is not None]
    if dm_j is not None:
        jaxprs.append(dm_j)
    data, offsets = ct._pack_tables(ct._runtime_tables(jaxprs))
    spec = _Spec(
        step_j=step_j if hs is None else hs.rest,
        prep_j=prep_j,
        step_vjp_j=step_vjp_j,
        prep_vjp_j=prep_vjp_j,
        vec=vec,
        ns=len(rp.statenames),
        nth=len(rp.param_names),
        nc=len(rp.covar_names),
        nstep=int(rp.nstep),
        acc=(),
        per_particle=per_particle,
        rng=rng,
        dm_j=dm_j if hd is None else hd.rest,
        ny=0 if dm is None else len(dm.y_names),
        ncd=0 if dm is None else len(dm.covar_names),
        noise_j=_trace_noise(len(rp.param_names)),
        offsets=offsets,
        res=() if hs is None else hs.avals,
        dm_res=() if hd is None else hd.avals,
    )
    ns, nth, nc, nstep = spec.ns, spec.nth, spec.nc, spec.nstep
    ny, ncd = spec.ny, spec.ncd
    launchers: dict[tuple[str, int, bool, tuple[int, ...]], Any] = {}

    def launcher(kind: str, J: int, transform: bool, acc: tuple[int, ...]) -> Any:
        k = (kind, J, transform, acc)
        if k not in launchers:
            launchers[k] = _build_launcher(kind, spec._replace(acc=acc), J, transform)
        return launchers[k]

    def flatten(th: jax.Array, sk: jax.Array, tab: jax.Array, B: int, J: int):
        TH = th.reshape(B * J if per_particle else B, nth)
        return TH, sk.reshape(B, nstep, 2), tab.reshape(B, nstep, tab.shape[-1])

    def make_core(transform: bool, acc: tuple[int, ...]) -> _Cores:
        # Shapes: Xp (..., J, ns); th (..., d), or (..., J, d) per particle;
        # sk (..., nstep, 2); tab (..., nstep, 2 + nc + hoisted).
        @jax.custom_batching.custom_vmap
        def core(Xp, th, sk, tab):
            lead, J = Xp.shape[:-2], Xp.shape[-2]
            B = math.prod(lead)
            X = Xp.reshape(B * J, ns)
            out = cutlass_call(
                launcher("fwd", J, transform, acc),
                output_shape_dtype=jax.ShapeDtypeStruct(X.shape, jnp.float32),
            )(X, *flatten(th, sk, tab, B, J), jnp.asarray(data))
            return out.reshape(lead + (J, ns))

        @core.def_vmap
        def _rule(axis_size, in_batched, *args):
            return core(*_broadcast_unbatched(axis_size, in_batched, args)), True

        @jax.custom_batching.custom_vmap
        def core_save(Xp, th, sk, tab):
            # As core, plus each step's input state: H (..., J, nstep * ns).
            lead, J = Xp.shape[:-2], Xp.shape[-2]
            B = math.prod(lead)
            X = Xp.reshape(B * J, ns)
            out, H = cutlass_call(
                launcher("save", J, transform, acc),
                output_shape_dtype=(
                    jax.ShapeDtypeStruct(X.shape, jnp.float32),
                    jax.ShapeDtypeStruct((B * J, nstep * ns), jnp.float32),
                ),
            )(X, *flatten(th, sk, tab, B, J), jnp.asarray(data))
            return out.reshape(lead + (J, ns)), H.reshape(lead + (J, nstep * ns))

        @core_save.def_vmap
        def _rule_save(axis_size, in_batched, *args):
            return core_save(*_broadcast_unbatched(axis_size, in_batched, args)), (
                True,
                True,
            )

        @jax.custom_batching.custom_vmap
        def core_bwd(H, th, sk, tab, gout):
            lead, J = gout.shape[:-2], gout.shape[-2]
            B = math.prod(lead)
            G = gout.reshape(B * J, ns)
            TH, SK, TAB = flatten(th, sk, tab, B, J)
            dX, dTH = cutlass_call(
                launcher("bwd", J, transform, acc),
                output_shape_dtype=(
                    jax.ShapeDtypeStruct(G.shape, jnp.float32),
                    jax.ShapeDtypeStruct((B * J, nth), jnp.float32),
                ),
            )(H.reshape(B * J, nstep * ns), TH, SK, TAB, jnp.asarray(data), G)
            dTH = dTH.reshape(lead + (J, nth))
            return dX.reshape(lead + (J, ns)), (dTH if per_particle else dTH.sum(-2))

        @core_bwd.def_vmap
        def _rule_bwd(axis_size, in_batched, *args):
            return core_bwd(*_broadcast_unbatched(axis_size, in_batched, args)), (
                True,
                True,
            )

        @jax.custom_vjp
        def diff_core(Xp, th, sk, tab):
            return core(Xp, th, sk, tab)

        def diff_fwd(Xp, th, sk, tab):
            out, H = core_save(Xp, th, sk, tab)
            return out, (H, th, sk, tab)

        def diff_bwd(res, gout):
            H, th, sk, tab = res
            dX, dth = core_bwd(H, th, sk, tab, gout)
            return dX, dth, np.zeros(sk.shape, jax.dtypes.float0), jnp.zeros_like(tab)

        diff_core.defvjp(diff_fwd, diff_bwd)
        if dm_j is None or dmeas_xla is None:
            return diff_core, None, None

        # obs (..., ny + 1 + ncd + hoisted): the observation, its time, its
        # covariates and the values hoisted out of the density.
        @jax.custom_batching.custom_vmap
        def core_dm(Xp, th, sk, tab, obs):
            lead, J = Xp.shape[:-2], Xp.shape[-2]
            B = math.prod(lead)
            X = Xp.reshape(B * J, ns)
            out, w = cutlass_call(
                launcher("dm", J, transform, acc),
                output_shape_dtype=(
                    jax.ShapeDtypeStruct(X.shape, jnp.float32),
                    jax.ShapeDtypeStruct((B * J,), jnp.float32),
                ),
            )(
                X,
                *flatten(th, sk, tab, B, J),
                jnp.asarray(data),
                obs.reshape(B, obs.shape[-1]),
            )
            return out.reshape(lead + (J, ns)), w.reshape(lead + (J,))

        @core_dm.def_vmap
        def _rule_dm(axis_size, in_batched, *args):
            return core_dm(*_broadcast_unbatched(axis_size, in_batched, args)), (
                True,
                True,
            )

        def dmeas(out, th, obs):
            assert dmeas_xla is not None
            cov = obs[ny + 1 :] if ncd else None
            return dmeas_xla(obs[:ny], out, th, cov, obs[ny], transform)

        # Derivatives go through the unfused kernels and XLA's density.
        @jax.custom_vjp
        def diff_dm(Xp, th, sk, tab, obs):
            return core_dm(Xp, th, sk, tab, obs)

        def dm_fwd(Xp, th, sk, tab, obs):
            out, H = core_save(Xp, th, sk, tab)
            w, pull = jax.vjp(lambda o, t: dmeas(o, t, obs), out, th)
            return (out, w), (H, th, sk, tab, obs, pull)

        def dm_bwd(res, g):
            H, th, sk, tab, obs, pull = res
            gout, gw = g
            gx_dm, gth_dm = pull(gw)
            dX, dth = core_bwd(H, th, sk, tab, gout + gx_dm)
            return (
                dX,
                dth + gth_dm,
                np.zeros(sk.shape, jax.dtypes.float0),
                jnp.zeros_like(tab),
                jnp.zeros_like(obs),
            )

        diff_dm.defvjp(dm_fwd, dm_bwd)
        if not per_particle:
            return diff_core, diff_dm, None

        # Iterated filtering's step, which no one differentiates: anc (..., J) are the
        # ancestors of the last resampling, pk (..., 2) the perturbation key's data,
        # sig (..., d) the random walk's standard deviations, and Xs and ths spent
        # buffers shaped as Xp and th that the new states and parameters overwrite.
        @jax.custom_batching.custom_vmap
        def core_per(Xp, th, anc, pk, sig, sk, tab, obs, Xs, ths):
            lead, J = Xp.shape[:-2], Xp.shape[-2]
            B = math.prod(lead)
            out, tho, w = cutlass_call(
                launcher("per", J, transform, acc),
                output_shape_dtype=(
                    jax.ShapeDtypeStruct((B * J, ns), jnp.float32),
                    jax.ShapeDtypeStruct((B * J, nth), jnp.float32),
                    jax.ShapeDtypeStruct((B * J,), jnp.float32),
                ),
                input_output_aliases={9: 0, 10: 1},
            )(
                Xp.reshape(B * J, ns),
                th.reshape(B * J, nth),
                anc.reshape(B * J),
                pk.reshape(B, 2),
                sig.reshape(B, nth),
                sk.reshape(B, nstep, 2),
                tab.reshape(B, nstep, tab.shape[-1]),
                jnp.asarray(data),
                obs.reshape(B, obs.shape[-1]),
                Xs.reshape(B * J, ns),
                ths.reshape(B * J, nth),
            )
            return (
                out.reshape(lead + (J, ns)),
                tho.reshape(lead + (J, nth)),
                w.reshape(lead + (J,)),
            )

        @core_per.def_vmap
        def _rule_per(axis_size, in_batched, *args):
            return core_per(*_broadcast_unbatched(axis_size, in_batched, args)), (
                True,
                True,
                True,
            )

        return diff_core, diff_dm, core_per

    cores: dict[tuple[bool, tuple[int, ...]], _Cores] = {}

    def natural(theta_: jax.Array, should_trans: bool) -> jax.Array:
        """Return a filter's parameters on the natural scale."""
        th = jnp.asarray(theta_, jnp.float32)
        return jaxpr_as_fun(prep_j)(th)[0] if should_trans else th

    def shared_theta(theta_: jax.Array, should_trans: bool) -> list[jax.Array]:
        """Return the parameters as the hoisted parts take them: none if every
        particle has its own."""
        return [] if per_particle else [natural(theta_, should_trans)]

    def kernel_inputs(
        keys: jax.Array,
        theta_: jax.Array,
        covars_extended: jax.Array | None,
        dt_array_extended: jax.Array,
        t: jax.Array,
        t_idx: jax.Array,
        accumvars_: tuple[int, ...] | None,
        should_trans: bool,
    ) -> tuple[_Cores, jax.Array, jax.Array]:
        """Return the kernels for this call, the substep keys and the step table:
        each step's time, step size, covariates and the values hoisted out of it."""
        key = keys[0] if jnp.ndim(keys) != 0 else keys
        subkeys, ts, dts, covs = [], [], [], []
        tk = t
        for k in range(nstep):  # the same key chain and clock as _time_interp
            key, sub = jax.random.split(key)
            subkeys.append(jax.random.key_data(sub))
            dt = dt_array_extended[t_idx + k]
            ts.append(tk)
            dts.append(dt)
            if nc:
                assert covars_extended is not None
                covs.append(covars_extended[t_idx + k])
            tk = tk + dt
        cols = [
            jnp.stack([jnp.asarray(ts, jnp.float32), jnp.asarray(dts, jnp.float32)], 1)
        ]
        if nc:
            cols.append(jnp.stack(covs).astype(jnp.float32))
        if hs is not None:
            f32 = jnp.float32
            cov = cols[1] if nc else jnp.zeros((nstep, 0), f32)
            th, step_shared = shared_theta(theta_, should_trans), hs.shared
            res = jax.vmap(lambda c, t, dt: jaxpr_as_fun(step_shared)(*th, c, t, dt))(
                cov, cols[0][:, 0], cols[0][:, 1]
            )
            cols.append(_as_columns(res, 1))
        tab = jnp.concatenate(cols, axis=1)
        acc = tuple(sorted({int(a) for a in (accumvars_ or ())}))
        if (bool(should_trans), acc) not in cores:
            cores[bool(should_trans), acc] = make_core(bool(should_trans), acc)
        return cores[bool(should_trans), acc], jnp.stack(subkeys), tab

    def on_kernel(X_: jax.Array, keys: jax.Array) -> bool:
        """Return whether the kernel can run these inputs."""
        key = keys[0] if jnp.ndim(keys) != 0 else keys
        # The kernel is float32 and reads keys of two 32-bit words. Its threefry
        # draws are XLA's only for threefry keys, so other keys (rbg, Philox) stay on
        # XLA then, as do float64 particles.
        return (
            X_.dtype == jnp.float32
            and jax.random.key_data(key).shape == (2,)
            and not (rng == "threefry" and not _is_threefry(key))
        )

    def rprocess_interp(
        X_: jax.Array,
        theta_: jax.Array,
        keys: jax.Array,
        covars_extended: jax.Array | None,
        dt_array_extended: jax.Array,
        t: jax.Array,
        t_idx: jax.Array,
        nstep_dynamic: jax.Array,
        accumvars_: tuple[int, ...] | None,
        should_trans: bool,
    ) -> tuple[jax.Array, jax.Array]:
        """Advance the particles over one observation interval with the kernel."""
        if not on_kernel(X_, keys):
            return xla(
                X_,
                theta_,
                keys,
                covars_extended,
                dt_array_extended,
                t,
                t_idx,
                nstep_dynamic,
                accumvars_,
                should_trans,
            )
        (core, *_), sk, tab = kernel_inputs(
            keys,
            theta_,
            covars_extended,
            dt_array_extended,
            t,
            t_idx,
            accumvars_,
            should_trans,
        )
        out = core(X_, jnp.asarray(theta_, jnp.float32), sk, tab)
        return out, t_idx + nstep

    def with_dmeasure(
        X_: jax.Array,
        theta_: jax.Array,
        keys: jax.Array,
        covars_extended: jax.Array | None,
        dt_array_extended: jax.Array,
        t: jax.Array,
        t_idx: jax.Array,
        nstep_dynamic: jax.Array,
        accumvars_: tuple[int, ...] | None,
        should_trans: bool,
        y: jax.Array,
        t_obs: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """As ``rprocess_interp``, plus the measurement log-density of observation `y`
        at time `t_obs` for each particle at the end of the interval."""
        assert dmeas_xla is not None
        args = (covars_extended, dt_array_extended, t, t_idx, accumvars_, should_trans)
        if not on_kernel(X_, keys):
            X, t_end = rprocess_interp(
                X_, theta_, keys, *args[:4], nstep_dynamic, *args[4:]
            )
            cov = None if covars_extended is None else covars_extended[t_end]
            return X, t_end, dmeas_xla(y, X, theta_, cov, t_obs, should_trans)
        (_, core_dm, _), sk, tab = kernel_inputs(keys, theta_, *args)
        assert core_dm is not None
        obs = obs_row(y, t_obs, covars_extended, t_idx + nstep, theta_, should_trans)
        out, w = core_dm(X_, jnp.asarray(theta_, jnp.float32), sk, tab, obs)
        return out, t_idx + nstep, w

    def obs_row(
        y: jax.Array,
        t_obs: jax.Array,
        covars_extended: jax.Array | None,
        t_end: Any,
        theta_: jax.Array,
        should_trans: bool,
    ) -> jax.Array:
        """Return the kernel's row for an observation: y, its time, its covariates
        and the values hoisted out of the density."""
        f32 = jnp.float32
        cols = [jnp.asarray(y, f32), jnp.asarray(t_obs, f32)[None]]
        if ncd:
            assert covars_extended is not None
            cols.append(covars_extended[t_end].astype(f32))
        if hd is not None:
            cov = cols[2] if ncd else jnp.zeros(0, f32)
            th = shared_theta(theta_, should_trans)
            res = jaxpr_as_fun(hd.shared)(cols[0], *th, cov, cols[1][0])
            cols.append(_as_columns(res, 0))
        return jnp.concatenate(cols)

    def perturb_step(
        X_: jax.Array,
        theta_: jax.Array,
        ancestors: jax.Array,
        key_perturb: jax.Array,
        sigmas: jax.Array,
        keys: jax.Array,
        covars_extended: jax.Array | None,
        dt_array_extended: jax.Array,
        t: jax.Array,
        t_idx: jax.Array,
        nstep_dynamic: jax.Array,
        accumvars_: tuple[int, ...] | None,
        should_trans: bool,
        y: jax.Array,
        t_obs: jax.Array,
        scratch: tuple[jax.Array, jax.Array] | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """One interval of iterated filtering in one kernel.

        Particle j starts from particle ``ancestors[j]`` of `X_` and `theta_`, which
        the last resampling picked, and its parameters move by `sigmas` times row j
        of standard normal draws, :func:`_perturbation` of `key_perturb`. Returns the
        states
        at the end of the interval, the perturbed parameters, the time index and
        the measurement log-density of observation `y` at time `t_obs`.

        The states and parameters are written over `scratch`, buffers shaped as
        `X_` and `theta_` whose values are no longer needed, if given. A loop that
        swaps them with its carry every step, two steps per iteration, then keeps
        XLA from copying the kernel's outputs into the carry.
        """
        assert dmeas_xla is not None
        J = X_.shape[-2]
        if not (on_kernel(X_, keys) and on_kernel(X_, key_perturb)):
            th = theta_[ancestors] + sigmas * _perturbation(
                rng, key_perturb, J, nth, theta_.dtype
            )
            X, t_end = rprocess_interp(
                X_[ancestors],
                th,
                keys,
                covars_extended,
                dt_array_extended,
                t,
                t_idx,
                nstep_dynamic,
                accumvars_,
                should_trans,
            )
            cov = None if covars_extended is None else covars_extended[t_end]
            return X, th, t_end, dmeas_xla(y, X, th, cov, t_obs, should_trans)
        args = (covars_extended, dt_array_extended, t, t_idx, accumvars_, should_trans)
        (_, _, core_per), sk, tab = kernel_inputs(keys, theta_, *args)
        assert core_per is not None
        if scratch is None:
            scratch = (
                jnp.zeros(X_.shape, jnp.float32),
                jnp.zeros(theta_.shape, jnp.float32),
            )
        out, tho, w = core_per(
            X_,
            jnp.asarray(theta_, jnp.float32),
            jnp.asarray(ancestors, jnp.int32),
            jax.random.key_data(key_perturb),
            jnp.asarray(sigmas, jnp.float32),
            sk,
            tab,
            obs_row(y, t_obs, covars_extended, t_idx + nstep, theta_, should_trans),
            *scratch,
        )
        return out, tho, t_idx + nstep, w

    return KernelRProc(
        kernel=rprocess_interp,
        xla=xla,
        resample=_resample_kernel,
        with_dmeasure=None if dm_j is None else with_dmeasure,
        dmeasure=dmeas_xla,
        perturb_step=perturb_step if per_particle and dm_j is not None else None,
    )
