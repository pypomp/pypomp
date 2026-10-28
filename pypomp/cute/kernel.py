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
buffer argument (see ``translate.py``).

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

from . import KernelRProc
from . import translate as ct
from .analysis import _Traced
from .resample import resample as _resample_kernel

if TYPE_CHECKING:
    from ..core.pomp import Pomp

# A cute.Tensor inside a kernel, and a CuTe DSL scalar or constant.
_Tensor = Any
_Scalar = ct._Scalar
# CUDA threads per block.
_BLOCK = 128
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
    fixed: bool
    """Whether every interval has `nstep` Euler steps; otherwise `nstep` is the
    most any has, and each launch reads its intervals' counts."""


def _trace_noise(d: int) -> ct.ClosedJaxpr:
    """Return the jaxpr of d standard normal draws from a key's data."""
    return jax.make_jaxpr(
        lambda kd: jax.random.normal(jax.random.wrap_key_data(kd), (d,), jnp.float32)
    )(jnp.zeros(2, jnp.uint32))


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
) -> tuple[_Scalar, ...]:
    """Return the state after Euler step k; k may be a runtime loop index."""
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
) -> _Scalar:
    """Return the measurement log-density of state `xs`; OBS[b] holds filter b's
    observation, then the observation time and its covariates."""
    y = np.empty(ny, dtype=object)
    for i in range(ny):
        y[i] = OBS[b, i]
    cov = np.empty(ncd, dtype=object)
    for c in range(ncd):
        cov[c] = OBS[b, ny + 1 + c]
    ctx = ct._Ctx(vectorized_J=None, particle=None, rng=rng, tables=tb)
    (out,) = ct._interpret(
        dm_j, [y, ct._obj(list(xs)), th, cov, ct._obj(OBS[b, ny])], ctx
    )
    return ct._as_dsl(np.asarray(out, dtype=object).reshape(-1)[0], np.float32)


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


def _nsteps(NS: _Tensor, b: _Scalar, nstep: int, fixed: bool) -> _Scalar:
    """Return the number of Euler steps in filter b's interval: `nstep` if every
    interval has that many, so the loop's trip count is known at compile time."""
    return nstep if fixed else NS[b]


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
from pypomp.cute.kernel import _nsteps, _load, _prep, _step_once, _store, _tables


def make(spec, J, transform, block):
    step_j, prep_j, step_vjp_j, prep_vjp_j, vec, ns, nth, nc, NSTEP, acc, per_particle, rng, dm_j, ny, ncd, noise_j, offsets, fixed = spec
    @cute.kernel
    def kernel(X: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor, TB: cute.Tensor, NS: cute.Tensor,
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
            for k in range(_nsteps(NS, b, NSTEP, fixed)):
                {xs} = _step_once(step_j, vec, rng, J, ns, nc, ({xs}), th, KEYS, TAB, tb, b, j, k)
            _store(OUT, g, ({xs}))

    @cute.jit
    def launch(stream, X: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor,
               TB: cute.Tensor, NS: cute.Tensor, OUT: cute.Tensor):
        N = cute.size(X, mode=[0])
        kernel(X, TH, KEYS, TAB, TB, NS, OUT, N).launch(grid=[(N + block - 1) // block, 1, 1],
                                                block=[block, 1, 1], stream=stream)

    return launch
"""


_SRC_SAVE = """
import cutlass
import cutlass.cute as cute
from pypomp.cute.kernel import _nsteps, _load, _prep, _step_once, _store, _save, _tables


def make(spec, J, transform, block):
    step_j, prep_j, step_vjp_j, prep_vjp_j, vec, ns, nth, nc, NSTEP, acc, per_particle, rng, dm_j, ny, ncd, noise_j, offsets, fixed = spec
    @cute.kernel
    def kernel(X: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor, TB: cute.Tensor, NS: cute.Tensor,
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
            for k in range(_nsteps(NS, b, NSTEP, fixed)):
                _save(H, g, k, ({xs}))
                {xs} = _step_once(step_j, vec, rng, J, ns, nc, ({xs}), th, KEYS, TAB, tb, b, j, k)
            _store(OUT, g, ({xs}))

    @cute.jit
    def launch(stream, X: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor,
               TB: cute.Tensor, NS: cute.Tensor, OUT: cute.Tensor, H: cute.Tensor):
        N = cute.size(X, mode=[0])
        kernel(X, TH, KEYS, TAB, TB, NS, OUT, H, N).launch(grid=[(N + block - 1) // block, 1, 1],
                                                   block=[block, 1, 1], stream=stream)

    return launch
"""


_SRC_DM = """
import cutlass
import cutlass.cute as cute
from pypomp.cute.kernel import _nsteps, _load, _prep, _step_once, _store, _dmeas_once, _tables


def make(spec, J, transform, block):
    step_j, prep_j, step_vjp_j, prep_vjp_j, vec, ns, nth, nc, NSTEP, acc, per_particle, rng, dm_j, ny, ncd, noise_j, offsets, fixed = spec
    @cute.kernel
    def kernel(X: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor, TB: cute.Tensor, NS: cute.Tensor,
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
            for k in range(_nsteps(NS, b, NSTEP, fixed)):
                {xs} = _step_once(step_j, vec, rng, J, ns, nc, ({xs}), th, KEYS, TAB, tb, b, j, k)
            _store(OUT, g, ({xs}))
            W[g] = _dmeas_once(dm_j, rng, ny, ncd, ({xs}), th, OBS, tb, b)

    @cute.jit
    def launch(stream, X: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor,
               TB: cute.Tensor, NS: cute.Tensor, OBS: cute.Tensor, OUT: cute.Tensor, W: cute.Tensor):
        N = cute.size(X, mode=[0])
        kernel(X, TH, KEYS, TAB, TB, NS, OBS, OUT, W, N).launch(grid=[(N + block - 1) // block, 1, 1],
                                                        block=[block, 1, 1], stream=stream)

    return launch
"""


_SRC_PER = """
import cutlass
import cutlass.cute as cute
from pypomp.cute.kernel import _nsteps, _load, _prep, _step_once, _store, _dmeas_once, _perturb, _tables


def make(spec, J, transform, block):
    step_j, prep_j, step_vjp_j, prep_vjp_j, vec, ns, nth, nc, NSTEP, acc, per_particle, rng, dm_j, ny, ncd, noise_j, offsets, fixed = spec
    @cute.kernel
    def kernel(X: cute.Tensor, TH: cute.Tensor, ANC: cute.Tensor, PK: cute.Tensor, SIG: cute.Tensor,
               KEYS: cute.Tensor, TAB: cute.Tensor, TB: cute.Tensor, NS: cute.Tensor, OBS: cute.Tensor, OUT: cute.Tensor,
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
            for k in range(_nsteps(NS, b, NSTEP, fixed)):
                {xs} = _step_once(step_j, vec, rng, J, ns, nc, ({xs}), th, KEYS, TAB, tb, b, j, k)
            _store(OUT, g, ({xs}))
            W[g] = _dmeas_once(dm_j, rng, ny, ncd, ({xs}), th, OBS, tb, b)

    @cute.jit
    def launch(stream, X: cute.Tensor, TH: cute.Tensor, ANC: cute.Tensor, PK: cute.Tensor,
               SIG: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor, TB: cute.Tensor, NS: cute.Tensor,
               OBS: cute.Tensor, OUT: cute.Tensor, THO: cute.Tensor, W: cute.Tensor):
        # OUT and THO are inputs aliased to outputs, so cutlass_call passes them once.
        N = cute.size(X, mode=[0])
        kernel(X, TH, ANC, PK, SIG, KEYS, TAB, TB, NS, OBS, OUT, THO, W, N).launch(
            grid=[(N + block - 1) // block, 1, 1], block=[block, 1, 1], stream=stream)

    return launch
"""


_SRC_BWD = """
import cutlass
import cutlass.cute as cute
from pypomp.cute.kernel import _nsteps, _load_th, _prep, _hist, _load_g, _zeros, _vjp_once, _store_bwd, _tables


def make(spec, J, transform, block):
    step_j, prep_j, step_vjp_j, prep_vjp_j, vec, ns, nth, nc, NSTEP, acc, per_particle, rng, dm_j, ny, ncd, noise_j, offsets, fixed = spec
    @cute.kernel
    def kernel(H: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor, TB: cute.Tensor, NS: cute.Tensor,
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
            n = _nsteps(NS, b, NSTEP, fixed)
            for kk in range(n):
                k = n - 1 - kk
                {gs} {ths} = _vjp_once(step_vjp_j, vec, rng, J, ns, nth, nc, _hist(H, g, k, ns), th, KEYS, TAB,
                                       tb, b, j, k, ({gs}), ({ths}))
            _store_bwd(DX, DTH, g, ({gs}), ({ths}), acc, prep_vjp_j, th_raw, transform, rng, tb)

    @cute.jit
    def launch(stream, H: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor,
               TB: cute.Tensor, NS: cute.Tensor, G: cute.Tensor, DX: cute.Tensor, DTH: cute.Tensor):
        N = cute.size(G, mode=[0])
        kernel(H, TH, KEYS, TAB, TB, NS, G, DX, DTH, N).launch(grid=[(N + block - 1) // block, 1, 1],
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


def _float0(x: jax.Array) -> np.ndarray:
    """Return the zero cotangent of integer array `x`."""
    return np.zeros(jnp.shape(x), jax.dtypes.float0)


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
    xla = rp.mechanics_per_interp if per_particle else rp.mechanics_pf_interp
    dmeas_xla = None
    if dm is not None:
        dmeas_xla = dm.mechanics_per if per_particle else dm.mechanics_pf
    step_j, prep_j, vec, step_vjp_j, prep_vjp_j = traced
    jaxprs = [step_j, prep_j, step_vjp_j, prep_vjp_j]
    if dm_j is not None:
        jaxprs.append(dm_j)
    data, offsets = ct._pack_tables(ct._runtime_tables(jaxprs))
    spec = _Spec(
        step_j=step_j,
        prep_j=prep_j,
        step_vjp_j=step_vjp_j,
        prep_vjp_j=prep_vjp_j,
        vec=vec,
        ns=len(rp.statenames),
        nth=len(rp.param_names),
        nc=len(rp.covar_names),
        nstep=int(rp.nstep or model._max_steps_per_interval),
        acc=(),
        per_particle=per_particle,
        rng=rng,
        dm_j=dm_j,
        ny=0 if dm is None else len(dm.y_names),
        ncd=0 if dm is None else len(dm.covar_names),
        noise_j=_trace_noise(len(rp.param_names)),
        offsets=offsets,
        fixed=rp.nstep is not None,
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
        return TH, sk.reshape(B, nstep, 2), tab.reshape(B, nstep, 2 + nc)

    def steps(n: jax.Array, lead: tuple[int, ...]) -> jax.Array:
        """Return the kernel's per-filter step counts from `n`, of shape `lead` or
        one count for all."""
        return jnp.broadcast_to(n, lead).reshape(-1).astype(jnp.int32)

    def make_core(transform: bool, acc: tuple[int, ...]) -> _Cores:
        # Shapes: Xp (..., J, ns); th (..., d), or (..., J, d) per particle;
        # sk (..., nstep, 2); tab (..., nstep, 2 + nc); n (...,), the step counts.
        @jax.custom_batching.custom_vmap
        def core(Xp, th, sk, tab, n):
            lead, J = Xp.shape[:-2], Xp.shape[-2]
            B = math.prod(lead)
            X = Xp.reshape(B * J, ns)
            out = cutlass_call(
                launcher("fwd", J, transform, acc),
                output_shape_dtype=jax.ShapeDtypeStruct(X.shape, jnp.float32),
            )(X, *flatten(th, sk, tab, B, J), jnp.asarray(data), steps(n, lead))
            return out.reshape(lead + (J, ns))

        @core.def_vmap
        def _rule(axis_size, in_batched, *args):
            return core(*_broadcast_unbatched(axis_size, in_batched, args)), True

        @jax.custom_batching.custom_vmap
        def core_save(Xp, th, sk, tab, n):
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
            )(X, *flatten(th, sk, tab, B, J), jnp.asarray(data), steps(n, lead))
            return out.reshape(lead + (J, ns)), H.reshape(lead + (J, nstep * ns))

        @core_save.def_vmap
        def _rule_save(axis_size, in_batched, *args):
            return core_save(*_broadcast_unbatched(axis_size, in_batched, args)), (
                True,
                True,
            )

        @jax.custom_batching.custom_vmap
        def core_bwd(H, th, sk, tab, n, gout):
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
            )(
                H.reshape(B * J, nstep * ns),
                TH,
                SK,
                TAB,
                jnp.asarray(data),
                steps(n, lead),
                G,
            )
            dTH = dTH.reshape(lead + (J, nth))
            return dX.reshape(lead + (J, ns)), (dTH if per_particle else dTH.sum(-2))

        @core_bwd.def_vmap
        def _rule_bwd(axis_size, in_batched, *args):
            return core_bwd(*_broadcast_unbatched(axis_size, in_batched, args)), (
                True,
                True,
            )

        @jax.custom_vjp
        def diff_core(Xp, th, sk, tab, n):
            return core(Xp, th, sk, tab, n)

        def diff_fwd(Xp, th, sk, tab, n):
            out, H = core_save(Xp, th, sk, tab, n)
            return out, (H, th, sk, tab, n)

        def diff_bwd(res, gout):
            H, th, sk, tab, n = res
            dX, dth = core_bwd(H, th, sk, tab, n, gout)
            return dX, dth, _float0(sk), jnp.zeros_like(tab), _float0(n)

        diff_core.defvjp(diff_fwd, diff_bwd)
        if dm_j is None or dmeas_xla is None:
            return diff_core, None, None

        # obs (..., ny + 1 + ncd): the observation, its time and its covariates.
        @jax.custom_batching.custom_vmap
        def core_dm(Xp, th, sk, tab, n, obs):
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
                steps(n, lead),
                obs.reshape(B, ny + 1 + ncd),
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
        def diff_dm(Xp, th, sk, tab, n, obs):
            return core_dm(Xp, th, sk, tab, n, obs)

        def dm_fwd(Xp, th, sk, tab, n, obs):
            out, H = core_save(Xp, th, sk, tab, n)
            w, pull = jax.vjp(lambda o, t: dmeas(o, t, obs), out, th)
            return (out, w), (H, th, sk, tab, n, obs, pull)

        def dm_bwd(res, g):
            H, th, sk, tab, n, obs, pull = res
            gout, gw = g
            gx_dm, gth_dm = pull(gw)
            dX, dth = core_bwd(H, th, sk, tab, n, gout + gx_dm)
            return (
                dX,
                dth + gth_dm,
                _float0(sk),
                jnp.zeros_like(tab),
                _float0(n),
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
        def core_per(Xp, th, anc, pk, sig, sk, tab, n, obs, Xs, ths):
            lead, J = Xp.shape[:-2], Xp.shape[-2]
            B = math.prod(lead)
            out, tho, w = cutlass_call(
                launcher("per", J, transform, acc),
                output_shape_dtype=(
                    jax.ShapeDtypeStruct((B * J, ns), jnp.float32),
                    jax.ShapeDtypeStruct((B * J, nth), jnp.float32),
                    jax.ShapeDtypeStruct((B * J,), jnp.float32),
                ),
                input_output_aliases={10: 0, 11: 1},
            )(
                Xp.reshape(B * J, ns),
                th.reshape(B * J, nth),
                anc.reshape(B * J),
                pk.reshape(B, 2),
                sig.reshape(B, nth),
                sk.reshape(B, nstep, 2),
                tab.reshape(B, nstep, 2 + nc),
                jnp.asarray(data),
                steps(n, lead),
                obs.reshape(B, ny + 1 + ncd),
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

    def kernel_inputs(
        keys: jax.Array,
        covars_extended: jax.Array | None,
        dt_array_extended: jax.Array,
        t: jax.Array,
        t_idx: jax.Array,
        accumvars_: tuple[int, ...] | None,
        should_trans: bool,
    ) -> tuple[_Cores, jax.Array, jax.Array]:
        """Return the kernels for this call, the substep keys and the step table."""
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
        tab = jnp.concatenate(cols, axis=1)
        acc = tuple(sorted({int(a) for a in (accumvars_ or ())}))
        if (bool(should_trans), acc) not in cores:
            cores[bool(should_trans), acc] = make_core(bool(should_trans), acc)
        return cores[bool(should_trans), acc], jnp.stack(subkeys), tab

    def steps_of(nstep_dynamic: Any) -> Any:
        """Return this interval's number of Euler steps: `nstep` if fixed, so the
        time index stays static where XLA's would."""
        return nstep if spec.fixed else jnp.asarray(nstep_dynamic, jnp.int32)

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
            keys, covars_extended, dt_array_extended, t, t_idx, accumvars_, should_trans
        )
        n = steps_of(nstep_dynamic)
        out = core(
            X_, jnp.asarray(theta_, jnp.float32), sk, tab, jnp.asarray(n, jnp.int32)
        )
        return out, t_idx + n

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
        (_, core_dm, _), sk, tab = kernel_inputs(keys, *args)
        assert core_dm is not None
        n = steps_of(nstep_dynamic)
        obs = obs_row(y, t_obs, covars_extended, t_idx + n)
        out, w = core_dm(
            X_,
            jnp.asarray(theta_, jnp.float32),
            sk,
            tab,
            jnp.asarray(n, jnp.int32),
            obs,
        )
        return out, t_idx + n, w

    def obs_row(
        y: jax.Array, t_obs: jax.Array, covars_extended: jax.Array | None, t_end: Any
    ) -> jax.Array:
        """Return the kernel's row for an observation: y, its time, its covariates."""
        cols = [jnp.asarray(y, jnp.float32), jnp.asarray(t_obs, jnp.float32)[None]]
        if ncd:
            assert covars_extended is not None
            cols.append(covars_extended[t_end].astype(jnp.float32))
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
        (_, _, core_per), sk, tab = kernel_inputs(keys, *args)
        assert core_per is not None
        n = steps_of(nstep_dynamic)
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
            jnp.asarray(n, jnp.int32),
            obs_row(y, t_obs, covars_extended, t_idx + n),
            *scratch,
        )
        return out, tho, t_idx + n, w

    return KernelRProc(
        kernel=rprocess_interp,
        xla=xla,
        resample=_resample_kernel,
        with_dmeasure=None if dm_j is None else with_dmeasure,
        dmeasure=dmeas_xla,
        perturb_step=perturb_step if per_particle and dm_j is not None else None,
    )
