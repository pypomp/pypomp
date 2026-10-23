"""One CuTe kernel per observation interval, in place of pypomp's ``rprocess_interp``.

Each CUDA thread owns one particle of one filter, keeps its state in registers and
runs the interval's Euler steps with the user's ``rproc``, translated from its jaxpr
by ``translate.py``. Every filter of a vmapped call (parameter sets, replicates,
iterated filtering searches) goes into the same launch: a ``custom_vmap`` rule folds
each batch dimension into the kernel's particle axis.

The rest of the particle filter is pypomp's own. The substep keys are split as
``_time_interp`` splits them, and the kernel draws from them with Philox4x32-10.

Gradients go through ``jax.custom_vjp``. When JAX differentiates through an interval,
the forward kernel also stores each Euler step's input state (nstep x ns floats per
particle, for one interval at a time, since pypomp checkpoints each observation). A
second kernel runs the vector-Jacobian product (VJP) of the user's step, from
``jax.vjp`` through the translator, over the steps in reverse. It regenerates the
Philox numbers instead of storing them.
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

from . import translate as ct

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
    xs = [X[s, g] for s in range(ns)]
    for a in acc:
        xs[a] = cutlass.Float32(0.0)
    return tuple(xs), _load_th(TH, g, b, nth, per_particle)


def _prep(prep_j: ct.ClosedJaxpr, th: np.ndarray) -> np.ndarray:
    """Return the parameters `th` moved from the estimation to the natural scale."""
    (out,) = ct._interpret(prep_j, [th], ct._Ctx(vectorized_J=None, particle=None))
    return np.array([ct._as_dsl(v, np.float32) for v in out], dtype=object)


def _step_keys(
    vec: bool, KEYS: _Tensor, b: _Scalar, j: _Scalar, k: _Scalar
) -> tuple[_Scalar, _Scalar]:
    """Return particle j's key for step k of filter b, as pypomp derives it."""
    k0, k1 = KEYS[b, k, 0], KEYS[b, k, 1]
    if vec:
        return k0, k1
    return ct._key_derive(k0, k1, j)  # split(substep_key, J)[j]


def _step_once(
    step_j: ct.ClosedJaxpr,
    vec: bool,
    J: int,
    ns: int,
    nc: int,
    xs: tuple[_Scalar, ...],
    th: np.ndarray,
    KEYS: _Tensor,
    TAB: _Tensor,
    b: _Scalar,
    j: _Scalar,
    k: _Scalar,
) -> tuple[_Scalar, ...]:
    """Return the state after Euler step k; k may be a runtime loop index."""
    ctx = ct._Ctx(vectorized_J=J if vec else None, particle=j)
    k0, k1 = _step_keys(vec, KEYS, b, j, k)
    cov = np.empty(nc, dtype=object)
    for c in range(nc):
        cov[c] = TAB[k, 2 + c]
    (out,) = ct._interpret(
        step_j,
        [
            ct._obj(list(xs)),
            th,
            ct._obj([k0, k1]),
            cov,
            ct._obj(TAB[k, 0]),
            ct._obj(TAB[k, 1]),
        ],
        ctx,
    )
    return tuple(ct._as_dsl(out[s], np.float32) for s in range(ns))


def _store(OUT: _Tensor, g: _Scalar, xs: tuple[_Scalar, ...]) -> None:
    """Write particle g's state."""
    for s, x in enumerate(xs):
        OUT[s, g] = x


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
    return tuple(G[s, g] for s in range(ns))


def _zeros(n: int) -> tuple[_Scalar, ...]:
    """Return n float32 zeros."""
    return tuple(cutlass.Float32(0.0) for _ in range(n))


def _vjp_once(
    step_vjp_j: ct.ClosedJaxpr,
    vec: bool,
    J: int,
    ns: int,
    nth: int,
    nc: int,
    xs: tuple[_Scalar, ...],
    th: np.ndarray,
    KEYS: _Tensor,
    TAB: _Tensor,
    b: _Scalar,
    j: _Scalar,
    k: _Scalar,
    gs: tuple[_Scalar, ...],
    acc_th: tuple[_Scalar, ...],
) -> tuple[_Scalar, ...]:
    """Return the state cotangent before step k and the summed theta cotangent."""
    ctx = ct._Ctx(vectorized_J=J if vec else None, particle=j)
    k0, k1 = _step_keys(vec, KEYS, b, j, k)
    cov = np.empty(nc, dtype=object)
    for c in range(nc):
        cov[c] = TAB[k, 2 + c]
    gx, gth = ct._interpret(
        step_vjp_j,
        [
            ct._obj(list(xs)),
            th,
            ct._obj([k0, k1]),
            cov,
            ct._obj(TAB[k, 0]),
            ct._obj(TAB[k, 1]),
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
) -> None:
    """Write particle g's input-state and parameter cotangents."""
    for s, v in enumerate(gx):
        # Accumulators are reset at the start of the interval, so get no gradient.
        DX[s, g] = cutlass.Float32(0.0) if s in acc else v
    if transform:
        ctx = ct._Ctx(vectorized_J=None, particle=None)
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
from pypomp.cute.kernel import _load, _prep, _step_once, _store


def make(spec, J, transform, block):
    step_j, prep_j, step_vjp_j, prep_vjp_j, vec, ns, nth, nc, NSTEP, acc, per_particle = spec
    @cute.kernel
    def kernel(X: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor, OUT: cute.Tensor,
               N: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        g = bidx * block + tidx
        if g < N:
            b = g // J
            j = (g - b * J).to(cutlass.Uint32)
            xs0, th = _load(X, TH, g, b, ns, nth, acc, per_particle)
            if cutlass.const_expr(transform):
                th = _prep(prep_j, th)
            {xs} = xs0
            for k in range(NSTEP):
                {xs} = _step_once(step_j, vec, J, ns, nc, ({xs}), th, KEYS, TAB, b, j, k)
            _store(OUT, g, ({xs}))

    @cute.jit
    def launch(stream, X: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor,
               OUT: cute.Tensor):
        N = cute.size(X, mode=[1])
        kernel(X, TH, KEYS, TAB, OUT, N).launch(grid=[(N + block - 1) // block, 1, 1],
                                                block=[block, 1, 1], stream=stream)

    return launch
"""


_SRC_SAVE = """
import cutlass
import cutlass.cute as cute
from pypomp.cute.kernel import _load, _prep, _step_once, _store, _save


def make(spec, J, transform, block):
    step_j, prep_j, step_vjp_j, prep_vjp_j, vec, ns, nth, nc, NSTEP, acc, per_particle = spec
    @cute.kernel
    def kernel(X: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor, OUT: cute.Tensor,
               H: cute.Tensor, N: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        g = bidx * block + tidx
        if g < N:
            b = g // J
            j = (g - b * J).to(cutlass.Uint32)
            xs0, th = _load(X, TH, g, b, ns, nth, acc, per_particle)
            if cutlass.const_expr(transform):
                th = _prep(prep_j, th)
            {xs} = xs0
            for k in range(NSTEP):
                _save(H, g, k, ({xs}))
                {xs} = _step_once(step_j, vec, J, ns, nc, ({xs}), th, KEYS, TAB, b, j, k)
            _store(OUT, g, ({xs}))

    @cute.jit
    def launch(stream, X: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor,
               OUT: cute.Tensor, H: cute.Tensor):
        N = cute.size(X, mode=[1])
        kernel(X, TH, KEYS, TAB, OUT, H, N).launch(grid=[(N + block - 1) // block, 1, 1],
                                                   block=[block, 1, 1], stream=stream)

    return launch
"""


_SRC_BWD = """
import cutlass
import cutlass.cute as cute
from pypomp.cute.kernel import _load_th, _prep, _hist, _load_g, _zeros, _vjp_once, _store_bwd


def make(spec, J, transform, block):
    step_j, prep_j, step_vjp_j, prep_vjp_j, vec, ns, nth, nc, NSTEP, acc, per_particle = spec
    @cute.kernel
    def kernel(H: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor, G: cute.Tensor,
               DX: cute.Tensor, DTH: cute.Tensor, N: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        g = bidx * block + tidx
        if g < N:
            b = g // J
            j = (g - b * J).to(cutlass.Uint32)
            th_raw = _load_th(TH, g, b, nth, per_particle)
            th = th_raw
            if cutlass.const_expr(transform):
                th = _prep(prep_j, th_raw)
            {gs} = _load_g(G, g, ns)
            {ths} = _zeros(nth)
            for kk in range(NSTEP):
                k = NSTEP - 1 - kk
                {gs} {ths} = _vjp_once(step_vjp_j, vec, J, ns, nth, nc, _hist(H, g, k, ns), th, KEYS, TAB,
                                       b, j, k, ({gs}), ({ths}))
            _store_bwd(DX, DTH, g, ({gs}), ({ths}), acc, prep_vjp_j, th_raw, transform)

    @cute.jit
    def launch(stream, H: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor, G: cute.Tensor,
               DX: cute.Tensor, DTH: cute.Tensor):
        N = cute.size(G, mode=[1])
        kernel(H, TH, KEYS, TAB, G, DX, DTH, N).launch(grid=[(N + block - 1) // block, 1, 1],
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
    """Return the launcher of the "fwd", "save" or "bwd" kernel for J particles."""
    gs, ths = _state_names("g", spec.ns), _state_names("a", spec.nth)
    src = (
        {"fwd": _SRC, "save": _SRC_SAVE, "bwd": _SRC_BWD}[kind]
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


def _make_rprocess_interp(
    model: Pomp,
    traced: _Traced,
    accumvars: tuple[int, ...] | None,
    per_particle: bool,
) -> Callable:
    """Return a CuTe replacement for the model's ``rprocess_interp``.

    With `per_particle`, every particle has its own parameters, as in iterated
    filtering (``mechanics_per_interp``); otherwise every filter has one parameter
    vector (``mechanics_pf_interp``).
    """
    rp = model.rproc
    assert rp.nstep is not None
    xla = rp.mechanics_per_interp if per_particle else rp.mechanics_pf_interp
    step_j, prep_j, vec, step_vjp_j, prep_vjp_j = traced
    spec = _Spec(
        step_j=step_j,
        prep_j=prep_j,
        step_vjp_j=step_vjp_j,
        prep_vjp_j=prep_vjp_j,
        vec=vec,
        ns=len(rp.statenames),
        nth=len(rp.param_names),
        nc=len(rp.covar_names),
        nstep=int(rp.nstep),
        acc=tuple(int(a) for a in (accumvars or ())),
        per_particle=per_particle,
    )
    ns, nth, nc, nstep = spec.ns, spec.nth, spec.nc, spec.nstep
    launchers: dict[tuple[str, int, bool], Any] = {}

    def launcher(kind: str, J: int, transform: bool) -> Any:
        if (kind, J, transform) not in launchers:
            launchers[kind, J, transform] = _build_launcher(kind, spec, J, transform)
        return launchers[kind, J, transform]

    def flatten(th: jax.Array, sk: jax.Array, tab: jax.Array, B: int, J: int):
        # Time and covariates are the same for every filter, so one table serves all.
        TH = th.reshape(B * J if per_particle else B, nth)
        return TH, sk.reshape(B, nstep, 2), tab.reshape(B, nstep, 2 + nc)[0]

    def make_core(transform: bool) -> Callable:
        # Shapes: Xp (..., J, ns); th (..., d), or (..., J, d) per particle;
        # sk (..., nstep, 2); tab (..., nstep, 2 + nc).
        @jax.custom_batching.custom_vmap
        def core(Xp, th, sk, tab):
            lead, J = Xp.shape[:-2], Xp.shape[-2]
            B = math.prod(lead)
            X = Xp.reshape(B * J, ns).T
            out = cutlass_call(
                launcher("fwd", J, transform),
                output_shape_dtype=jax.ShapeDtypeStruct(X.shape, jnp.float32),
            )(X, *flatten(th, sk, tab, B, J))
            return out.T.reshape(lead + (J, ns))

        @core.def_vmap
        def _rule(axis_size, in_batched, *args):
            return core(*_broadcast_unbatched(axis_size, in_batched, args)), True

        @jax.custom_batching.custom_vmap
        def core_save(Xp, th, sk, tab):
            # As core, plus each step's input state: H (..., J, nstep * ns).
            lead, J = Xp.shape[:-2], Xp.shape[-2]
            B = math.prod(lead)
            X = Xp.reshape(B * J, ns).T
            out, H = cutlass_call(
                launcher("save", J, transform),
                output_shape_dtype=(
                    jax.ShapeDtypeStruct(X.shape, jnp.float32),
                    jax.ShapeDtypeStruct((B * J, nstep * ns), jnp.float32),
                ),
            )(X, *flatten(th, sk, tab, B, J))
            return out.T.reshape(lead + (J, ns)), H.reshape(lead + (J, nstep * ns))

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
            G = gout.reshape(B * J, ns).T
            TH, SK, TAB = flatten(th, sk, tab, B, J)
            dX, dTH = cutlass_call(
                launcher("bwd", J, transform),
                output_shape_dtype=(
                    jax.ShapeDtypeStruct(G.shape, jnp.float32),
                    jax.ShapeDtypeStruct((B * J, nth), jnp.float32),
                ),
            )(H.reshape(B * J, nstep * ns), TH, SK, TAB, G)
            dTH = dTH.reshape(lead + (J, nth))
            return dX.T.reshape(lead + (J, ns)), (dTH if per_particle else dTH.sum(-2))

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
        return diff_core

    cores = {False: make_core(False), True: make_core(True)}

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
        if X_.dtype != jnp.float32:  # the kernel is float32; keep x64 runs on XLA
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
        if jnp.ndim(keys) != 0:
            keys = keys[0]
        subkeys, ts, dts, covs = [], [], [], []
        tk = t
        for k in range(nstep):  # the same key chain and clock as _time_interp
            keys, sub = jax.random.split(keys)
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
        out = cores[bool(should_trans)](
            X_,
            jnp.asarray(theta_, jnp.float32),
            jnp.stack(subkeys),
            tab,
        )
        return out, t_idx + nstep

    return rprocess_interp
