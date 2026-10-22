"""One CuTe kernel per observation interval, in place of pypomp's ``rprocess_interp``.

Each CUDA thread owns one particle of one filter, keeps its state in registers and
runs the interval's Euler steps by executing the user's ``rproc``, translated from
its jaxpr (``translate.py``). Every filter of a vmapped call (parameter sets,
replicates, IF2 searches) goes into the same launch: a ``custom_vmap`` rule folds
each batch level into the kernel's particle axis.

The surrounding filter is pypomp's own: the substep keys are split as
``_time_interp`` splits them, and the kernel draws from them with Philox4x32-10
(``translate.Ctx.rng``). A per-particle ``rproc`` gets the key
``split(substep_key, J)[j]``, derived the same way.

Gradients (``jax.custom_vjp``): when JAX differentiates through the interval, the
forward kernel also writes each Euler step's input state to a buffer (nstep x ns
floats per particle; pypomp checkpoints each observation step, so this exists for one
interval at a time). A second kernel then runs the VJP of the user's step, which JAX
derives with ``jax.vjp`` and the translator emits, over the steps in reverse, reading
the states from that buffer. Random numbers are regenerated from the same Philox
counters rather than stored.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import tempfile
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import cutlass
import cutlass.cute as cute
from cutlass.jax import cutlass_call

from . import translate as ct


def trace_step(model):
    """Per-particle jaxprs: the Euler step, the parameter transform, and their VJPs.

    step: (x (ns,), theta (d,), key_data (2,), covars (nc,), t, dt) -> x (ns,), natural scale
    prep: theta on the estimation scale (d,) -> natural scale (d,)
    step_vjp: (same, cotangent of x) -> (cotangent of x, cotangent of theta)
    prep_vjp: (theta, cotangent of natural-scale theta) -> cotangent of theta
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
        res = rp._call_original(X_=X, theta_=theta, key=jax.random.wrap_key_data(kd), covars=covd, t=t, dt=dt)
        return jnp.stack([jnp.asarray(res[n], jnp.float32).reshape(()) for n in snames])

    def prep(th):
        theta = rp._prepare_theta(th, True)
        return jnp.stack([jnp.asarray(theta[n], jnp.float32).reshape(()) for n in pnames])

    def step_vjp(x, th, kd, cov, t, dt, gx):
        _, pull = jax.vjp(lambda x, th: step(x, th, kd, cov, t, dt), x, th)
        return pull(gx)

    def prep_vjp(th, gth):
        return jax.vjp(prep, th)[1](gth)[0]

    f32 = jnp.float32
    args = (jnp.zeros(ns, f32), jnp.ones(d, f32), jnp.zeros(2, jnp.uint32), jnp.ones(max(nc, 1), f32)[:nc],
            f32(0.0), f32(0.01))
    step_j = jax.make_jaxpr(step)(*args)
    prep_j = jax.make_jaxpr(prep)(jnp.zeros(d, f32))
    step_vjp_j = jax.make_jaxpr(step_vjp)(*args, jnp.zeros(ns, f32))
    prep_vjp_j = jax.make_jaxpr(prep_vjp)(jnp.zeros(d, f32), jnp.zeros(d, f32))
    return step_j, prep_j, vec, step_vjp_j, prep_vjp_j


def unsupported(cj) -> set[str]:
    """Primitives in `cj` (recursively) that the translator does not handle."""
    bad = set()

    def walk(jaxpr):
        for e in jaxpr.eqns:
            if e.primitive.name not in ct.SUPPORTED:
                bad.add(e.primitive.name)
            for p in e.params.values():
                for q in (p if isinstance(p, (tuple, list)) else (p,)):
                    j = getattr(q, "jaxpr", None)
                    if j is not None:
                        walk(j if hasattr(j, "eqns") else j.jaxpr)

    walk(cj.jaxpr)
    return bad


# ---- what one thread does (plain Python, run at trace time inside the kernel) ----------

def _load(X, TH, g, b, ns, nth, acc, per_particle):
    xs = [X[s, g] for s in range(ns)]
    for a in acc:
        xs[a] = cutlass.Float32(0.0)
    row = g if per_particle else b
    th = np.empty(nth, dtype=object)
    for k in range(nth):
        th[k] = TH[row, k]
    return tuple(xs), th


def _prep(prep_j, th, rng):
    (out,) = ct.interpret(prep_j, [th], ct.Ctx(rng=rng))
    return np.array([ct.as_dsl(v, np.float32) for v in out], dtype=object)


def _step_once(step_j, vec, J, ns, nc, rng, xs, th, KEYS, TAB, b, j, k):
    """One Euler step for one particle; k may be a runtime loop index."""
    ctx = ct.Ctx(vectorized_J=J if vec else None, particle=j, rng=rng)
    k0, k1 = KEYS[b, k, 0], KEYS[b, k, 1]
    if not vec:  # pypomp's step_pf: split(substep_key, J)[j]
        k0, k1 = ct._key_derive(ctx, k0, k1, j)
    cov = np.empty(nc, dtype=object)
    for c in range(nc):
        cov[c] = TAB[k, 2 + c]
    (out,) = ct.interpret(step_j, [ct.obj(list(xs)), th, ct.obj([k0, k1]), cov, ct.obj(TAB[k, 0]),
                                   ct.obj(TAB[k, 1])], ctx)
    return tuple(ct.as_dsl(out[s], np.float32) for s in range(ns))


def _store(OUT, g, xs):
    for s, x in enumerate(xs):
        OUT[s, g] = x


def _save(H, g, k, xs):
    """Store step k's input state in particle g's row of the state buffer."""
    ns = len(xs)
    for s, x in enumerate(xs):
        H[g, k * ns + s] = x


def _hist(H, g, k, ns):
    return tuple(H[g, k * ns + s] for s in range(ns))


def _load_th(TH, g, b, nth, per_particle):
    row = g if per_particle else b
    th = np.empty(nth, dtype=object)
    for k in range(nth):
        th[k] = TH[row, k]
    return th


def _load_g(G, g, ns):
    return tuple(G[s, g] for s in range(ns))


def _zeros(n):
    return tuple(cutlass.Float32(0.0) for _ in range(n))


def _vjp_once(step_vjp_j, vec, J, ns, nth, nc, rng, xs, th, KEYS, TAB, b, j, k, gs, acc_th):
    """Cotangents through one Euler step; k may be a runtime loop index."""
    ctx = ct.Ctx(vectorized_J=J if vec else None, particle=j, rng=rng)
    k0, k1 = KEYS[b, k, 0], KEYS[b, k, 1]
    if not vec:
        k0, k1 = ct._key_derive(ctx, k0, k1, j)
    cov = np.empty(nc, dtype=object)
    for c in range(nc):
        cov[c] = TAB[k, 2 + c]
    gx, gth = ct.interpret(step_vjp_j, [ct.obj(list(xs)), th, ct.obj([k0, k1]), cov, ct.obj(TAB[k, 0]),
                                        ct.obj(TAB[k, 1]), ct.obj(list(gs))], ctx)
    gx = tuple(ct.as_dsl(gx[s], np.float32) for s in range(ns))
    acc_th = tuple(a + ct.as_dsl(gth[i], np.float32) for i, a in enumerate(acc_th))
    return gx + acc_th


def _store_bwd(DX, DTH, g, gx, gth, acc, prep_vjp_j, th_raw, transform, rng):
    for s, v in enumerate(gx):
        DX[s, g] = cutlass.Float32(0.0) if s in acc else v  # accumulators are reset at the start
    if transform:
        (out,) = ct.interpret(prep_vjp_j, [th_raw, ct.obj(list(gth))], ct.Ctx(rng=rng))
        gth = [ct.as_dsl(v, np.float32) for v in out]
    for i, v in enumerate(gth):
        DTH[g, i] = v


# CuTe's AST preprocessor turns `for k in range(NSTEP)` into a device loop with the
# state as loop-carried registers, so the step's code is emitted once, not NSTEP times.
# That needs one named variable per state, hence a generated module per model.
_SRC = """
import cutlass
import cutlass.cute as cute
from pypomp.cute.kernel import _load, _prep, _step_once, _store


def make(step_j, prep_j, vec, J, ns, nth, nc, NSTEP, acc, per_particle, transform, rng, block):
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
                th = _prep(prep_j, th, rng)
            {xs} = xs0
            for k in range(NSTEP):
                {xs} = _step_once(step_j, vec, J, ns, nc, rng, ({xs}), th, KEYS, TAB, b, j, k)
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


def make(step_j, prep_j, vec, J, ns, nth, nc, NSTEP, acc, per_particle, transform, rng, block):
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
                th = _prep(prep_j, th, rng)
            {xs} = xs0
            for k in range(NSTEP):
                _save(H, g, k, ({xs}))
                {xs} = _step_once(step_j, vec, J, ns, nc, rng, ({xs}), th, KEYS, TAB, b, j, k)
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


def make(step_j, prep_j, step_vjp_j, prep_vjp_j, vec, J, ns, nth, nc, NSTEP, acc, per_particle, transform,
         rng, block):
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
                th = _prep(prep_j, th_raw, rng)
            {gs} = _load_g(G, g, ns)
            {ths} = _zeros(nth)
            for kk in range(NSTEP):
                k = NSTEP - 1 - kk
                {gs} {ths} = _vjp_once(step_vjp_j, vec, J, ns, nth, nc, rng, _hist(H, g, k, ns), th, KEYS, TAB,
                                       b, j, k, ({gs}), ({ths}))
            _store_bwd(DX, DTH, g, ({gs}), ({ths}), acc, prep_vjp_j, th_raw, transform, rng)

    @cute.jit
    def launch(stream, H: cute.Tensor, TH: cute.Tensor, KEYS: cute.Tensor, TAB: cute.Tensor, G: cute.Tensor,
               DX: cute.Tensor, DTH: cute.Tensor):
        N = cute.size(G, mode=[1])
        kernel(H, TH, KEYS, TAB, G, DX, DTH, N).launch(grid=[(N + block - 1) // block, 1, 1],
                                                       block=[block, 1, 1], stream=stream)

    return launch
"""


def _import_generated(src):
    path = os.path.join(tempfile.gettempdir(), f"pypomp_cute_{hashlib.md5(src.encode()).hexdigest()[:12]}.py")
    if not os.path.exists(path):
        with open(path, "w") as fh:
            fh.write(src)
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_bwd_launcher(step_j, prep_j, step_vjp_j, prep_vjp_j, vec, J, ns, nth, nc, nstep, acc, per_particle,
                        transform, rng, block):
    names = lambda p, n: "".join(f"{p}{i}, " for i in range(n))  # noqa: E731
    src = (_SRC_BWD.replace("{xs}", names("x", ns).rstrip()).replace("{gs} {ths}", names("g", ns) + names("a", nth)
           .rstrip()).replace("{gs}", names("g", ns).rstrip()).replace("{ths}", names("a", nth).rstrip()))
    return _import_generated(src).make(step_j, prep_j, step_vjp_j, prep_vjp_j, vec, J, ns, nth, nc, nstep, acc,
                                       per_particle, transform, rng, block)


def _build_launcher(step_j, prep_j, vec, J, ns, nth, nc, nstep, acc, per_particle, transform, rng, block,
                    save=False):
    src = (_SRC_SAVE if save else _SRC).replace("{xs}", "".join(f"x{s}, " for s in range(ns)).rstrip())
    return _import_generated(src).make(step_j, prep_j, vec, J, ns, nth, nc, nstep, acc, per_particle, transform,
                                       rng, block)


def make_rprocess_interp(model, accumvars, *, per_particle: bool, rng: str = "philox", block: int = 128):
    """A drop-in for the model's ``mechanics_pf_interp``, or ``mechanics_per_interp`` (IF2,
    one parameter vector per particle) when ``per_particle``."""
    rp = model.rproc
    nstep = int(rp.nstep)
    step_j, prep_j, vec, step_vjp_j, prep_vjp_j = trace_step(model)
    ns, nth, nc = len(rp.statenames), len(rp.param_names), len(rp.covar_names)
    acc = tuple(int(a) for a in (accumvars or ()))
    launchers: dict[Any, Any] = {}

    def bwd_launcher(J, transform):
        if ("bwd", J, transform) not in launchers:
            launchers["bwd", J, transform] = _build_bwd_launcher(
                step_j, prep_j, step_vjp_j, prep_vjp_j, vec, J, ns, nth, nc, nstep, acc, per_particle, transform,
                rng, block)
        return launchers["bwd", J, transform]

    def launcher(J, transform, save=False):
        if (J, transform, save) not in launchers:
            launchers[J, transform, save] = _build_launcher(step_j, prep_j, vec, J, ns, nth, nc, nstep, acc,
                                                            per_particle, transform, rng, block, save)
        return launchers[J, transform, save]

    def make_core(transform):
        @jax.custom_batching.custom_vmap
        def core(Xp, th, sk, tab):
            # Xp (..., J, ns); th (..., d), or (..., J, d) per particle; sk (..., nstep, 2);
            # tab (..., nstep, 2 + nc)
            lead, J = Xp.shape[:-2], Xp.shape[-2]
            B = math.prod(lead)
            X = Xp.reshape(B * J, ns).T
            TH = th.reshape(B * J if per_particle else B, nth)
            SK = sk.reshape(B, nstep, 2)
            TAB = tab.reshape(B, nstep, 2 + nc)[0]  # time and covariates are the same for every filter
            out = cutlass_call(launcher(J, transform),
                               output_shape_dtype=jax.ShapeDtypeStruct(X.shape, jnp.float32))(X, TH, SK, TAB)
            return out.T.reshape(lead + (J, ns))

        @core.def_vmap
        def _rule(axis_size, in_batched, Xp, th, sk, tab):
            args = [a if bat else jnp.broadcast_to(a, (axis_size,) + a.shape)
                    for a, bat in zip((Xp, th, sk, tab), in_batched)]
            return core(*args), True

        @jax.custom_batching.custom_vmap
        def core_save(Xp, th, sk, tab):
            # As core, plus the per-step input states: H (..., J, nstep * ns).
            lead, J = Xp.shape[:-2], Xp.shape[-2]
            B = math.prod(lead)
            X = Xp.reshape(B * J, ns).T
            TH = th.reshape(B * J if per_particle else B, nth)
            SK = sk.reshape(B, nstep, 2)
            TAB = tab.reshape(B, nstep, 2 + nc)[0]
            out, H = cutlass_call(launcher(J, transform, save=True), output_shape_dtype=(
                jax.ShapeDtypeStruct(X.shape, jnp.float32),
                jax.ShapeDtypeStruct((B * J, nstep * ns), jnp.float32)))(X, TH, SK, TAB)
            return out.T.reshape(lead + (J, ns)), H.reshape(lead + (J, nstep * ns))

        @core_save.def_vmap
        def _rule_save(axis_size, in_batched, Xp, th, sk, tab):
            args = [a if bat else jnp.broadcast_to(a, (axis_size,) + a.shape)
                    for a, bat in zip((Xp, th, sk, tab), in_batched)]
            return core_save(*args), (True, True)

        @jax.custom_batching.custom_vmap
        def core_bwd(H, th, sk, tab, gout):
            lead, J = gout.shape[:-2], gout.shape[-2]
            B = math.prod(lead)
            G = gout.reshape(B * J, ns).T
            TH = th.reshape(B * J if per_particle else B, nth)
            SK = sk.reshape(B, nstep, 2)
            TAB = tab.reshape(B, nstep, 2 + nc)[0]
            dX, dTH = cutlass_call(bwd_launcher(J, transform), output_shape_dtype=(
                jax.ShapeDtypeStruct(G.shape, jnp.float32), jax.ShapeDtypeStruct((B * J, nth), jnp.float32)))(
                H.reshape(B * J, nstep * ns), TH, SK, TAB, G)
            dTH = dTH.reshape(lead + (J, nth))
            return dX.T.reshape(lead + (J, ns)), (dTH if per_particle else dTH.sum(-2))

        @core_bwd.def_vmap
        def _rule_bwd(axis_size, in_batched, H, th, sk, tab, gout):
            args = [a if bat else jnp.broadcast_to(a, (axis_size,) + a.shape)
                    for a, bat in zip((H, th, sk, tab, gout), in_batched)]
            return core_bwd(*args), (True, True)

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

    def rprocess_interp(X_, theta_, keys, covars_extended, dt_array_extended, t, t_idx, nstep_dynamic,
                        accumvars_, should_trans):
        if jnp.ndim(keys) != 0:
            keys = keys[0]
        subkeys, ts, dts, covs = [], [], [], []
        tk = t
        for k in range(nstep):  # the same key chain and clock as pypomp's _time_interp
            keys, sub = jax.random.split(keys)
            subkeys.append(jax.random.key_data(sub))
            dt = dt_array_extended[t_idx + k]
            ts.append(tk)
            dts.append(dt)
            if nc:
                covs.append(covars_extended[t_idx + k])
            tk = tk + dt
        cols = [jnp.stack([jnp.asarray(ts, jnp.float32), jnp.asarray(dts, jnp.float32)], 1)]
        if nc:
            cols.append(jnp.stack(covs).astype(jnp.float32))
        tab = jnp.concatenate(cols, axis=1)
        out = cores[bool(should_trans)](X_.astype(jnp.float32), jnp.asarray(theta_, jnp.float32),
                                        jnp.stack(subkeys), tab)
        return out, t_idx + nstep

    return rprocess_interp
