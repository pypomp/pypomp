"""Emit CuTe DSL scalar code for a JAX function of one particle.

`interpret(closed_jaxpr, args, ctx)` walks the jaxpr inside a `@cute.kernel` and
returns, for each output, a numpy object array holding CuTe DSL scalars (or plain
constants). Every jaxpr value keeps its (small, per-particle) shape as an object
array, so slices, reshapes and reductions over e.g. an (11,) draw of uniforms are
just bookkeeping and the emitted code is straight-line scalar math in registers.

Random numbers (`ctx.rng`):
  "philox"   (default) Philox4x32-10 with all four output words used, laid out as in
             pypomp.random.philox: element (l, m) of a draw of shape (..., L) is word
             l % 4 of the call with counter (l // 4) * L + m. A He10 step's 11
             uniforms per particle then cost 3 calls.
  "threefry" JAX's threefry2x32 exactly (one call per 32-bit word), which reproduces
             pypomp's default keys; slower, kept for checks.
For an rproc written over all J particles at once (pypomp's @vectorized), the
function is traced with J = 1; `ctx.vectorized_J` and `ctx.particle` then make the
trailing particle axis of a random draw use the counters the full (..., J) draw
would have used, so every particle gets its own numbers.

Unsupported primitives raise NotImplementedError naming the primitive, so a caller
can fall back to the XLA path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import Literal

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith as _arith

DSL = {
    np.dtype("float32"): cutlass.Float32,
    np.dtype("int32"): cutlass.Int32,
    np.dtype("uint32"): cutlass.Uint32,
    np.dtype("bool"): cutlass.Boolean,
}
NumericBase = cutlass.Numeric


@dataclass
class Ctx:
    vectorized_J: int | None = None  # J for a @vectorized rproc traced with J = 1
    particle: Any = None  # this thread's particle index (DSL Uint32) within its filter
    rng: str = "philox"  # "philox" or "threefry"


def is_dsl(x):
    return isinstance(x, NumericBase)


def obj(a):
    a = np.asarray(a, dtype=object) if not isinstance(a, np.ndarray) or a.dtype != object else a
    return a


def const(v, dtype):
    """A constant as a DSL scalar of the given numpy dtype."""
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        return cutlass.Boolean(bool(v))
    if np.issubdtype(dtype, np.floating):
        return DSL[dtype](float(v))
    return DSL[dtype](int(np.asarray(v).astype(dtype)))


def as_dsl(x, dtype):
    return x if is_dsl(x) else const(x, dtype)


def select(p, t, f, dtype):
    """p ? t : f for scalars (p a DSL Boolean or a Python bool)."""
    if not is_dsl(p):
        return t if bool(p) else f
    if not is_dsl(t) and not is_dsl(f) and np.asarray(t) == np.asarray(f):
        return t
    t, f = as_dsl(t, dtype), as_dsl(f, dtype)
    return DSL[np.dtype(dtype)](_arith.select(p.ir_value(), t.ir_value(), f.ir_value()))


# ---- threefry2x32, exactly as JAX's _threefry2x32_lowering (unrolled) -------------------

ROT = ([13, 15, 26, 6], [17, 29, 16, 24])


def _u32(x):
    return x if is_dsl(x) else np.uint32(x)


def _rotl(x, r):
    if is_dsl(x):
        return (x << cutlass.Uint32(r)) | (x >> cutlass.Uint32(32 - r))
    x = np.uint32(x)
    return np.uint32((x << np.uint32(r)) | (x >> np.uint32(32 - r)))


def _add(a, b):
    if is_dsl(a) or is_dsl(b):
        return as_dsl(a, np.uint32) + as_dsl(b, np.uint32)
    return np.uint32((int(a) + int(b)) & 0xFFFFFFFF)


def _xor(a, b):
    if is_dsl(a) or is_dsl(b):
        return as_dsl(a, np.uint32) ^ as_dsl(b, np.uint32)
    return np.uint32(a) ^ np.uint32(b)


def threefry2x32(k0, k1, x0, x1):
    ks = [_u32(k0), _u32(k1), _xor(_xor(k0, k1), np.uint32(0x1BD11BDA))]
    x = [_add(x0, ks[0]), _add(x1, ks[1])]

    def rnd(x, r):
        a = _add(x[0], x[1])
        b = _xor(_rotl(x[1], r), a)
        return [a, b]

    for g in range(5):
        for r in ROT[g % 2]:
            x = rnd(x, r)
        x = [_add(x[0], ks[(g + 1) % 3]), _add(_add(x[1], ks[(g + 2) % 3]), np.uint32(g + 1))]
    return x[0], x[1]


# ---- Philox4x32-10, as pypomp.random.philox.philox4x32 --------------------------------

PM0, PM1 = 0xD2511F53, 0xCD9E8D57
PW0, PW1 = np.uint32(0x9E3779B9), np.uint32(0xBB67AE85)


def _mulhilo(m, x):
    if not is_dsl(x):
        p = m * int(x)
        return np.uint32(p & 0xFFFFFFFF), np.uint32(p >> 32)
    p = cutlass.Uint64(m) * x.to(cutlass.Uint64)
    return p.to(cutlass.Uint32), (p >> cutlass.Uint64(32)).to(cutlass.Uint32)


def philox4x32(k0, k1, x0, x1, x2, x3):
    k0, k1, x0, x1, x2, x3 = (_u32(v) for v in (k0, k1, x0, x1, x2, x3))
    for r in range(10):
        if r:
            k0, k1 = _add(k0, PW0), _add(k1, PW1)
        lo0, hi0 = _mulhilo(PM0, x0)
        lo1, hi1 = _mulhilo(PM1, x2)
        x0, x1, x2, x3 = _xor(_xor(hi1, x1), k0), lo1, _xor(_xor(hi0, x3), k1), lo0
    return x0, x1, x2, x3


def _key_derive(ctx, k0, k1, cnt):
    """split / fold_in: a new key from (key, counter)."""
    if ctx.rng == "philox":
        y = philox4x32(k0, k1, 0, 0, 0, cnt)
        return y[0], y[1]
    return threefry2x32(k0, k1, np.uint32(0), cnt)


def _draw(ctx, k0, k1, shape):
    """32-bit words for a draw of `shape` from key (k0, k1), as an object array."""
    out = np.empty(shape, dtype=object)
    n = math.prod(shape)
    if n == 0:
        return out
    if ctx.rng != "philox":
        for f, idx in enumerate(np.ndindex(*shape) if shape else [()]):
            y0, y1 = threefry2x32(k0, k1, np.uint32(0), _counter(f, shape, ctx))
            out[idx] = _xor(y0, y1)
        return out
    vec = bool(ctx.vectorized_J)
    if len(shape) >= 2:
        lane = shape[-1]
        flat = out.reshape(-1, lane)
        lead = flat.shape[0]
        if vec and lane == 1:  # the particle axis, length J at run time
            lanes = [(0, ctx.particle, ctx.vectorized_J)]
        else:
            lanes = [(m, np.uint32(m), lane) for m in range(lane)]
        for col, m, L in lanes:
            words = {}
            for l in range(lead):
                g = l // 4
                if g not in words:
                    words[g] = philox4x32(k0, k1, _add(np.uint32(g * L), m), 0, 0, 0)
                flat[l, col] = words[g][l % 4]
        return out
    flat = out.reshape(-1)
    if vec and shape == (1,):  # a 1-D draw over particles: element j of a (J,) draw
        j = ctx.particle
        g = j >> cutlass.Uint32(2)
        w = philox4x32(k0, k1, g, 0, 0, 0)
        r = j & cutlass.Uint32(3)
        flat[0] = select(r == cutlass.Uint32(0), w[0], select(r == cutlass.Uint32(1), w[1], select(
            r == cutlass.Uint32(2), w[2], w[3], np.uint32), np.uint32), np.uint32)
        return out
    words = {}
    for f in range(n):
        g = f // 4
        if g not in words:
            words[g] = philox4x32(k0, k1, np.uint32(g), 0, 0, 0)
        flat[f] = words[g][f % 4]
    return out


# ---- elementwise primitives ------------------------------------------------------------

def _erf_inv32(x):
    """Giles' single-precision erfinv, the approximation XLA uses for float32."""
    w = -cute.math.log((cutlass.Float32(1.0) - x) * (cutlass.Float32(1.0) + x))
    lt = w < cutlass.Float32(5.0)
    w1 = w - cutlass.Float32(2.5)
    w2 = cute.math.sqrt(w) - cutlass.Float32(3.0)
    c1 = [2.81022636e-08, 3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087,
          -0.00125372503, -0.00417768164, 0.246640727, 1.50140941]
    c2 = [-0.000200214257, 0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773,
          -0.0076224613, 0.00943887047, 1.00167406, 2.83297682]
    w = select(lt, w1, w2, np.float32)
    p = select(lt, cutlass.Float32(c1[0]), cutlass.Float32(c2[0]), np.float32)
    for a, b in zip(c1[1:], c2[1:]):
        p = select(lt, cutlass.Float32(a), cutlass.Float32(b), np.float32) + p * w
    return p * x


def _ipow(x, n):
    if n == 0:
        return cutlass.Float32(1.0) if is_dsl(x) else 1.0
    neg, n = n < 0, abs(n)
    acc, base = None, x
    while n:
        if n & 1:
            acc = base if acc is None else acc * base
        n >>= 1
        if n:
            base = base * base
    return (cutlass.Float32(1.0) / acc) if neg else acc


def _ew(name, args, params, out_dtype, in_dtypes):
    """One elementwise op on scalars (DSL values or constants)."""
    if not any(is_dsl(a) for a in args):
        return None  # caller folds constants with numpy/JAX
    a = [as_dsl(v, d) for v, d in zip(args, in_dtypes)]
    x = a[0]
    f = np.issubdtype(np.dtype(in_dtypes[0]), np.floating)
    m = cute.math
    if name in ("add", "add_any"):
        return x + a[1]
    if name == "sub":
        return x - a[1]
    if name == "mul":
        return x * a[1]
    if name == "div":
        return x / a[1] if f else x // a[1]
    if name == "rem":
        if f:
            q = x / a[1]
            return x - a[1] * select(q < cutlass.Float32(0.0), m.ceil(q), m.floor(q), np.float32)
        return x % a[1]
    if name == "neg":
        return -x
    if name == "max":
        return m.max(x, a[1]) if f else select(x > a[1], x, a[1], in_dtypes[0])
    if name == "min":
        return m.min(x, a[1]) if f else select(x < a[1], x, a[1], in_dtypes[0])
    if name == "abs":
        return m.abs(x) if f else select(x < const(0, in_dtypes[0]), -x, x, in_dtypes[0])
    if name == "sign":
        z = const(0, in_dtypes[0])
        return select(x > z, const(1, in_dtypes[0]), select(x < z, const(-1, in_dtypes[0]), z, in_dtypes[0]),
                      in_dtypes[0])
    if name in ("exp", "log", "sqrt", "rsqrt", "expm1", "log1p", "tanh", "sin", "cos", "erf", "erfc",
                "floor", "ceil", "exp2", "log2"):
        return getattr(m, name)(x)
    if name == "round":
        return m.round(x) if params.get("rounding_method", 0) in (0, "0") or str(
            params.get("rounding_method")).endswith("AWAY_FROM_ZERO") else m.roundeven(x)
    if name == "pow":
        return m.pow(x, a[1])
    if name == "integer_pow":
        return _ipow(x, params["y"])
    if name == "square":
        return x * x
    if name == "logistic":
        return cutlass.Float32(1.0) / (cutlass.Float32(1.0) + m.exp(-x))
    if name == "erf_inv":
        return _erf_inv32(x)
    if name == "is_finite":
        return m.isfinite(x)
    if name in ("eq", "ne", "lt", "le", "gt", "ge"):
        return {"eq": x == a[1], "ne": x != a[1], "lt": x < a[1], "le": x <= a[1],
                "gt": x > a[1], "ge": x >= a[1]}[name]
    if name == "and":
        return x & a[1]
    if name == "or":
        return x | a[1]
    if name == "xor":
        return x ^ a[1]
    if name == "not":
        return ~x
    if name == "shift_left":
        return x << a[1]
    if name in ("shift_right_logical", "shift_right_arithmetic"):
        return x >> a[1]
    if name == "convert_element_type":
        return x.to(DSL[np.dtype(out_dtype)])
    if name == "bitcast_convert_type":
        return x.bitcast(DSL[np.dtype(out_dtype)])
    if name == "clamp":  # clamp(lo, x, hi)
        lo, v, hi = a
        return m.min(m.max(v, lo), hi) if np.issubdtype(np.dtype(in_dtypes[1]), np.floating) else select(
            v < lo, lo, select(v > hi, hi, v, in_dtypes[1]), in_dtypes[1])
    if name == "nextafter":
        raise NotImplementedError("nextafter")
    raise NotImplementedError(name)


ELEMENTWISE = {"add", "add_any", "sub", "mul", "div", "rem", "neg", "max", "min", "abs", "sign", "exp", "log", "sqrt",
               "rsqrt", "expm1", "log1p", "tanh", "sin", "cos", "erf", "erfc", "floor", "ceil", "exp2", "log2",
               "round", "pow", "integer_pow", "square", "logistic", "erf_inv", "is_finite", "eq", "ne", "lt",
               "le", "gt", "ge", "and", "or", "xor", "not", "shift_left", "shift_right_logical",
               "shift_right_arithmetic", "convert_element_type", "bitcast_convert_type", "clamp"}
IDENTITY = {"stop_gradient", "optimization_barrier", "copy", "copy_p", "reduce_precision", "pvary"}
CALLS = {"pjit", "jit", "closed_call", "core_call", "remat", "checkpoint", "custom_jvp_call",
         "custom_vjp_call", "custom_vjp_call_jaxpr", "custom_lin"}
SUPPORTED = ELEMENTWISE | IDENTITY | CALLS | {
    "select_n", "polynomial", "pad", "broadcast_in_dim", "reshape", "squeeze", "expand_dims", "transpose", "slice",
    "dynamic_slice", "dynamic_update_slice", "concatenate", "split", "unstack", "stack", "iota", "reduce_sum",
    "reduce_prod", "reduce_max", "reduce_min", "reduce_and", "reduce_or", "dot_general", "cond", "while",
    "scan", "random_wrap", "random_unwrap", "random_fold_in", "random_split", "random_bits"}


def _fold(eqn, vals):
    """Evaluate an op whose inputs are all constants, eagerly on the CPU."""
    # This runs while the CuTe bridge is itself being traced by JAX, so force eager
    # evaluation; the inputs are concrete.
    with jax.ensure_compile_time_eval(), jax.default_device(jax.devices("cpu")[0]):
        out = eqn.primitive.bind(*[jnp.asarray(v) for v in vals], **eqn.params)
    outs = out if eqn.primitive.multiple_results else [out]
    return [np.asarray(o) for o in outs]


def _sub_jaxpr(eqn):
    for k in ("jaxpr", "call_jaxpr", "fun_jaxpr"):
        if k in eqn.params:
            j = eqn.params[k]
            return j if hasattr(j, "consts") else jax.extend.core.ClosedJaxpr(j, ())
    raise NotImplementedError(f"{eqn.primitive.name}: no sub-jaxpr")


def _all_const(vals):
    return all(not (isinstance(v, np.ndarray) and v.dtype == object and any(is_dsl(e) for e in v.flat))
               and not is_dsl(v) for v in vals)


def _as_np_const(v, dtype):
    v = np.asarray(v)
    if v.dtype == object:
        v = v.astype(dtype)
    return v


def interpret(cj, args, ctx: Ctx):
    """Evaluate ClosedJaxpr `cj` on object-array args; returns a list of object arrays."""
    jaxpr = cj.jaxpr
    env = {}

    def read(v):
        if isinstance(v, Literal):
            return np.asarray(v.val)
        return env[v]

    for v, c in zip(jaxpr.constvars, cj.consts):
        env[v] = np.asarray(c)
    for v, a in zip(jaxpr.invars, args):
        env[v] = a

    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        ins = [read(v) for v in eqn.invars]
        in_dt = [getattr(v.aval, "dtype", None) for v in eqn.invars]
        out_avals = [v.aval for v in eqn.outvars]
        outs = _eval_eqn(eqn, name, ins, in_dt, out_avals, ctx)
        for v, o in zip(eqn.outvars, outs):
            env[v] = o
    return [read(v) for v in jaxpr.outvars]


def _is_key(dtype):
    return dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key)


def _eval_eqn(eqn, name, ins, in_dt, out_avals, ctx):
    # Constant folding: no DSL value among the inputs and no PRNG keys.
    if (_all_const(ins) and name not in CALLS and name not in ("cond", "while", "scan")
            and not any(_is_key(d) for d in in_dt) and not any(_is_key(a.dtype) for a in out_avals)):
        return _fold(eqn, [_as_np_const(v, d) for v, d in zip(ins, in_dt)])

    if name in IDENTITY:
        return ins[: len(out_avals)]

    if name in ELEMENTWISE:
        shape = out_avals[0].shape
        b = [np.broadcast_to(obj(v), shape) for v in ins]
        out = np.empty(shape, dtype=object)
        for idx in np.ndindex(*shape) if shape else [()]:
            elems = [x[idx] for x in b]
            r = _ew(name, elems, eqn.params, out_avals[0].dtype, in_dt)
            if r is None:
                r = _fold(eqn, [np.asarray(e).astype(d) for e, d in zip(elems, in_dt)])[0][()]
            out[idx] = r
        return [out]

    if name == "select_n":
        pred, cases = ins[0], ins[1:]
        shape = out_avals[0].shape
        dt = out_avals[0].dtype
        p = np.broadcast_to(obj(pred), shape)
        cs = [np.broadcast_to(obj(c), shape) for c in cases]
        out = np.empty(shape, dtype=object)
        for idx in np.ndindex(*shape) if shape else [()]:
            pi = p[idx]
            if not is_dsl(pi):
                out[idx] = cs[int(pi)][idx]
                continue
            if len(cs) == 2 and np.dtype(in_dt[0]) == np.bool_:
                out[idx] = select(pi, cs[1][idx], cs[0][idx], dt)
            else:
                r = cs[-1][idx]
                for c in range(len(cs) - 2, -1, -1):
                    r = select(pi == const(c, in_dt[0]), cs[c][idx], r, dt)
                out[idx] = r
        return [out]

    if name == "polynomial":  # coefficients lowest degree first
        x, coeffs = ins[0], ins[1:]
        shape = out_avals[0].shape
        xb = np.broadcast_to(obj(x), shape)
        cb = [np.broadcast_to(obj(c), shape) for c in coeffs]
        out = np.empty(shape, dtype=object)
        for idx in np.ndindex(*shape) if shape else [()]:
            r = as_dsl(cb[-1][idx], np.float32)
            for c in reversed(cb[:-1]):
                r = as_dsl(c[idx], np.float32) + r * as_dsl(xb[idx], np.float32)
            out[idx] = r
        return [out]

    # ---- shape bookkeeping on object arrays ----
    if name == "broadcast_in_dim":
        x = obj(ins[0])
        shape, bdims = eqn.params["shape"], eqn.params["broadcast_dimensions"]
        expanded = [1] * len(shape)
        for i, d in enumerate(bdims):
            expanded[d] = x.shape[i]
        return [np.broadcast_to(x.reshape(expanded), shape).copy()]
    if name == "reshape":
        return [obj(ins[0]).reshape(eqn.params["new_sizes"])]
    if name == "squeeze":
        return [np.squeeze(obj(ins[0]), axis=tuple(eqn.params["dimensions"]))]
    if name == "expand_dims":
        return [np.expand_dims(obj(ins[0]), eqn.params["dimensions"])]
    if name == "transpose":
        return [np.transpose(obj(ins[0]), eqn.params["permutation"])]
    if name == "slice":
        st, lim, strd = eqn.params["start_indices"], eqn.params["limit_indices"], eqn.params["strides"]
        strd = strd or (1,) * len(st)
        return [obj(ins[0])[tuple(slice(a, b, c) for a, b, c in zip(st, lim, strd))]]
    if name == "dynamic_slice":
        starts = ins[1:]
        if not _all_const(starts):
            raise NotImplementedError("dynamic_slice with a runtime index")
        x = obj(ins[0])
        sizes = eqn.params["slice_sizes"]
        st = [int(np.clip(int(s), 0, n - k)) for s, n, k in zip(starts, x.shape, sizes)]
        return [x[tuple(slice(a, a + k) for a, k in zip(st, sizes))]]
    if name == "dynamic_update_slice":
        x, upd, starts = obj(ins[0]).copy(), obj(ins[1]), ins[2:]
        if not _all_const(starts):
            raise NotImplementedError("dynamic_update_slice with a runtime index")
        st = [int(np.clip(int(s), 0, n - k)) for s, n, k in zip(starts, x.shape, upd.shape)]
        x[tuple(slice(a, a + k) for a, k in zip(st, upd.shape))] = upd
        return [x]
    if name == "pad":  # also the transpose of slice in gradients
        x, pv = obj(ins[0]), obj(ins[1]).reshape(())[()]
        cfg = eqn.params["padding_config"]
        shape = out_avals[0].shape
        out = np.empty(shape, dtype=object)
        for idx in np.ndindex(*shape) if shape else [()]:
            out[idx] = pv
        for idx in np.ndindex(*x.shape) if x.shape else [()]:
            pos = tuple(lo + i * (interior + 1) for i, (lo, _, interior) in zip(idx, cfg))
            if all(0 <= q < n for q, n in zip(pos, shape)):
                out[pos] = x[idx]
        return [out]
    if name == "concatenate":
        return [np.concatenate([obj(v) for v in ins], axis=eqn.params["dimension"])]
    if name == "split":
        x = obj(ins[0])
        cuts = np.cumsum(eqn.params["sizes"])[:-1]
        return list(np.split(x, cuts, axis=eqn.params["axis"]))
    if name == "unstack":
        x = obj(ins[0])
        ax = eqn.params["axis"]
        return [np.take(x, i, axis=ax) for i in range(x.shape[ax])]
    if name == "stack":
        return [np.stack([obj(v) for v in ins], axis=eqn.params["axis"])]
    if name == "iota":
        shape, dim = eqn.params["shape"], eqn.params["dimension"]
        a = np.arange(shape[dim], dtype=eqn.params["dtype"])
        sh = [1] * len(shape)
        sh[dim] = shape[dim]
        return [np.broadcast_to(a.reshape(sh), shape).copy()]
    if name in ("reduce_sum", "reduce_prod", "reduce_max", "reduce_min", "reduce_and", "reduce_or"):
        x = obj(ins[0])
        axes = tuple(eqn.params["axes"])
        dt = out_avals[0].dtype
        op = {"reduce_sum": "add", "reduce_prod": "mul", "reduce_max": "max", "reduce_min": "min",
              "reduce_and": "and", "reduce_or": "or"}[name]
        keep = [d for d in range(x.ndim) if d not in axes]
        xt = np.transpose(x, keep + list(axes)).reshape([x.shape[d] for d in keep] + [-1])
        out = np.empty(xt.shape[:-1], dtype=object)
        for idx in np.ndindex(*out.shape) if out.shape else [()]:
            r = xt[idx][0]
            for e in xt[idx][1:]:
                rr = _ew(op, [r, e], {}, dt, [dt, dt])
                r = rr if rr is not None else np.asarray(
                    {"add": np.add, "mul": np.multiply, "max": np.maximum, "min": np.minimum,
                     "and": np.logical_and, "or": np.logical_or}[op](r, e)).astype(dt)
            out[idx] = r
        return [out]
    if name == "dot_general":
        (lc, rc), (lb, rb) = eqn.params["dimension_numbers"]
        if lb or rb:
            raise NotImplementedError("batched dot_general")
        a, b = obj(ins[0]), obj(ins[1])
        dt = out_avals[0].dtype
        af = [d for d in range(a.ndim) if d not in lc]
        bf = [d for d in range(b.ndim) if d not in rc]
        at = np.transpose(a, af + list(lc)).reshape([a.shape[d] for d in af] + [-1])
        bt = np.transpose(b, bf + list(rc)).reshape([b.shape[d] for d in bf] + [-1])
        out = np.empty(out_avals[0].shape, dtype=object)
        for i in np.ndindex(*at.shape[:-1]) if at.ndim > 1 else [()]:
            for k in np.ndindex(*bt.shape[:-1]) if bt.ndim > 1 else [()]:
                r = None
                for x, y in zip(at[i], bt[k]):
                    p = _ew("mul", [x, y], {}, dt, [dt, dt])
                    if p is None:
                        p = np.asarray(np.asarray(x, dt) * np.asarray(y, dt), dt)
                    if r is None:
                        r = p
                    else:
                        rp = _ew("add", [r, p], {}, dt, [dt, dt])
                        r = rp if rp is not None else np.asarray(np.asarray(r, dt) + np.asarray(p, dt), dt)
                out[i + k] = r
        return [out]

    # ---- calls and control flow ----
    if name in CALLS:
        return interpret(_sub_jaxpr(eqn), ins, ctx)
    if name == "cond":
        index, ops = ins[0], ins[1:]
        branches = eqn.params["branches"]
        idx = obj(index).reshape(())[()] if np.ndim(index) == 0 else None
        if idx is None:
            raise NotImplementedError("cond with a batched index")
        if not is_dsl(idx):
            return interpret(branches[int(idx)], ops, ctx)
        results = [interpret(b, ops, ctx) for b in branches]  # both sides, then select
        outs = []
        for k, av in enumerate(out_avals):
            shape = av.shape
            cs = [np.broadcast_to(obj(r[k]), shape) for r in results]
            out = np.empty(shape, dtype=object)
            for p in np.ndindex(*shape) if shape else [()]:
                r = cs[-1][p]
                for c in range(len(cs) - 2, -1, -1):
                    r = select(idx == const(c, in_dt[0]), cs[c][p], r, av.dtype)
                out[p] = r
            outs.append(out)
        return outs
    if name == "while":
        cn, bn = eqn.params["cond_nconsts"], eqn.params["body_nconsts"]
        cconsts, bconsts, carry = ins[:cn], ins[cn:cn + bn], ins[cn + bn:]
        for _ in range(100000):
            (c,) = interpret(eqn.params["cond_jaxpr"], list(cconsts) + list(carry), ctx)
            c = obj(c).reshape(())[()]
            if is_dsl(c):
                raise NotImplementedError("while loop with a runtime trip count")
            if not bool(c):
                return carry
            carry = interpret(eqn.params["body_jaxpr"], list(bconsts) + list(carry), ctx)
        raise NotImplementedError("while loop did not terminate during translation")
    if name == "scan":
        p = eqn.params
        nc, ncarry, length = p["num_consts"], p["num_carry"], p["length"]
        consts, carry, xs = ins[:nc], ins[nc:nc + ncarry], ins[nc + ncarry:]
        ys = []
        order = range(length - 1, -1, -1) if p["reverse"] else range(length)
        for t in order:
            x_t = [obj(x)[t] for x in xs]
            out = interpret(p["jaxpr"], list(consts) + list(carry) + x_t, ctx)
            carry, y = out[:ncarry], out[ncarry:]
            ys.append((t, y))
        ys.sort(key=lambda z: z[0])
        stacked = [np.stack([obj(y[k]) for _, y in ys]) for k in range(len(out_avals) - ncarry)]
        return list(carry) + stacked

    # ---- PRNG keys: represented by their raw uint32 data, shape (..., 2) ----
    if name in ("random_wrap", "random_unwrap"):
        return [obj(ins[0])]
    if name == "random_fold_in":
        keys, msgs = obj(ins[0]), obj(ins[1])
        bshape = out_avals[0].shape
        kb = np.broadcast_to(keys, bshape + (2,))
        mb = np.broadcast_to(msgs, bshape)
        out = np.empty(bshape + (2,), dtype=object)
        for idx in np.ndindex(*bshape) if bshape else [()]:
            out[idx + (0,)], out[idx + (1,)] = _key_derive(ctx, kb[idx + (0,)], kb[idx + (1,)], _u32(mb[idx]))
        return [out]
    if name == "random_split":
        keys = obj(ins[0])
        shape = tuple(eqn.params["shape"])
        kshape = keys.shape[:-1]
        out = np.empty(kshape + shape + (2,), dtype=object)
        for kidx in np.ndindex(*kshape) if kshape else [()]:
            for f, idx in enumerate(np.ndindex(*shape) if shape else [()]):
                cnt = _counter(f, shape, ctx)
                out[kidx + idx + (0,)], out[kidx + idx + (1,)] = _key_derive(
                    ctx, keys[kidx + (0,)], keys[kidx + (1,)], cnt)
        return [out]
    if name == "random_bits":
        keys = obj(ins[0])
        if eqn.params["bit_width"] != 32:
            raise NotImplementedError("random_bits with bit_width != 32")
        shape = tuple(eqn.params["shape"])
        kshape = keys.shape[:-1]
        out = np.empty(kshape + shape, dtype=object)
        for kidx in np.ndindex(*kshape) if kshape else [()]:
            if shape:
                out[kidx] = _draw(ctx, keys[kidx + (0,)], keys[kidx + (1,)], shape)
            else:  # a scalar draw is shared, not one per particle: word 0 of call 0
                out[kidx] = _draw(Ctx(rng=ctx.rng), keys[kidx + (0,)], keys[kidx + (1,)], (1,))[0]
        return [out]

    raise NotImplementedError(name)


def _counter(f, shape, ctx):
    """Counter for flat index f of a draw of `shape` (threefry draws, and splits)."""
    if ctx.vectorized_J and shape and shape[-1] == 1:
        lead = f  # index over the leading (non-particle) dims, particle axis has length 1
        return as_dsl(np.uint32(lead * ctx.vectorized_J), np.uint32) + ctx.particle
    return np.uint32(f)
