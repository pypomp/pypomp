"""Translate a JAX function of one particle into CuTe DSL scalar code.

CuTe DSL is the Python interface to NVIDIA CUTLASS's CuTe library, used here to
write CUDA kernels. ``_interpret(closed_jaxpr, args, ctx)`` walks a jaxpr inside a
``@cute.kernel`` and returns, for each output, a numpy object array of DSL scalars
(or plain constants). Each jaxpr value keeps its small per-particle shape as an
object array, so slices, reshapes and reductions are bookkeeping at trace time, and
the emitted code is straight-line scalar math in registers, apart from branches
for runtime conds and calls of large sub-jaxprs that run more than once.

Random numbers are Philox4x32-10 with all four output words used, in the layout of
:mod:`pypomp.random.philox`: element (l, m) of a draw of shape (..., L) is word
l % 4 of the call with counter (l // 4) * L + m. An rproc written over all J
particles at once (``@vectorized``) is traced with J = 1, and ``ctx.vectorized_J``
and ``ctx.particle`` give the particle axis of a draw the counters that the full
(..., J) draw would use.

Unsupported primitives raise NotImplementedError naming the primitive.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import ClosedJaxpr  # pyright: ignore[reportPrivateImportUsage]
from jax.extend.core import Literal

import cutlass  # pyright: ignore[reportMissingImports]
import cutlass.cute as cute  # pyright: ignore[reportMissingImports]
from cutlass._mlir.dialects import arith as _arith  # pyright: ignore[reportMissingImports]
from cutlass._mlir import ir as _mlir_ir  # pyright: ignore[reportMissingImports]
from cutlass._mlir.dialects import llvm as _llvm  # pyright: ignore[reportMissingImports]

# A CuTe DSL scalar, or a numpy or Python constant.
_Scalar = Any

_DSL = {
    np.dtype("float32"): cutlass.Float32,
    np.dtype("int32"): cutlass.Int32,
    np.dtype("uint32"): cutlass.Uint32,
    np.dtype("bool"): cutlass.Boolean,
}

# Philox4x32-10 multipliers and key increments (Salmon et al., 2011).
_PM0, _PM1 = 0xD2511F53, 0xCD9E8D57
_PW0, _PW1 = np.uint32(0x9E3779B9), np.uint32(0xBB67AE85)


@dataclass
class _Ctx:
    """Where a translated function runs: its particle and the rproc's layout."""

    vectorized_J: int | None
    """J for a ``@vectorized`` rproc traced with J = 1, otherwise None."""
    particle: _Scalar
    """This thread's particle index within its filter, or None."""
    outlined: dict[Any, Any] = field(default_factory=dict)
    """The device functions emitted so far, by sub-jaxpr and constant inputs."""


def _is_dsl(x: Any) -> bool:
    """Return whether `x` is a CuTe DSL value rather than a constant."""
    return isinstance(x, cutlass.Numeric)


def _obj(a: Any) -> np.ndarray:
    """Return `a` as a numpy object array."""
    if isinstance(a, np.ndarray) and a.dtype == object:
        return a
    return np.asarray(a, dtype=object)


def _const(v: Any, dtype: Any) -> _Scalar:
    """Return the constant `v` as a DSL scalar of numpy dtype `dtype`."""
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        return cutlass.Boolean(bool(v))
    if np.issubdtype(dtype, np.floating):
        return _DSL[dtype](float(v))
    return _DSL[dtype](int(np.asarray(v).astype(dtype)))


def _as_dsl(x: _Scalar, dtype: Any) -> _Scalar:
    """Return `x` as a DSL scalar, converting a constant to `dtype`."""
    return x if _is_dsl(x) else _const(x, dtype)


def _select(p: _Scalar, t: _Scalar, f: _Scalar, dtype: Any) -> _Scalar:
    """Return `t` if `p` holds, otherwise `f`, for scalars of numpy dtype `dtype`."""
    if not _is_dsl(p):
        return t if bool(p) else f
    if not _is_dsl(t) and not _is_dsl(f) and np.asarray(t) == np.asarray(f):
        return t
    t, f = _as_dsl(t, dtype), _as_dsl(f, dtype)
    return _DSL[np.dtype(dtype)](
        _arith.select(p.ir_value(), t.ir_value(), f.ir_value())
    )


def _u32(x: _Scalar) -> _Scalar:
    """Return `x` as a uint32 constant unless it is a DSL value."""
    return x if _is_dsl(x) else np.uint32(x)


def _add(a: _Scalar, b: _Scalar) -> _Scalar:
    """Return a + b modulo 2**32 for uint32 scalars."""
    if _is_dsl(a) or _is_dsl(b):
        return _as_dsl(a, np.uint32) + _as_dsl(b, np.uint32)
    return np.uint32((int(a) + int(b)) & 0xFFFFFFFF)


def _xor(a: _Scalar, b: _Scalar) -> _Scalar:
    """Return the bitwise XOR of two uint32 scalars."""
    if _is_dsl(a) or _is_dsl(b):
        return _as_dsl(a, np.uint32) ^ _as_dsl(b, np.uint32)
    return np.uint32(a) ^ np.uint32(b)


def _mulhilo(m: int, x: _Scalar) -> tuple[_Scalar, _Scalar]:
    """Return the low and high 32 bits of m * x."""
    if not _is_dsl(x):
        p = m * int(x)
        return np.uint32(p & 0xFFFFFFFF), np.uint32(p >> 32)
    p = cutlass.Uint64(m) * x.to(cutlass.Uint64)
    return p.to(cutlass.Uint32), (p >> cutlass.Uint64(32)).to(cutlass.Uint32)


def _philox4x32(
    k0: _Scalar, k1: _Scalar, x0: _Scalar, x1: _Scalar, x2: _Scalar, x3: _Scalar
) -> tuple[_Scalar, _Scalar, _Scalar, _Scalar]:
    """Return the four Philox4x32-10 words for key (k0, k1) and counter (x0..x3)."""
    k0, k1, x0, x1, x2, x3 = (_u32(v) for v in (k0, k1, x0, x1, x2, x3))
    for r in range(10):
        if r:
            k0, k1 = _add(k0, _PW0), _add(k1, _PW1)
        lo0, hi0 = _mulhilo(_PM0, x0)
        lo1, hi1 = _mulhilo(_PM1, x2)
        x0, x1, x2, x3 = _xor(_xor(hi1, x1), k0), lo1, _xor(_xor(hi0, x3), k1), lo0
    return x0, x1, x2, x3


def _key_derive(k0: _Scalar, k1: _Scalar, cnt: _Scalar) -> tuple[_Scalar, _Scalar]:
    """Return the key that split and fold_in derive from key (k0, k1) and `cnt`."""
    y = _philox4x32(k0, k1, 0, 0, 0, cnt)
    return y[0], y[1]


def _draw(ctx: _Ctx, k0: _Scalar, k1: _Scalar, shape: tuple[int, ...]) -> np.ndarray:
    """Return the 32-bit words of a draw of `shape` from key (k0, k1)."""
    out = np.empty(shape, dtype=object)
    n = math.prod(shape)
    if n == 0:
        return out
    vec = bool(ctx.vectorized_J)
    if len(shape) >= 2:
        lane = shape[-1]
        flat = out.reshape(-1, lane)
        lead = flat.shape[0]
        if ctx.vectorized_J and lane == 1:  # the particle axis, of length J at run time
            lanes = [(0, ctx.particle, ctx.vectorized_J)]
        else:
            lanes = [(m, np.uint32(m), lane) for m in range(lane)]
        for col, m, L in lanes:
            words = {}
            for l in range(lead):
                g = l // 4
                if g not in words:
                    words[g] = _philox4x32(k0, k1, _add(np.uint32(g * L), m), 0, 0, 0)
                flat[l, col] = words[g][l % 4]
        return out
    flat = out.reshape(-1)
    if vec and shape == (1,):  # element j of a (J,) draw over particles
        j = ctx.particle
        w = _philox4x32(k0, k1, j >> cutlass.Uint32(2), 0, 0, 0)
        r = j & cutlass.Uint32(3)
        flat[0] = _select(
            r == cutlass.Uint32(0),
            w[0],
            _select(
                r == cutlass.Uint32(1),
                w[1],
                _select(r == cutlass.Uint32(2), w[2], w[3], np.uint32),
                np.uint32,
            ),
            np.uint32,
        )
        return out
    words = {}
    for f in range(n):
        g = f // 4
        if g not in words:
            words[g] = _philox4x32(k0, k1, np.uint32(g), 0, 0, 0)
        flat[f] = words[g][f % 4]
    return out


def _erf_inv32(x: _Scalar) -> _Scalar:
    """Return erfinv(x) by Giles' single-precision approximation, as XLA computes it."""
    w = -cute.math.log((cutlass.Float32(1.0) - x) * (cutlass.Float32(1.0) + x))
    lt = w < cutlass.Float32(5.0)
    w1 = w - cutlass.Float32(2.5)
    w2 = cute.math.sqrt(w) - cutlass.Float32(3.0)
    c1 = [
        2.81022636e-08,
        3.43273939e-07,
        -3.5233877e-06,
        -4.39150654e-06,
        0.00021858087,
        -0.00125372503,
        -0.00417768164,
        0.246640727,
        1.50140941,
    ]
    c2 = [
        -0.000200214257,
        0.000100950558,
        0.00134934322,
        -0.00367342844,
        0.00573950773,
        -0.0076224613,
        0.00943887047,
        1.00167406,
        2.83297682,
    ]
    w = _select(lt, w1, w2, np.float32)
    p = _select(lt, cutlass.Float32(c1[0]), cutlass.Float32(c2[0]), np.float32)
    for a, b in zip(c1[1:], c2[1:], strict=True):
        p = _select(lt, cutlass.Float32(a), cutlass.Float32(b), np.float32) + p * w
    return p * x


def _ir(op: str, dtype: Any, *xs: _Scalar) -> _Scalar:
    """Return MLIR ``arith`` op `op` on DSL scalars, as a DSL scalar of `dtype`.

    Used where the DSL's Python operators follow Python's semantics (floor
    division, for one) rather than XLA's.
    """
    return _DSL[np.dtype(dtype)](getattr(_arith, op)(*(x.ir_value() for x in xs)))


def _shift(name: str, x: _Scalar, s: _Scalar, dtype: Any) -> _Scalar:
    """Return `x` shifted by `s` as XLA defines it, including shifts by a negative
    amount or by at least the bit width, which LLVM leaves undefined."""
    dt = np.dtype(dtype)
    bits = dt.itemsize * 8
    op = {
        "shift_left": "shli",
        "shift_right_logical": "shrui",
        "shift_right_arithmetic": "shrsi",
    }[name]
    out = s >= _const(bits, dt)
    if np.issubdtype(dt, np.signedinteger):
        out = out | (s < _const(0, dt))
    fill = (
        _ir("shrsi", dt, x, _const(bits - 1, dt))
        if name == "shift_right_arithmetic"
        else _const(0, dt)
    )
    return _select(out, fill, _ir(op, dt, x, s), dt)


def _ipow(x: _Scalar, n: int) -> _Scalar:
    """Return x**n for an integer n by repeated squaring."""
    if n == 0:
        return cutlass.Float32(1.0) if _is_dsl(x) else 1.0
    neg, n = n < 0, abs(n)
    acc, base = None, x
    while n:
        if n & 1:
            acc = base if acc is None else acc * base
        n >>= 1
        if n:
            base = base * base
    return (cutlass.Float32(1.0) / acc) if neg else acc


def _ew(
    name: str,
    args: list[_Scalar],
    params: dict[str, Any],
    out_dtype: Any,
    in_dtypes: list[Any],
) -> _Scalar:
    """Return elementwise primitive `name` on scalars, or None if all are constants."""
    if not any(_is_dsl(a) for a in args):
        return None
    a = [_as_dsl(v, d) for v, d in zip(args, in_dtypes, strict=True)]
    x = a[0]
    f = np.issubdtype(np.dtype(in_dtypes[0]), np.floating)
    m = cute.math
    if name in ("add", "add_any"):
        return x + a[1]
    if name == "sub":
        return x - a[1]
    if name == "mul":
        return x * a[1]
    signed = np.issubdtype(np.dtype(in_dtypes[0]), np.signedinteger)
    if name == "div":  # integer division truncates toward zero, as in C
        return x / a[1] if f else _ir("divsi" if signed else "divui", in_dtypes[0], *a)
    if name == "rem":  # the sign of the dividend, as in C's fmod and %
        if f:
            return _ir("remf", in_dtypes[0], *a)
        return _ir("remsi" if signed else "remui", in_dtypes[0], *a)
    if name == "neg":
        return -x
    if name == "max":
        return m.max(x, a[1]) if f else _select(x > a[1], x, a[1], in_dtypes[0])
    if name == "min":
        return m.min(x, a[1]) if f else _select(x < a[1], x, a[1], in_dtypes[0])
    if name == "abs":
        return (
            m.abs(x) if f else _select(x < _const(0, in_dtypes[0]), -x, x, in_dtypes[0])
        )
    if name == "sign":  # x itself for zeros and NaN, as XLA returns
        z = _const(0, in_dtypes[0])
        return _select(
            x > z,
            _const(1, in_dtypes[0]),
            _select(x < z, _const(-1, in_dtypes[0]), x, in_dtypes[0]),
            in_dtypes[0],
        )
    if name in (
        "exp",
        "log",
        "sqrt",
        "rsqrt",
        "expm1",
        "log1p",
        "tanh",
        "sin",
        "cos",
        "erf",
        "erfc",
        "floor",
        "ceil",
        "exp2",
        "log2",
    ):
        return getattr(m, name)(x)
    if name == "round":
        return (
            m.round(x)
            if params.get("rounding_method", 0) in (0, "0")
            or str(params.get("rounding_method")).endswith("AWAY_FROM_ZERO")
            else m.roundeven(x)
        )
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
        return {
            "eq": x == a[1],
            "ne": x != a[1],
            "lt": x < a[1],
            "le": x <= a[1],
            "gt": x > a[1],
            "ge": x >= a[1],
        }[name]
    if name == "and":
        return x & a[1]
    if name == "or":
        return x | a[1]
    if name == "xor":
        return x ^ a[1]
    if name == "not":
        return ~x
    if name in ("shift_left", "shift_right_logical", "shift_right_arithmetic"):
        return _shift(name, x, a[1], in_dtypes[0])
    if name == "convert_element_type":
        return x.to(_DSL[np.dtype(out_dtype)])
    if name == "bitcast_convert_type":
        return x.bitcast(_DSL[np.dtype(out_dtype)])
    if name == "clamp":  # clamp(lo, x, hi)
        lo, v, hi = a
        return (
            m.min(m.max(v, lo), hi)
            if np.issubdtype(np.dtype(in_dtypes[1]), np.floating)
            else _select(v < lo, lo, _select(v > hi, hi, v, in_dtypes[1]), in_dtypes[1])
        )
    if name == "nextafter":
        raise NotImplementedError("nextafter")
    raise NotImplementedError(name)


_ELEMENTWISE = {
    "add",
    "add_any",
    "sub",
    "mul",
    "div",
    "rem",
    "neg",
    "max",
    "min",
    "abs",
    "sign",
    "exp",
    "log",
    "sqrt",
    "rsqrt",
    "expm1",
    "log1p",
    "tanh",
    "sin",
    "cos",
    "erf",
    "erfc",
    "floor",
    "ceil",
    "exp2",
    "log2",
    "round",
    "pow",
    "integer_pow",
    "square",
    "logistic",
    "erf_inv",
    "is_finite",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "and",
    "or",
    "xor",
    "not",
    "shift_left",
    "shift_right_logical",
    "shift_right_arithmetic",
    "convert_element_type",
    "bitcast_convert_type",
    "clamp",
}
_IDENTITY = {
    "stop_gradient",
    "optimization_barrier",
    "copy",
    "copy_p",
    "pvary",
}
_CALLS = {
    "pjit",
    "jit",
    "closed_call",
    "core_call",
    "remat",
    "checkpoint",
    "custom_jvp_call",
    "custom_vjp_call",
    "custom_vjp_call_jaxpr",
    "custom_lin",
}
_SUPPORTED = (
    _ELEMENTWISE
    | _IDENTITY
    | _CALLS
    | {
        "select_n",
        "polynomial",
        "empty",
        "empty2",
        "pad",
        "broadcast_in_dim",
        "reshape",
        "squeeze",
        "expand_dims",
        "transpose",
        "slice",
        "dynamic_slice",
        "dynamic_update_slice",
        "concatenate",
        "split",
        "unstack",
        "stack",
        "iota",
        "reduce_sum",
        "reduce_prod",
        "reduce_max",
        "reduce_min",
        "reduce_and",
        "reduce_or",
        "dot_general",
        "cond",
        "while",
        "scan",
        "random_wrap",
        "random_unwrap",
        "random_fold_in",
        "random_split",
        "random_bits",
    }
)


def _fold(eqn: Any, vals: list[np.ndarray]) -> list[np.ndarray]:
    """Evaluate equation `eqn` on constant inputs, eagerly on the CPU."""
    # This runs while JAX traces the CuTe call, so evaluation must be forced eager.
    with jax.ensure_compile_time_eval(), jax.default_device(jax.devices("cpu")[0]):
        out = eqn.primitive.bind(*[jnp.asarray(v) for v in vals], **eqn.params)
    outs = out if eqn.primitive.multiple_results else [out]
    return [np.asarray(o) for o in outs]


def _sub_jaxpr(eqn: Any) -> ClosedJaxpr:
    """Return the jaxpr that a call primitive's equation `eqn` calls."""
    for k in ("jaxpr", "call_jaxpr", "fun_jaxpr"):
        if k in eqn.params:
            j = eqn.params[k]
            return j if hasattr(j, "consts") else ClosedJaxpr(j, ())
    raise NotImplementedError(f"{eqn.primitive.name}: no sub-jaxpr")


def _all_const(vals: list[Any]) -> bool:
    """Return whether no value in `vals` holds a DSL scalar."""
    return all(
        not (
            isinstance(v, np.ndarray)
            and v.dtype == object
            and any(_is_dsl(e) for e in v.flat)
        )
        and not _is_dsl(v)
        for v in vals
    )


def _as_np_const(v: Any, dtype: Any) -> np.ndarray:
    """Return the constant `v` as a numpy array of `dtype`."""
    v = np.asarray(v)
    if v.dtype == object:
        v = v.astype(dtype)
    return v


def _interpret(cj: ClosedJaxpr, args: list[Any], ctx: _Ctx) -> list[np.ndarray]:
    """Evaluate `cj` on object arrays, emitting DSL code, and return its outputs."""
    jaxpr = cj.jaxpr
    env = {}

    def read(v):
        if isinstance(v, Literal):
            return np.asarray(v.val)
        return env[v]

    for v, c in zip(jaxpr.constvars, cj.consts, strict=True):
        env[v] = np.asarray(c)
    for v, a in zip(jaxpr.invars, args, strict=True):
        env[v] = a

    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        ins = [read(v) for v in eqn.invars]
        in_dt = [getattr(v.aval, "dtype", None) for v in eqn.invars]
        out_avals = [v.aval for v in eqn.outvars]
        outs = _eval_eqn(eqn, name, ins, in_dt, out_avals, ctx)
        for v, o in zip(eqn.outvars, outs, strict=True):
            env[v] = o
    return [read(v) for v in jaxpr.outvars]


def _is_key(dtype: Any) -> bool:
    """Return whether `dtype` is a JAX PRNG key dtype."""
    return dtype is not None and jnp.issubdtype(dtype, jax.dtypes.prng_key)


def _eval_eqn(
    eqn: Any,
    name: str,
    ins: list[Any],
    in_dt: list[Any],
    out_avals: list[Any],
    ctx: _Ctx,
) -> list[Any]:
    """Return the outputs of one jaxpr equation, emitting DSL code as needed."""
    if name in ("empty", "empty2"):
        # Unset buffers, such as residuals that a differentiated cond's branch
        # doesn't produce; keys among them are carried as their data.
        av = out_avals[0]
        key = _is_key(av.dtype)
        return [
            np.zeros(av.shape + ((2,) if key else ()), np.uint32 if key else av.dtype)
        ]
    if (
        _all_const(ins)
        and name not in _CALLS
        and name not in ("cond", "while", "scan")
        and not any(_is_key(d) for d in in_dt)
        and not any(_is_key(a.dtype) for a in out_avals)
    ):
        return _fold(eqn, [_as_np_const(v, d) for v, d in zip(ins, in_dt, strict=True)])

    if name in _IDENTITY:
        return ins[: len(out_avals)]

    if name in _ELEMENTWISE:
        shape = out_avals[0].shape
        b = [np.broadcast_to(_obj(v), shape) for v in ins]
        out = np.empty(shape, dtype=object)
        for idx in np.ndindex(*shape) if shape else [()]:
            elems = [x[idx] for x in b]
            r = _ew(name, elems, eqn.params, out_avals[0].dtype, in_dt)
            if r is None:
                r = _fold(
                    eqn,
                    [
                        np.asarray(e).astype(d)
                        for e, d in zip(elems, in_dt, strict=True)
                    ],
                )[0][()]
            out[idx] = r
        return [out]

    if name == "select_n":
        pred, cases = ins[0], ins[1:]
        shape = out_avals[0].shape
        dt = out_avals[0].dtype
        p = np.broadcast_to(_obj(pred), shape)
        cs = [np.broadcast_to(_obj(c), shape) for c in cases]
        out = np.empty(shape, dtype=object)
        for idx in np.ndindex(*shape) if shape else [()]:
            pi = p[idx]
            if not _is_dsl(pi):
                out[idx] = cs[int(pi)][idx]
                continue
            if len(cs) == 2 and np.dtype(in_dt[0]) == np.bool_:
                out[idx] = _select(pi, cs[1][idx], cs[0][idx], dt)
            else:
                r = cs[-1][idx]
                for c in range(len(cs) - 2, -1, -1):
                    r = _select(pi == _const(c, in_dt[0]), cs[c][idx], r, dt)
                out[idx] = r
        return [out]

    if name == "polynomial":  # coefficients lowest degree first
        x, coeffs = ins[0], ins[1:]
        shape = out_avals[0].shape
        xb = np.broadcast_to(_obj(x), shape)
        cb = [np.broadcast_to(_obj(c), shape) for c in coeffs]
        out = np.empty(shape, dtype=object)
        for idx in np.ndindex(*shape) if shape else [()]:
            r = _as_dsl(cb[-1][idx], np.float32)
            for c in reversed(cb[:-1]):
                r = _as_dsl(c[idx], np.float32) + r * _as_dsl(xb[idx], np.float32)
            out[idx] = r
        return [out]

    if name == "broadcast_in_dim":
        x = _obj(ins[0])
        shape, bdims = eqn.params["shape"], eqn.params["broadcast_dimensions"]
        expanded = [1] * len(shape)
        for i, d in enumerate(bdims):
            expanded[d] = x.shape[i]
        return [np.broadcast_to(x.reshape(expanded), shape).copy()]
    if name == "reshape":
        x = _obj(ins[0])
        if eqn.params.get("dimensions") is not None:
            x = np.transpose(x, eqn.params["dimensions"])
        return [x.reshape(eqn.params["new_sizes"])]
    if name == "squeeze":
        return [np.squeeze(_obj(ins[0]), axis=tuple(eqn.params["dimensions"]))]
    if name == "expand_dims":
        return [np.expand_dims(_obj(ins[0]), eqn.params["dimensions"])]
    if name == "transpose":
        return [np.transpose(_obj(ins[0]), eqn.params["permutation"])]
    if name == "slice":
        st, lim, strd = (
            eqn.params["start_indices"],
            eqn.params["limit_indices"],
            eqn.params["strides"],
        )
        strd = strd or (1,) * len(st)
        return [
            _obj(ins[0])[
                tuple(slice(a, b, c) for a, b, c in zip(st, lim, strd, strict=True))
            ]
        ]
    if name == "dynamic_slice":
        starts = ins[1:]
        if not _all_const(starts):
            raise NotImplementedError("dynamic_slice with a runtime index")
        x = _obj(ins[0])
        sizes = eqn.params["slice_sizes"]
        st = [
            int(np.clip(int(s), 0, n - k))
            for s, n, k in zip(starts, x.shape, sizes, strict=True)
        ]
        return [x[tuple(slice(a, a + k) for a, k in zip(st, sizes, strict=True))]]
    if name == "dynamic_update_slice":
        x, upd, starts = _obj(ins[0]).copy(), _obj(ins[1]), ins[2:]
        if not _all_const(starts):
            raise NotImplementedError("dynamic_update_slice with a runtime index")
        st = [
            int(np.clip(int(s), 0, n - k))
            for s, n, k in zip(starts, x.shape, upd.shape, strict=True)
        ]
        x[tuple(slice(a, a + k) for a, k in zip(st, upd.shape, strict=True))] = upd
        return [x]
    if name == "pad":  # also the transpose of slice in gradients
        x, pv = _obj(ins[0]), _obj(ins[1]).reshape(())[()]
        cfg = eqn.params["padding_config"]
        shape = out_avals[0].shape
        out = np.empty(shape, dtype=object)
        for idx in np.ndindex(*shape) if shape else [()]:
            out[idx] = pv
        for idx in np.ndindex(*x.shape) if x.shape else [()]:
            pos = tuple(
                lo + i * (interior + 1)
                for i, (lo, _, interior) in zip(idx, cfg, strict=True)
            )
            if all(0 <= q < n for q, n in zip(pos, shape, strict=True)):
                out[pos] = x[idx]
        return [out]
    if name == "concatenate":
        return [np.concatenate([_obj(v) for v in ins], axis=eqn.params["dimension"])]
    if name == "split":
        x = _obj(ins[0])
        cuts = np.cumsum(eqn.params["sizes"])[:-1]
        return list(np.split(x, cuts, axis=eqn.params["axis"]))
    if name == "unstack":
        x = _obj(ins[0])
        ax = eqn.params["axis"]
        return [np.take(x, i, axis=ax) for i in range(x.shape[ax])]
    if name == "stack":
        return [np.stack([_obj(v) for v in ins], axis=eqn.params["axis"])]
    if name == "iota":
        shape, dim = eqn.params["shape"], eqn.params["dimension"]
        a = np.arange(shape[dim], dtype=eqn.params["dtype"])
        sh = [1] * len(shape)
        sh[dim] = shape[dim]
        return [np.broadcast_to(a.reshape(sh), shape).copy()]
    if name in (
        "reduce_sum",
        "reduce_prod",
        "reduce_max",
        "reduce_min",
        "reduce_and",
        "reduce_or",
    ):
        x = _obj(ins[0])
        axes = tuple(eqn.params["axes"])
        dt = out_avals[0].dtype
        op = {
            "reduce_sum": "add",
            "reduce_prod": "mul",
            "reduce_max": "max",
            "reduce_min": "min",
            "reduce_and": "and",
            "reduce_or": "or",
        }[name]
        keep = [d for d in range(x.ndim) if d not in axes]
        xt = np.transpose(x, keep + list(axes)).reshape(
            [x.shape[d] for d in keep] + [-1]
        )
        red: np.ndarray = np.empty(xt.shape[:-1], dtype=object)
        for idx in np.ndindex(*red.shape) if red.shape else [()]:
            r = xt[idx][0]
            for e in xt[idx][1:]:
                rr = _ew(op, [r, e], {}, dt, [dt, dt])
                r = (
                    rr
                    if rr is not None
                    else np.asarray(
                        {
                            "add": np.add,
                            "mul": np.multiply,
                            "max": np.maximum,
                            "min": np.minimum,
                            "and": np.logical_and,
                            "or": np.logical_or,
                        }[op](r, e)
                    ).astype(dt)
                )
            red[idx] = r
        return [red]
    if name == "dot_general":
        (lc, rc), (lb, rb) = eqn.params["dimension_numbers"]
        if lb or rb:
            raise NotImplementedError("batched dot_general")
        a, b = _obj(ins[0]), _obj(ins[1])
        dt = out_avals[0].dtype
        af = [d for d in range(a.ndim) if d not in lc]
        bf = [d for d in range(b.ndim) if d not in rc]
        at = np.transpose(a, af + list(lc)).reshape([a.shape[d] for d in af] + [-1])
        bt = np.transpose(b, bf + list(rc)).reshape([b.shape[d] for d in bf] + [-1])
        out = np.empty(out_avals[0].shape, dtype=object)
        for i in np.ndindex(*at.shape[:-1]) if at.ndim > 1 else [()]:
            for k in np.ndindex(*bt.shape[:-1]) if bt.ndim > 1 else [()]:
                r = None
                for x, y in zip(at[i], bt[k], strict=True):
                    p = _ew("mul", [x, y], {}, dt, [dt, dt])
                    if p is None:
                        p = np.asarray(np.asarray(x, dt) * np.asarray(y, dt), dt)
                    if r is None:
                        r = p
                    else:
                        rp = _ew("add", [r, p], {}, dt, [dt, dt])
                        r = (
                            rp
                            if rp is not None
                            else np.asarray(np.asarray(r, dt) + np.asarray(p, dt), dt)
                        )
                out[i + k] = r
        return [out]

    if name in _CALLS:
        sub = _sub_jaxpr(eqn)
        if _outlinable(sub, ins):
            return _call_outlined(sub, ins, out_avals, ctx)
        return _interpret(sub, ins, ctx)
    if name == "cond":
        index, ops = ins[0], ins[1:]
        branches = eqn.params["branches"]
        idx = _obj(index).reshape(())[()] if np.ndim(index) == 0 else None
        if idx is None:
            raise NotImplementedError("cond with a batched index")
        if not _is_dsl(idx):
            return _interpret(branches[int(idx)], ops, ctx)
        return _branch(idx, in_dt[0], branches, ops, out_avals, ctx)
    if name == "while":
        cn, bn = eqn.params["cond_nconsts"], eqn.params["body_nconsts"]
        cconsts, bconsts, carry = ins[:cn], ins[cn : cn + bn], ins[cn + bn :]
        for _ in range(100000):
            (c,) = _interpret(
                eqn.params["cond_jaxpr"], list(cconsts) + list(carry), ctx
            )
            c = _obj(c).reshape(())[()]
            if _is_dsl(c):
                raise NotImplementedError("while loop with a runtime trip count")
            if not bool(c):
                return carry
            carry = _interpret(
                eqn.params["body_jaxpr"], list(bconsts) + list(carry), ctx
            )
        raise NotImplementedError("while loop did not terminate during translation")
    if name == "scan":
        p = eqn.params
        nc, ncarry, length = p["num_consts"], p["num_carry"], p["length"]
        consts, carry, xs = ins[:nc], ins[nc : nc + ncarry], ins[nc + ncarry :]
        ys = []
        order = range(length - 1, -1, -1) if p["reverse"] else range(length)
        for t in order:
            x_t = [_obj(x)[t] for x in xs]
            out = _interpret(p["jaxpr"], list(consts) + list(carry) + x_t, ctx)
            carry, y = out[:ncarry], out[ncarry:]
            ys.append((t, y))
        ys.sort(key=lambda z: z[0])
        stacked = [
            np.stack([_obj(y[k]) for _, y in ys])
            for k in range(len(out_avals) - ncarry)
        ]
        return list(carry) + stacked

    # PRNG keys are represented by their uint32 data, of shape (..., 2).
    if name in ("random_wrap", "random_unwrap"):
        return [_obj(ins[0])]
    if name == "random_fold_in":
        keys, msgs = _obj(ins[0]), _obj(ins[1])
        bshape = out_avals[0].shape
        kb = np.broadcast_to(keys, bshape + (2,))
        mb = np.broadcast_to(msgs, bshape)
        out = np.empty(bshape + (2,), dtype=object)
        for idx in np.ndindex(*bshape) if bshape else [()]:
            out[idx + (0,)], out[idx + (1,)] = _key_derive(
                kb[idx + (0,)], kb[idx + (1,)], _u32(mb[idx])
            )
        return [out]
    if name == "random_split":
        keys = _obj(ins[0])
        shape = tuple(eqn.params["shape"])
        kshape = keys.shape[:-1]
        out = np.empty(kshape + shape + (2,), dtype=object)
        for kidx in np.ndindex(*kshape) if kshape else [()]:
            for f, idx in enumerate(np.ndindex(*shape) if shape else [()]):
                cnt = _counter(f, shape, ctx)
                out[kidx + idx + (0,)], out[kidx + idx + (1,)] = _key_derive(
                    keys[kidx + (0,)], keys[kidx + (1,)], cnt
                )
        return [out]
    if name == "random_bits":
        keys = _obj(ins[0])
        if eqn.params["bit_width"] != 32:
            raise NotImplementedError("random_bits with bit_width != 32")
        shape = tuple(eqn.params["shape"])
        kshape = keys.shape[:-1]
        out = np.empty(kshape + shape, dtype=object)
        for kidx in np.ndindex(*kshape) if kshape else [()]:
            if shape:
                out[kidx] = _draw(ctx, keys[kidx + (0,)], keys[kidx + (1,)], shape)
            else:  # a scalar draw is shared, not one per particle: word 0 of call 0
                out[kidx] = _draw(
                    _Ctx(vectorized_J=None, particle=None),
                    keys[kidx + (0,)],
                    keys[kidx + (1,)],
                    (1,),
                )[0]
        return [out]

    raise NotImplementedError(name)


def _branch(
    idx: _Scalar,
    idx_dtype: Any,
    branches: list[ClosedJaxpr],
    ops: list[Any],
    out_avals: list[Any],
    ctx: _Ctx,
) -> list[np.ndarray]:
    """Return the outputs of the branch that runtime index `idx` picks, running only
    that branch.

    Each thread branches on its own index. A warp whose threads pick different
    branches runs each of them in turn, so this costs at most what computing every
    branch and selecting would, and skips the branches no thread in a warp needs.
    """
    # Keys are carried as their uint32 data, with a trailing axis of 2.
    shapes = [av.shape + ((2,) if _is_key(av.dtype) else ()) for av in out_avals]
    dtypes = [
        np.dtype(np.uint32 if _is_key(av.dtype) else av.dtype) for av in out_avals
    ]
    types = [
        _DSL[dt]
        for sh, dt in zip(shapes, dtypes, strict=True)
        for _ in range(math.prod(sh))
    ]

    def body(b: int) -> Callable[[], list[_Scalar]]:
        def run() -> list[_Scalar]:
            # An empty asm with side effects can't be speculated, so LLVM keeps the
            # branch instead of if-converting it back into running every branch and
            # selecting, which it otherwise does for side-effect-free branches.
            _llvm.inline_asm(None, [], "", "", has_side_effects=True)
            outs = _interpret(branches[b], ops, ctx)
            return [
                _as_dsl(v, dt)
                for o, sh, dt in zip(outs, shapes, dtypes, strict=True)
                for v in np.broadcast_to(_obj(o), sh).flat
            ]

        return run

    def pick(b: int) -> list[_Scalar]:
        if b == len(branches) - 1:
            return body(b)()
        r = cutlass.if_generate(
            idx == _const(b, idx_dtype), body(b), lambda: pick(b + 1), [], types
        )
        return [r] if len(types) == 1 else list(r)

    flat = iter(pick(0))
    outs = []
    for sh in shapes:
        out = np.empty(sh, dtype=object)
        for p in np.ndindex(*sh) if sh else [()]:
            out[p] = next(flat)
        outs.append(out)
    return outs


# Emitted inline, every call of a sampler gets its own copy of the sampler's code,
# and a step with several no longer fits in the instruction cache. So a call of a
# sub-jaxpr with at least this many equations, counting nested ones, becomes a
# device function, which is kept out of line once it is called twice.
_OUTLINE_MIN_EQNS = 60
_RANDOM = {
    "random_bits",
    "random_split",
    "random_fold_in",
    "random_wrap",
    "random_seed",
}
_LLVM_TYPE = {
    np.dtype("float32"): "f32",
    np.dtype("int32"): "i32",
    np.dtype("uint32"): "i32",
    np.dtype("bool"): "i1",
}


@dataclass
class _Outlined:
    """A sub-jaxpr emitted as an ``llvm.func`` for one pattern of constant inputs."""

    name: str
    func: Any
    struct: Any
    outs: list[np.ndarray]
    """Its outputs: constants where the body computes one, else the index of the
    result in the returned struct and its numpy dtype."""
    cj: ClosedJaxpr
    """Held so that its id, in the cache key, isn't reused."""
    calls: list[Any] = field(default_factory=list)


def _size(cj: Any) -> int:
    """Return the number of equations in `cj`, counting those of nested jaxprs."""
    jaxpr = getattr(cj, "jaxpr", cj)
    n = 0
    for eqn in jaxpr.eqns:
        n += 1
        if eqn.primitive.name in _CALLS:
            n += _size(_sub_jaxpr(eqn))
        for b in eqn.params.get("branches", ()):
            n += _size(b)
    return n


def _uses_random(cj: Any) -> bool:
    """Return whether `cj` or a jaxpr it calls draws or derives random numbers."""
    jaxpr = getattr(cj, "jaxpr", cj)
    for eqn in jaxpr.eqns:
        if eqn.primitive.name in _RANDOM:
            return True
        subs = [_sub_jaxpr(eqn)] if eqn.primitive.name in _CALLS else []
        subs += list(eqn.params.get("branches", ()))
        for k in ("cond_jaxpr", "body_jaxpr"):
            if k in eqn.params:
                subs.append(eqn.params[k])
        if eqn.primitive.name == "scan":
            subs.append(eqn.params["jaxpr"])
        if any(_uses_random(b) for b in subs):
            return True
    return False


def _outlinable(cj: ClosedJaxpr, ins: list[Any]) -> bool:
    """Return whether a call of `cj` on `ins` should become a device function.

    Random draws read the thread's particle index from the caller, so jaxprs that
    draw stay inline, as do keys, which aren't plain scalars.
    """
    if _all_const(ins):
        return False
    avals = [v.aval for v in [*cj.jaxpr.invars, *cj.jaxpr.outvars]]
    if any(_is_key(a.dtype) or np.dtype(a.dtype) not in _LLVM_TYPE for a in avals):
        return False
    return _size(cj) >= _OUTLINE_MIN_EQNS and not _uses_random(cj)


def _gpu_module() -> Any:
    """Return the ``gpu.module`` that the code being emitted belongs to."""
    op = _mlir_ir.InsertionPoint.current.block.owner.operation
    while op.name != "gpu.module":
        op = op.parent
    return op


def _call_outlined(
    cj: ClosedJaxpr, ins: list[Any], out_avals: list[Any], ctx: _Ctx
) -> list[np.ndarray]:
    """Return the outputs of `cj` on `ins`, computed by a call to a device function.

    Runtime input elements become the function's arguments and constant ones are
    folded into its body, so each pattern of constants gets its own function.
    """
    in_dt = [np.dtype(getattr(v.aval, "dtype", None)) for v in cj.jaxpr.invars]
    flat = [
        np.broadcast_to(_obj(a), _shape(v))
        for a, v in zip(ins, cj.jaxpr.invars, strict=True)
    ]
    pattern = tuple(
        tuple(None if _is_dsl(x) else np.asarray(x, dt).tobytes() for x in a.flat)
        for a, dt in zip(flat, in_dt, strict=True)
    )
    args = [x for a in flat for x in a.flat if _is_dsl(x)]
    key = (id(cj), pattern)
    fn = ctx.outlined.get(key)
    if fn is None:
        fn = _define(cj, flat, in_dt, out_avals, ctx)
        ctx.outlined[key] = fn
    call = _llvm.CallOp(fn.struct, [a.ir_value() for a in args], [], [], callee=fn.name)
    fn.calls.append(call)
    if len(fn.calls) == 2:
        fn.func.no_inline = True
    outs = []
    for o in fn.outs:
        out = np.empty(o.shape, dtype=object)
        for idx in np.ndindex(*o.shape) if o.shape else [()]:
            v = o[idx]
            if isinstance(v, tuple):
                i, dt = v
                ty = _mlir_ir.Type.parse(_LLVM_TYPE[dt])
                out[idx] = _DSL[dt](_llvm.ExtractValueOp(ty, call.result, [i]).result)
            else:
                out[idx] = v
        outs.append(out)
    return outs


def _define(
    cj: ClosedJaxpr,
    flat: list[np.ndarray],
    in_dt: list[np.dtype],
    out_avals: list[Any],
    ctx: _Ctx,
) -> _Outlined:
    """Emit `cj` as an ``llvm.func`` taking the runtime elements of `flat`."""
    arg_dt = [
        dt for a, dt in zip(flat, in_dt, strict=True) for x in a.flat if _is_dsl(x)
    ]
    module = _gpu_module()
    # Unique within the module, which may hold functions from other translations.
    name = f"pypomp_fn{len(module.regions[0].blocks[0].operations)}"
    with _mlir_ir.InsertionPoint.at_block_begin(module.regions[0].blocks[0]):
        # The result type is set once the body shows which outputs are runtime values.
        placeholder = _mlir_ir.Type.parse("!llvm.struct<()>")
        ftype = _mlir_ir.Type.parse(
            f"!llvm.func<{placeholder} ("
            + ", ".join(_LLVM_TYPE[d] for d in arg_dt)
            + ")>"
        )
        func = _llvm.LLVMFuncOp(
            name,
            _mlir_ir.TypeAttr.get(ftype),
            linkage=_mlir_ir.Attribute.parse("#llvm.linkage<internal>"),
        )
        entry = func.body.blocks.append(
            *[_mlir_ir.Type.parse(_LLVM_TYPE[d]) for d in arg_dt]
        )
        with _mlir_ir.InsertionPoint(entry):
            it = iter(entry.arguments)
            sub_ins = []
            for a, dt in zip(flat, in_dt, strict=True):
                x = np.empty(a.shape, dtype=object)
                for idx in np.ndindex(*a.shape) if a.shape else [()]:
                    x[idx] = _DSL[dt](next(it)) if _is_dsl(a[idx]) else a[idx]
                sub_ins.append(x)
            body_ctx = _Ctx(vectorized_J=None, particle=None, outlined=ctx.outlined)
            res = _interpret(cj, sub_ins, body_ctx)
            outs, vals = [], []
            for r, av in zip(res, out_avals, strict=True):
                dt = np.dtype(av.dtype)
                rb = np.broadcast_to(_obj(r), av.shape)
                o = np.empty(av.shape, dtype=object)
                for idx in np.ndindex(*av.shape) if av.shape else [()]:
                    v = rb[idx]
                    if _is_dsl(v):
                        o[idx] = (len(vals), dt)
                        vals.append((v, dt))
                    else:
                        o[idx] = v
                outs.append(o)
            struct = _mlir_ir.Type.parse(
                "!llvm.struct<(" + ", ".join(_LLVM_TYPE[dt] for _, dt in vals) + ")>"
            )
            s = _llvm.UndefOp(struct).result
            for i, (v, dt) in enumerate(vals):
                s = _llvm.InsertValueOp(s, _as_dsl(v, dt).ir_value(), [i]).result
            _llvm.ReturnOp(arg=s)
        ftype = _mlir_ir.Type.parse(
            f"!llvm.func<{struct} (" + ", ".join(_LLVM_TYPE[d] for d in arg_dt) + ")>"
        )
        func.attributes["function_type"] = _mlir_ir.TypeAttr.get(ftype)
    return _Outlined(name, func, struct, outs, cj)


def _counter(f: int, shape: tuple[int, ...], ctx: _Ctx) -> _Scalar:
    """Return the counter of key `f` in a split of `shape`."""
    if ctx.vectorized_J and shape and shape[-1] == 1:
        # f indexes the leading axes; the particle axis has length J at run time.
        return _as_dsl(np.uint32(f * ctx.vectorized_J), np.uint32) + ctx.particle
    return np.uint32(f)


# Checks run before CuTe is chosen, for uses the translator can't run exactly.

_REDUCTIONS = {
    "reduce_sum",
    "reduce_prod",
    "reduce_max",
    "reduce_min",
    "reduce_and",
    "reduce_or",
    "argmax",
    "argmin",
}
# Primitives that act on each particle separately, wherever the particle axis is.
_PARTICLEWISE = (
    _ELEMENTWISE
    | _IDENTITY
    | _CALLS
    | {
        "select_n",
        "polynomial",
        "broadcast_in_dim",
        "squeeze",
        "expand_dims",
        "transpose",
        "stack",
        "cond",
        "while",
        "random_wrap",
        "random_unwrap",
        "random_fold_in",
    }
)


def _untranslatable(cj: ClosedJaxpr) -> set[str]:
    """Return the primitives and uses in `cj` that the translator can't run."""
    bad: set[str] = set()
    _runtime(cj.jaxpr, [True] * len(cj.jaxpr.invars), bad)
    return bad


def _runtime(jaxpr: Any, marks: list[bool], bad: set[str]) -> list[bool]:
    """Return which outputs of `jaxpr` depend on runtime inputs, noting misuses."""
    env = dict(zip(jaxpr.invars, marks, strict=True))

    def rt(v: Any) -> bool:
        return not isinstance(v, Literal) and env.get(v, False)

    for eqn in jaxpr.eqns:
        name, p = eqn.primitive.name, eqn.params
        ins = [rt(v) for v in eqn.invars]
        if name not in _SUPPORTED:
            bad.add(name)
        for v in eqn.outvars:
            dt = getattr(v.aval, "dtype", None)
            if dt is not None and not _is_key(dt) and np.dtype(dt) not in _DSL:
                bad.add(f"{np.dtype(dt)} values")
        if (name == "dynamic_slice" and any(ins[1:])) or (
            name == "dynamic_update_slice" and any(ins[2:])
        ):
            bad.add(f"{name} with a runtime index")
        if name == "dot_general" and any(p["dimension_numbers"][1]):
            bad.add("batched dot_general")
        if name == "random_bits" and p["bit_width"] != 32:
            bad.add("random_bits with bit_width != 32")
        if name in _CALLS:
            outs = _runtime(_sub_jaxpr(eqn).jaxpr, ins, bad)
        elif name == "cond":
            branches = [_runtime(b.jaxpr, ins[1:], bad) for b in p["branches"]]
            outs = [ins[0] or any(o) for o in zip(*branches, strict=True)]
        elif name == "while":
            cn, bn = p["cond_nconsts"], p["body_nconsts"]
            body = p["body_jaxpr"].jaxpr
            outs = _loop(body, ins[cn : cn + bn], ins[cn + bn :], [], bad)
            if any(_runtime(p["cond_jaxpr"].jaxpr, ins[:cn] + outs, bad)):
                bad.add("while loop with a runtime trip count")
        elif name == "scan":
            nc, ncarry = p["num_consts"], p["num_carry"]
            consts, carry, xs = ins[:nc], ins[nc : nc + ncarry], ins[nc + ncarry :]
            outs = _loop(p["jaxpr"].jaxpr, consts, carry, xs, bad)
        else:
            outs = [any(ins)] * len(eqn.outvars)
        env.update(zip(eqn.outvars, outs, strict=True))
    return [rt(v) for v in jaxpr.outvars]


def _loop(
    body: Any, consts: list[bool], carry: list[bool], xs: list[bool], bad: set[str]
) -> list[bool]:
    """Return the runtime marks of a loop's carry and stacked outputs."""
    while True:
        outs = _runtime(body, consts + carry + xs, bad)
        new = [a or b for a, b in zip(carry, outs, strict=False)]
        if new == carry:
            return new + outs[len(carry) :]
        carry = new


def _particle_mixing(two: Any, probe: Any, J: int) -> set[str]:
    """Return what in a ``@vectorized`` rproc couples its particles.

    `two` and `probe` are jaxprs of one Euler step on 2 and on J particles; an axis
    of length J in `probe` is the particle axis. The kernel runs each particle on
    its own, as if J were 1, so a reduction, slice or concatenation over particles,
    a random draw without particles as its last axis, or a constant that depends on
    J would compute something other than XLA does.
    """
    bad: set[str] = set()
    _compare(two, probe, J, bad)
    return bad


def _particle_axis(shape: tuple[int, ...], J: int) -> int | None:
    """Return the index of the axis of length J in `shape`, or None."""
    axes = [i for i, n in enumerate(shape) if n == J]
    return axes[0] if axes else None


def _shape(v: Any) -> tuple[int, ...]:
    """Return the shape of a jaxpr variable or literal."""
    return tuple(getattr(v.aval, "shape", ()))


def _sub_jaxprs(eqn: Any) -> list[Any]:
    """Return the jaxprs that an equation calls, in a fixed order."""
    p = eqn.params
    if eqn.primitive.name in _CALLS:
        return [_sub_jaxpr(eqn).jaxpr]
    if eqn.primitive.name == "cond":
        return [b.jaxpr for b in p["branches"]]
    if eqn.primitive.name == "while":
        return [p["cond_jaxpr"].jaxpr, p["body_jaxpr"].jaxpr]
    if eqn.primitive.name == "scan":
        return [p["jaxpr"].jaxpr]
    return []


def _compare(two: Any, probe: Any, J: int, bad: set[str]) -> None:
    """Walk two traces of a step together, noting uses that couple particles."""
    if len(two.eqns) != len(probe.eqns):
        bad.add("operations that depend on the number of particles")
        return
    for e2, e in zip(two.eqns, probe.eqns, strict=True):
        name, p = e.primitive.name, e.params
        if e2.primitive.name != name:
            bad.add("operations that depend on the number of particles")
            return
        for v2, v in zip(e2.invars, e.invars, strict=True):
            if isinstance(v2, Literal) != isinstance(v, Literal) or (
                isinstance(v, Literal) and not _same_constant(v2.val, v.val)
            ):
                bad.add("constants that depend on the number of particles")
        shapes = [_shape(v) for v in [*e.invars, *e.outvars]]
        if any(s.count(J) > 1 for s in shapes):
            bad.add("several particle axes")
        ax = [_particle_axis(_shape(v), J) for v in e.invars]
        if name in ("random_bits", "random_split"):
            shape = tuple(p["shape"])
            k = _particle_axis(shape, J)
            if k is not None and k != len(shape) - 1:
                bad.add(f"{name} with particles on a non-trailing axis")
            if k is None and tuple(e2.params["shape"])[-1:] == (1,):
                bad.add(f"{name} with a trailing axis of length 1")
        elif name == "iota" and _particle_axis(_shape(e.outvars[0]), J) is not None:
            bad.add("iota over particles")
        elif name in _REDUCTIONS:
            if ax[0] is not None and ax[0] in p["axes"]:
                bad.add(f"{name} over particles")
        elif name == "dot_general":
            (lc, rc), _ = p["dimension_numbers"]
            if ax[0] in lc or ax[1] in rc:
                bad.add("dot_general over particles")
        elif name == "scan":
            if any(a == 0 for a in ax[p["num_consts"] + p["num_carry"] :]):
                bad.add("scan over particles")
        elif any(a is not None for a in ax) and name not in _PARTICLEWISE:
            if not _keeps_particles(e, ax, J):
                bad.add(f"{name} across particles")
        for s2, s in zip(_sub_jaxprs(e2), _sub_jaxprs(e), strict=True):
            _compare(s2, s, J, bad)


def _same_constant(a: Any, b: Any) -> bool:
    """Return whether two literals are equal, counting NaN as equal to NaN."""
    a, b = np.asarray(a), np.asarray(b)
    if np.issubdtype(a.dtype, np.inexact) and np.issubdtype(b.dtype, np.inexact):
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _keeps_particles(eqn: Any, ax: list[int | None], J: int) -> bool:
    """Return whether a shape operation leaves every particle's values in place."""
    name, p, k = eqn.primitive.name, eqn.params, ax[0]
    if name == "slice":
        strides = p["strides"] or (1,) * len(p["start_indices"])
        return k is None or (
            p["start_indices"][k] == 0
            and p["limit_indices"][k] == J
            and strides[k] == 1
        )
    if name == "dynamic_slice":
        return k is None or p["slice_sizes"][k] == J
    if name == "dynamic_update_slice":
        return k is None or _shape(eqn.invars[1])[k] == J
    if name == "pad":
        return k is None or tuple(p["padding_config"][k]) == (0, 0, 0)
    if name in ("concatenate", "split", "unstack"):
        dim = p["dimension"] if name == "concatenate" else p["axis"]
        return dim not in ax
    if name == "reshape":
        out = _particle_axis(_shape(eqn.outvars[0]), J)
        if k is None or out is None:
            return k is None and out is None
        return math.prod(_shape(eqn.invars[0])[:k]) == math.prod(
            _shape(eqn.outvars[0])[:out]
        )
    return False
