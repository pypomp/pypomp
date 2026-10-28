"""Translate a JAX function of one particle into CuTe DSL scalar code.

CuTe DSL is the Python interface to NVIDIA CUTLASS's CuTe library, used here to
write CUDA kernels. ``_interpret(closed_jaxpr, args, ctx)`` walks a jaxpr inside a
``@cute.kernel`` and returns, for each output, a numpy object array of DSL scalars
(or plain constants). Each jaxpr value keeps its small per-particle shape as an
object array, so slices, reshapes and reductions are bookkeeping at trace time, and
the emitted code is straight-line scalar math in registers, apart from branches
for runtime conds and calls of large sub-jaxprs that run more than once.

Constant tables that the function indexes at runtime, such as a table of weekly
data read at the current week, can't be registers. :func:`_runtime_tables` finds
them, the kernel takes them in one buffer, and a runtime-indexed slice of one is a
:class:`_View` whose elements are loaded where they are used.

:func:`_hoist` splits off the part of a jaxpr that depends only on inputs every
particle of a filter shares, for XLA to compute once per filter.

Random numbers follow ``ctx.rng``. With "threefry" they are JAX's (partitionable)
threefry: element f of a flattened draw is the XOR of the two Threefry2x32-20 words
for counter (0, f), and split and fold_in keys are both words. With "philox" they
are Philox4x32-10 with all four output words used, in the layout of
:mod:`pypomp.random.philox`: element (l, m) of a draw of shape (..., L) is word
l % 4 of the call with counter (l // 4) * L + m. An rproc written over all J
particles at once (``@vectorized``) is traced with J = 1, and ``ctx.vectorized_J``
and ``ctx.particle`` give the particle axis of a draw the counters that the full
(..., J) draw would use.

Unsupported primitives raise NotImplementedError naming the primitive.
"""

from __future__ import annotations

import hashlib
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
# Threefry2x32-20 rotations and key-schedule parity (Salmon et al., 2011).
_TF_ROT = ((13, 15, 26, 6), (17, 29, 16, 24))
_TF_PARITY = np.uint32(0x1BD11BDA)


@dataclass
class _Ctx:
    """Where a translated function runs: its particle and the rproc's layout."""

    vectorized_J: int | None
    """J for a ``@vectorized`` rproc traced with J = 1, otherwise None."""
    particle: _Scalar
    """This thread's particle index within its filter, or None."""
    rng: str
    """The generator draws follow: "threefry" or "philox"."""
    counter_offset: _Scalar = None
    """Added to every threefry counter: the flattened index of this thread's first
    element in a draw over all particles, so its draws are that draw's; or None."""
    outlined: dict[Any, Any] = field(default_factory=dict)
    """The device functions emitted so far, by sub-jaxpr and constant inputs."""
    tables: _Tables | None = None
    """The kernel's table buffer, or None where there is none."""


@dataclass(frozen=True)
class _Tables:
    """The constant tables a kernel reads at runtime indices, packed in one buffer."""

    buf: Any
    """A 1-D uint32 ``cute.Tensor`` holding every table's bits."""
    offsets: dict[tuple, int]
    """Each table's first element in `buf`, by :func:`_table_key`."""


@dataclass(frozen=True)
class _View:
    """Part of a table in the kernel's buffer: element ``idx`` is at buffer offset
    ``base + sum(idx * strides)``, and ``base`` may be a runtime value."""

    base: _Scalar
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: np.dtype


def _table_key(a: np.ndarray) -> tuple:
    """Return the key of constant `a` in :attr:`_Tables.offsets`."""
    a = np.ascontiguousarray(a)
    return (a.dtype.str, a.shape, hashlib.blake2b(a.tobytes(), digest_size=16).digest())


def _pack_tables(tables: list[np.ndarray]) -> tuple[np.ndarray, dict[tuple, int]]:
    """Return the uint32 buffer holding `tables` and each one's offset in it."""
    parts, offsets, n = [], {}, 0
    for a in tables:
        if np.dtype(a.dtype).itemsize != 4:
            raise NotImplementedError(f"runtime-indexed {a.dtype} table")
        offsets[_table_key(a)] = n
        parts.append(np.ascontiguousarray(a).reshape(-1).view(np.uint32))
        n += a.size
    # A kernel argument can't be empty.
    return np.concatenate(parts or [np.zeros(1, np.uint32)]), offsets


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


def _rotl(x: _Scalar, r: int) -> _Scalar:
    """Return the uint32 `x` rotated left by `r` bits."""
    if not _is_dsl(x):
        v = int(x)
        return np.uint32(((v << r) | (v >> (32 - r))) & 0xFFFFFFFF)
    return (x << cutlass.Uint32(r)) | (x >> cutlass.Uint32(32 - r))


def _threefry2x32(
    k0: _Scalar, k1: _Scalar, x0: _Scalar, x1: _Scalar
) -> tuple[_Scalar, _Scalar]:
    """Return the two Threefry2x32-20 words for key (k0, k1) and counter (x0, x1)."""
    k0, k1, x0, x1 = (_u32(v) for v in (k0, k1, x0, x1))
    ks = (k0, k1, _xor(_xor(k0, k1), _TF_PARITY))
    x0, x1 = _add(x0, ks[0]), _add(x1, ks[1])
    for i in range(5):
        for r in _TF_ROT[i % 2]:
            x0 = _add(x0, x1)
            x1 = _xor(x0, _rotl(x1, r))
        x0 = _add(x0, ks[(i + 1) % 3])
        x1 = _add(x1, _add(ks[(i + 2) % 3], np.uint32(i + 1)))
    return x0, x1


def _key_derive(
    rng: str, k0: _Scalar, k1: _Scalar, cnt: _Scalar
) -> tuple[_Scalar, _Scalar]:
    """Return the key that split and fold_in derive from key (k0, k1) and `cnt`."""
    if rng == "threefry":
        return _threefry2x32(k0, k1, 0, cnt)
    y = _philox4x32(k0, k1, 0, 0, 0, cnt)
    return y[0], y[1]


def _draw_threefry(
    ctx: _Ctx, k0: _Scalar, k1: _Scalar, shape: tuple[int, ...]
) -> np.ndarray:
    """Return the 32-bit words of a draw of `shape`, as partitionable threefry
    lays them out: element f of the flattened draw hashes counter (0, f)."""
    out = np.empty(shape, dtype=object)
    flat = out.reshape(-1)
    J = ctx.vectorized_J
    for f in range(flat.size):
        if J and shape[-1] == 1:  # the particle axis, of length J at run time
            cnt = _as_dsl(np.uint32(f * J), np.uint32) + ctx.particle
        elif ctx.counter_offset is not None:
            cnt = _add(ctx.counter_offset, np.uint32(f))
        else:
            cnt = np.uint32(f)
        y0, y1 = _threefry2x32(k0, k1, 0, cnt)
        flat[f] = _xor(y0, y1)
    return out


def _draw(ctx: _Ctx, k0: _Scalar, k1: _Scalar, shape: tuple[int, ...]) -> np.ndarray:
    """Return the 32-bit words of a draw of `shape` from key (k0, k1)."""
    if ctx.rng == "threefry":
        return _draw_threefry(ctx, k0, k1, shape)
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
        "gather",
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
# Primitives that take a view of a table as it is; the rest get its elements.
_VIEW_OPS = (
    _CALLS
    | _IDENTITY
    | {"dynamic_slice", "gather", "slice", "squeeze", "reshape", "expand_dims"}
)


def _fold_device() -> Any:
    """Return a CPU device, or the default device if JAX runs without a CPU backend."""
    try:
        return jax.devices("cpu")[0]
    except RuntimeError:
        return jax.devices()[0]


def _fold(eqn: Any, vals: list[np.ndarray]) -> list[np.ndarray]:
    """Evaluate equation `eqn` on constant inputs, eagerly on the CPU if JAX has one."""
    # This runs while JAX traces the CuTe call, so evaluation must be forced eager.
    with jax.ensure_compile_time_eval(), jax.default_device(_fold_device()):
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
        and not isinstance(v, _View)
        for v in vals
    )


def _as_np_const(v: Any, dtype: Any) -> np.ndarray:
    """Return the constant `v` as a numpy array of `dtype`."""
    v = np.asarray(v)
    if v.dtype == object:
        v = v.astype(dtype)
    return v


def _index(x: Any) -> _Scalar:
    """Return the scalar index `x`, which may be a 0-d array, as an int, or as an
    Int32 if it is a runtime value."""
    if isinstance(x, np.ndarray):
        x = _obj(x).reshape(())[()]
    return x.to(cutlass.Int32) if _is_dsl(x) else int(x)


def _clamp(s: Any, hi: int) -> _Scalar:
    """Return the index `s` clamped to [0, hi], as XLA clamps slice starts."""
    s = _index(s)
    if not _is_dsl(s):
        return min(max(s, 0), hi)
    lo = _select(s < cutlass.Int32(0), cutlass.Int32(0), s, np.int32)
    return _select(lo > cutlass.Int32(hi), cutlass.Int32(hi), lo, np.int32)


def _table_view(a: Any, dtype: Any, ctx: _Ctx) -> _View:
    """Return the whole of constant table `a` as a view of the kernel's buffer."""
    a = _as_np_const(a, dtype)
    off = None if ctx.tables is None else ctx.tables.offsets.get(_table_key(a))
    if off is None:
        raise NotImplementedError("a runtime-indexed table the kernel wasn't given")
    strides = tuple(int(s) // a.itemsize for s in np.ascontiguousarray(a).strides)
    return _View(off, a.shape, strides, np.dtype(a.dtype))


def _offset(v: _View, idx: tuple[_Scalar, ...]) -> _Scalar:
    """Return the buffer offset of the element of `v` at `idx`."""
    off = v.base
    for i, s in zip(idx, v.strides, strict=True):
        if s and (_is_dsl(i) or int(i)):
            off = off + _index(i) * s
    return off


def _load(ctx: _Ctx, off: _Scalar, dtype: np.dtype) -> _Scalar:
    """Return the table element at buffer offset `off`."""
    assert ctx.tables is not None
    bits = ctx.tables.buf[_as_dsl(off, np.int32)]
    return bits if dtype == np.uint32 else bits.bitcast(_DSL[dtype])


def _materialize(v: Any, ctx: _Ctx) -> Any:
    """Return `v` with the elements of a view loaded, so as an object array."""
    if not isinstance(v, _View):
        return v
    out = np.empty(v.shape, dtype=object)
    for idx in np.ndindex(*v.shape) if v.shape else [()]:
        out[idx] = _load(ctx, _offset(v, idx), v.dtype)
    return out


def _view_slice(
    v: _View, starts: list[_Scalar], sizes: tuple[int, ...], steps: tuple[int, ...]
) -> _View:
    """Return the part of `v` from `starts`, `sizes` elements long in steps."""
    return _View(
        _offset(v, tuple(starts)),
        tuple(sizes),
        tuple(s * k for s, k in zip(v.strides, steps, strict=True)),
        v.dtype,
    )


def _view_reshape(v: _View, shape: tuple[int, ...]) -> _View | None:
    """Return `v` reshaped to `shape` if that only adds or drops axes of length 1."""
    kept = [(n, s) for n, s in zip(v.shape, v.strides, strict=True) if n != 1]
    if [n for n, _ in kept] != [n for n in shape if n != 1]:
        return None
    it = iter(s for _, s in kept)
    strides = tuple(0 if n == 1 else next(it) for n in shape)
    return _View(v.base, tuple(shape), strides, v.dtype)


# Most candidates a runtime index may pick among in registers, per element read.
_MAX_PICK = 64


def _pick(
    x: np.ndarray, pos: list[_Scalar], ranges: list[tuple[Any, Any]], dtype: Any
) -> _Scalar:
    """Return ``x[pos]`` for object array `x`, where ``pos[d]`` may be a runtime
    index known to lie in ``ranges[d]``, by a tree of selects over the candidates."""
    d = next((d for d, p in enumerate(pos) if _is_dsl(p)), None)
    if d is None:
        return x[tuple(int(p) for p in pos)]

    def tree(lo: int, hi: int) -> _Scalar:
        if lo == hi:
            return _pick(x, [*pos[:d], lo, *pos[d + 1 :]], ranges, dtype)
        mid = (lo + hi + 1) // 2
        lt = pos[d] < cutlass.Int32(mid)
        return _select(lt, tree(lo, mid - 1), tree(mid, hi), dtype)

    return tree(*ranges[d])


def _pick_count(eqn: Any, runtime: list[bool]) -> int:
    """Return how many candidates each element of dynamic_slice or gather `eqn`
    selects among when its operand is in registers; ``runtime[d]`` says whether
    dynamic_slice's start index d is a runtime value."""
    shape, sizes = _shape(eqn.invars[0]), eqn.params["slice_sizes"]
    if eqn.primitive.name == "gather":
        dims = eqn.params["dimension_numbers"].start_index_map
    else:
        dims = [d for d, r in enumerate(runtime) if r]
    return math.prod(shape[d] - sizes[d] + 1 for d in dims)


def _gather(eqn: Any, x: Any, dtype: Any, idx: Any, ctx: _Ctx) -> np.ndarray:
    """Return XLA's gather of operand `x` at start indices `idx`.

    `x` is a view, a constant table or an object array. At a runtime start index,
    a view's or table's elements are loaded from the table buffer, and an object
    array's are selected among in registers.
    """
    p = eqn.params
    dn = p["dimension_numbers"]
    if dn.operand_batching_dims:
        raise NotImplementedError("gather with batching dimensions")
    idx = _obj(idx)
    shape = eqn.outvars[0].aval.shape
    offset_dims, collapsed = tuple(dn.offset_dims), set(dn.collapsed_slice_dims)
    sizes = p["slice_sizes"]
    ndim = len(sizes)
    offset_src = [d for d in range(ndim) if d not in collapsed]
    runtime = any(_is_dsl(e) for e in idx.flat)
    in_regs = runtime and not isinstance(x, _View) and not _all_const([x])
    if in_regs and _pick_count(eqn, []) > _MAX_PICK:
        raise NotImplementedError("gather of many runtime values at a runtime index")
    if runtime and not in_regs and not isinstance(x, _View):
        x = _table_view(x, dtype, ctx)
    # Out-of-bounds starts are clamped, except that "fill" mode returns its fill
    # value for the whole slice.
    fill = "FILL" in str(p["mode"])
    fv = _fill_value(p.get("fill_value"), dtype) if fill else None
    out = np.empty(shape, dtype=object)
    batch_dims = [d for d in range(len(shape)) if d not in offset_dims]
    for o in np.ndindex(*shape) if shape else [()]:
        b = tuple(o[d] for d in batch_dims)
        pos: list[_Scalar] = [0] * ndim
        ranges: list[tuple[Any, Any]] = [(0, 0)] * ndim
        inside: list[_Scalar] = []
        for k, d in enumerate(dn.start_index_map):
            s, hi = idx[b + (k,)], x.shape[d] - sizes[d]
            if fill:
                si = _index(s)
                inside.append(
                    (si >= cutlass.Int32(0)) & (si <= cutlass.Int32(hi))
                    if _is_dsl(si)
                    else 0 <= si <= hi
                )
            pos[d], ranges[d] = _clamp(s, hi), (0, hi)
        for od, d in zip(offset_dims, offset_src, strict=True):
            pos[d] = pos[d] + o[od]
            ranges[d] = (ranges[d][0] + o[od], ranges[d][1] + o[od])
        if isinstance(x, _View):
            val = _load(ctx, _offset(x, tuple(pos)), x.dtype)
        elif in_regs:
            val = _pick(_obj(x), pos, ranges, dtype)
        else:
            val = _obj(x)[tuple(int(q) for q in pos)]
        for ok in inside:
            val = _select(ok, val, fv, dtype)
        out[o] = val
    return out


def _fill_value(fv: Any, dtype: Any) -> Any:
    """Return gather's fill value for `dtype`, given its ``fill_value`` parameter."""
    dt = np.dtype(dtype)
    if fv is None:
        if np.issubdtype(dt, np.inexact):
            fv = np.nan
        elif np.issubdtype(dt, np.signedinteger):
            fv = np.iinfo(dt).min
        elif np.issubdtype(dt, np.integer):
            fv = np.iinfo(dt).max
        else:
            fv = True
    return np.asarray(fv, dt)[()]


def _interpret(
    cj: ClosedJaxpr, args: list[Any], ctx: _Ctx, keep_views: bool = False
) -> list[Any]:
    """Evaluate `cj` on object arrays, emitting DSL code, and return its outputs.

    Outputs that are views of tables stay views if `keep_views`, as for a call
    whose caller goes on to index them; otherwise their elements are loaded.
    """
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
        if name not in _VIEW_OPS:
            ins = [_materialize(v, ctx) for v in ins]
        in_dt = [getattr(v.aval, "dtype", None) for v in eqn.invars]
        out_avals = [v.aval for v in eqn.outvars]
        outs = _eval_eqn(eqn, name, ins, in_dt, out_avals, ctx)
        for v, o in zip(eqn.outvars, outs, strict=True):
            env[v] = o
    outs = [read(v) for v in jaxpr.outvars]
    return outs if keep_views else [_materialize(v, ctx) for v in outs]


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
    if isinstance(ins[0], _View) and name in ("reshape", "squeeze", "expand_dims"):
        v = None
        if name != "reshape" or eqn.params.get("dimensions") is None:
            v = _view_reshape(ins[0], out_avals[0].shape)
        if v is not None:
            return [v]
        ins = [_materialize(ins[0], ctx), *ins[1:]]
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
        if isinstance(ins[0], _View):
            return [_view_slice(ins[0], list(st), out_avals[0].shape, strd)]
        return [
            _obj(ins[0])[
                tuple(slice(a, b, c) for a, b, c in zip(st, lim, strd, strict=True))
            ]
        ]
    if name == "dynamic_slice":
        starts = ins[1:]
        sizes = eqn.params["slice_sizes"]
        if (
            not isinstance(ins[0], _View)
            and not _all_const(starts)
            and not _all_const([ins[0]])
        ):
            # Runtime values at a runtime start: select among them in registers.
            if _pick_count(eqn, [not _all_const([s]) for s in starts]) > _MAX_PICK:
                raise NotImplementedError(
                    "dynamic_slice of many runtime values at a runtime index"
                )
            x = _obj(ins[0])
            hi = [n - k for n, k in zip(x.shape, sizes, strict=True)]
            st = [_clamp(s, h) for s, h in zip(starts, hi, strict=True)]
            out = np.empty(sizes, dtype=object)
            for o in np.ndindex(*sizes) if sizes else [()]:
                pos = [s + i for s, i in zip(st, o, strict=True)]
                ranges = [(i, h + i) for h, i in zip(hi, o, strict=True)]
                out[o] = _pick(x, pos, ranges, in_dt[0])
            return [out]
        if isinstance(ins[0], _View) or not _all_const(starts):
            # A runtime start of a constant operand reads it from the table buffer.
            x = ins[0]
            if not isinstance(x, _View):
                x = _table_view(x, in_dt[0], ctx)
            st = [
                _clamp(s, n - k) for s, n, k in zip(starts, x.shape, sizes, strict=True)
            ]
            return [_view_slice(x, st, sizes, (1,) * len(sizes))]
        x = _obj(ins[0])
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

    if name == "gather":
        return [_gather(eqn, ins[0], in_dt[0], ins[1], ctx)]

    if name in _CALLS:
        sub = _sub_jaxpr(eqn)
        if not any(isinstance(v, _View) for v in ins) and _outlinable(sub, ins):
            return _call_outlined(sub, ins, out_avals, ctx)
        return _interpret(sub, ins, ctx, keep_views=True)
    if name == "cond":
        index, ops = _materialize(ins[0], ctx), ins[1:]
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
                ctx.rng, kb[idx + (0,)], kb[idx + (1,)], _u32(mb[idx])
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
                    ctx.rng, keys[kidx + (0,)], keys[kidx + (1,)], cnt
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
            else:  # a scalar draw is shared, not one per particle: element 0
                out[kidx] = _draw(
                    _Ctx(vectorized_J=None, particle=None, rng=ctx.rng),
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

    Random draws read the thread's particle index from the caller, and table reads
    the kernel's table buffer, so jaxprs that do either stay inline, as do keys,
    which aren't plain scalars.
    """
    if _all_const(ins):
        return False
    avals = [v.aval for v in [*cj.jaxpr.invars, *cj.jaxpr.outvars]]
    if any(_is_key(a.dtype) or np.dtype(a.dtype) not in _LLVM_TYPE for a in avals):
        return False
    return (
        _size(cj) >= _OUTLINE_MIN_EQNS
        and not _uses_random(cj)
        and not _runtime_tables([cj])
    )


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
            body_ctx = _Ctx(
                vectorized_J=None, particle=None, rng=ctx.rng, outlined=ctx.outlined
            )
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


def _runtime_tables(cjs: list[ClosedJaxpr]) -> list[np.ndarray]:
    """Return the constants that `cjs` slice or gather at runtime indices, which
    the kernel must hold in memory.

    Constants are folded as the translator folds them, so each table found is the
    array the translator meets. Loop carries count as runtime values, which may
    find a table that the translator then indexes at constant indices.
    """
    found: dict[tuple, np.ndarray] = {}
    for cj in cjs:
        _find_tables(cj, [None] * len(cj.jaxpr.invars), found)
    return list(found.values())


def _find_tables(
    cj: ClosedJaxpr, args: list[np.ndarray | None], found: dict[tuple, np.ndarray]
) -> list[np.ndarray | None]:
    """Return the outputs of `cj` that are constants (None for runtime values),
    adding the tables it indexes at runtime to `found`."""
    jaxpr = cj.jaxpr
    env: dict[Any, np.ndarray | None] = {
        v: np.asarray(c) for v, c in zip(jaxpr.constvars, cj.consts, strict=True)
    }
    env.update(zip(jaxpr.invars, args, strict=True))

    def read(v: Any) -> np.ndarray | None:
        return np.asarray(v.val) if isinstance(v, Literal) else env.get(v)

    for eqn in jaxpr.eqns:
        name, p = eqn.primitive.name, eqn.params
        ins = [read(v) for v in eqn.invars]
        n_out = len(eqn.outvars)
        consts = [x for x in ins if x is not None]
        known = len(consts) == len(ins)
        if name in ("dynamic_slice", "gather") and ins[0] is not None and not known:
            found.setdefault(_table_key(ins[0]), ins[0])
        if name in _CALLS:
            outs = _find_tables(_sub_jaxpr(eqn), ins, found)
        elif name == "cond" and ins[0] is not None:
            outs = _find_tables(p["branches"][int(ins[0])], ins[1:], found)
        elif name == "cond":
            for b in p["branches"]:
                _find_tables(b, ins[1:], found)
            outs = [None] * n_out
        elif name == "while":
            cn, bn = p["cond_nconsts"], p["body_nconsts"]
            carry = [None] * (len(ins) - cn - bn)
            _find_tables(p["cond_jaxpr"], ins[:cn] + carry, found)
            _find_tables(p["body_jaxpr"], ins[cn : cn + bn] + carry, found)
            outs = [None] * n_out
        elif name == "scan":
            nc = p["num_consts"]
            _find_tables(p["jaxpr"], ins[:nc] + [None] * (len(ins) - nc), found)
            outs = [None] * n_out
        elif (
            known
            and name not in ("empty", "empty2")
            and not any(_is_key(getattr(v.aval, "dtype", None)) for v in eqn.invars)
            and not any(_is_key(getattr(v.aval, "dtype", None)) for v in eqn.outvars)
        ):
            outs = _fold(eqn, consts)
        else:
            outs = [None] * n_out
        env.update(zip(eqn.outvars, outs, strict=True))
    return [read(v) for v in jaxpr.outvars]


def _untranslatable(cj: ClosedJaxpr) -> set[str]:
    """Return the primitives and uses in `cj` that the translator can't run."""
    bad: set[str] = set()
    _runtime(cj.jaxpr, [True] * len(cj.jaxpr.invars), bad)
    return bad


# The mark of a runtime value that is a table, or part of one, still in memory.
_MEM = "in memory"


def _runtime(jaxpr: Any, marks: list[Any], bad: set[str]) -> list[Any]:
    """Return which outputs of `jaxpr` depend on runtime inputs, noting misuses.

    A mark is False for a constant, True for a runtime value in registers and
    :data:`_MEM` for a table or part of one, which a runtime index may read.
    """
    env = dict(zip(jaxpr.invars, marks, strict=True))

    def rt(v: Any) -> Any:
        return not isinstance(v, Literal) and env.get(v, False)

    def reg(ms: list[Any]) -> list[bool]:
        return [bool(m) for m in ms]

    for eqn in jaxpr.eqns:
        name, p = eqn.primitive.name, eqn.params
        ins = [rt(v) for v in eqn.invars]
        if name not in _SUPPORTED:
            bad.add(name)
        for v in eqn.outvars:
            dt = getattr(v.aval, "dtype", None)
            if dt is not None and not _is_key(dt) and np.dtype(dt) not in _DSL:
                bad.add(f"{np.dtype(dt)} values")
        if (
            name in ("dynamic_slice", "gather")
            and any(ins[1:])
            and ins[0] is True
            and _pick_count(eqn, reg(ins[1:])) > _MAX_PICK
        ):
            bad.add(f"{name} of many runtime values at a runtime index")
        if name == "dynamic_update_slice" and any(ins[2:]):
            bad.add(f"{name} of runtime values at a runtime index")
        if name == "gather" and p["dimension_numbers"].operand_batching_dims:
            bad.add("gather with batching dimensions")
        if name == "dot_general" and any(p["dimension_numbers"][1]):
            bad.add("batched dot_general")
        if name == "random_bits" and p["bit_width"] != 32:
            bad.add("random_bits with bit_width != 32")
        if name in _CALLS:
            outs = _runtime(_sub_jaxpr(eqn).jaxpr, ins, bad)
        elif name == "cond":
            branches = [_runtime(b.jaxpr, reg(ins[1:]), bad) for b in p["branches"]]
            outs = [bool(ins[0]) or any(o) for o in zip(*branches, strict=True)]
        elif name == "while":
            cn, bn = p["cond_nconsts"], p["body_nconsts"]
            body = p["body_jaxpr"].jaxpr
            outs = _loop(body, reg(ins[cn : cn + bn]), reg(ins[cn + bn :]), [], bad)
            if any(_runtime(p["cond_jaxpr"].jaxpr, reg(ins[:cn]) + outs, bad)):
                bad.add("while loop with a runtime trip count")
        elif name == "scan":
            nc, ncarry = p["num_consts"], p["num_carry"]
            ins = reg(ins)
            consts, carry, xs = ins[:nc], ins[nc : nc + ncarry], ins[nc + ncarry :]
            outs = _loop(p["jaxpr"].jaxpr, consts, carry, xs, bad)
        elif name == "dynamic_slice" and (ins[0] is _MEM or (not ins[0] and any(ins))):
            outs = [_MEM]
        elif name in ("slice", "squeeze", "expand_dims") and ins[0] is _MEM:
            outs = [_MEM]
        elif name == "reshape" and ins[0] is _MEM:
            outs = [_MEM if _keeps_view(eqn) else True]
        elif name in _IDENTITY:
            outs = ins[: len(eqn.outvars)]
        else:
            outs = [any(ins)] * len(eqn.outvars)
        env.update(zip(eqn.outvars, outs, strict=True))
    return [rt(v) for v in jaxpr.outvars]


def _keeps_view(eqn: Any) -> bool:
    """Return whether the translator keeps a view through reshape `eqn`, as
    :func:`_view_reshape` does, rather than loading its elements."""
    if eqn.params.get("dimensions") is not None:
        return False
    before, after = _shape(eqn.invars[0]), _shape(eqn.outvars[0])
    return [n for n in before if n != 1] == [n for n in after if n != 1]


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


# Most elements one hoisted value may have, and all of a step's together. A larger
# value, such as a table's row that a particle indexes, is computed in the kernel
# instead, from smaller hoisted inputs.
_MAX_HOISTED = 32
_MAX_HOISTED_TOTAL = 256
_HOISTABLE = {
    np.dtype("float32"),
    np.dtype("int32"),
    np.dtype("uint32"),
    np.dtype("bool"),
}


@dataclass(frozen=True)
class _Hoisted:
    """A jaxpr split into what every particle of a filter shares and the rest.

    ``shared`` maps the inputs marked shared to the hoisted values, which XLA
    evaluates once per filter (and step); ``rest`` maps all the original inputs,
    then the hoisted values, to the original outputs."""

    shared: ClosedJaxpr
    rest: ClosedJaxpr
    avals: tuple[tuple[tuple[int, ...], np.dtype], ...]
    """The hoisted values' shapes and dtypes."""


def _struct(v: Any) -> jax.ShapeDtypeStruct:
    """Return the shape and dtype of jaxpr variable `v`."""
    return jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype)


def _eval_jaxpr(
    jaxpr: Any,
    consts: list[Any],
    args: list[Any],
    eqns: list[Any],
    known: dict[Any, Any] | None = None,
) -> dict:
    """Return the values of `jaxpr`'s variables after binding `eqns`, some of its
    equations, given values `known` of some variables; `args` may hold None for
    inputs that none of `eqns` reads."""
    env: dict[Any, Any] = dict(zip(jaxpr.constvars, consts, strict=True))
    env.update(zip(jaxpr.invars, args, strict=True))
    env.update(known or {})
    for eqn in eqns:
        ins = [v.val if isinstance(v, Literal) else env[v] for v in eqn.invars]
        out = eqn.primitive.bind(*ins, **eqn.primitive.get_bind_params(eqn.params))
        outs = out if eqn.primitive.multiple_results else [out]
        env.update(zip(eqn.outvars, outs, strict=True))
    return env


def _inline_mixed(cj: ClosedJaxpr, shared: list[bool]) -> ClosedJaxpr:
    """Return `cj` with each call that takes both shared and per-particle values
    inlined, so its shared part can be hoisted, unless it is large enough that the
    translator would outline it."""

    def run(jaxpr: Any, consts: list[Any], args: list[Any], marks: list[int]):
        env = dict(zip(jaxpr.constvars, consts, strict=True))
        env.update(zip(jaxpr.invars, args, strict=True))
        mk = {v: 0 for v in jaxpr.constvars}
        mk.update(zip(jaxpr.invars, marks, strict=True))
        for eqn in jaxpr.eqns:
            ins = [v.val if isinstance(v, Literal) else env[v] for v in eqn.invars]
            ms = [0 if isinstance(v, Literal) else mk[v] for v in eqn.invars]
            if (
                eqn.primitive.name in _CALLS
                and {1, 2} <= set(ms)
                and _size(_sub_jaxpr(eqn)) < _OUTLINE_MIN_EQNS
            ):
                sub = _sub_jaxpr(eqn)
                outs, oms = run(sub.jaxpr, list(sub.consts), ins, ms)
            else:
                p = eqn.primitive.get_bind_params(eqn.params)
                out = eqn.primitive.bind(*ins, **p)
                outs = out if eqn.primitive.multiple_results else [out]
                oms = [max(ms, default=0)] * len(outs)
            env.update(zip(eqn.outvars, outs, strict=True))
            mk.update(zip(eqn.outvars, oms, strict=True))
        read = lambda v: v.val if isinstance(v, Literal) else env[v]  # noqa: E731
        mread = lambda v: 0 if isinstance(v, Literal) else mk[v]  # noqa: E731
        return [read(v) for v in jaxpr.outvars], [mread(v) for v in jaxpr.outvars]

    marks = [1 if s else 2 for s in shared]
    avals = [_struct(v) for v in cj.jaxpr.invars]
    return jax.make_jaxpr(lambda *a: run(cj.jaxpr, list(cj.consts), list(a), marks)[0])(
        *avals
    )


def _hoist(cj: ClosedJaxpr, shared: list[bool]) -> _Hoisted | None:
    """Split `cj` so that what depends only on the inputs marked `shared` (and on
    constants) runs once per filter, or return None if nothing is worth hoisting.

    XLA's vmap over particles computes such values once; a kernel thread per
    particle would otherwise compute them, table reads included, in every thread.
    """
    flat = _inline_mixed(cj, shared)
    jaxpr, eqns = flat.jaxpr, flat.jaxpr.eqns
    # 0: a constant, 1: shared by the filter's particles, 2: per particle.
    mark: dict[Any, int] = {v: 0 for v in jaxpr.constvars}
    mark.update({v: 1 if s else 2 for v, s in zip(jaxpr.invars, shared, strict=True)})
    eq_mark: list[int] = []
    defined: dict[Any, int] = {}
    for i, eqn in enumerate(eqns):
        vs = [v for v in eqn.invars if not isinstance(v, Literal)]
        eq_mark.append(max((mark[v] for v in vs), default=0))
        for v in eqn.outvars:
            mark[v], defined[v] = eq_mark[i], i
    shared_vars = lambda vs: [  # noqa: E731
        v
        for v in vs
        if not isinstance(v, Literal) and mark[v] == 1 and v not in jaxpr.invars
    ]
    # The kernel's equations: per-particle ones, constants, which the translator
    # folds, and shared ones whose output is too large to hoist.
    in_kernel = {i for i, m in enumerate(eq_mark) if m != 1}
    while True:
        res = {
            v: None
            for i in sorted(in_kernel)
            for v in shared_vars(eqns[i].invars)
            if defined[v] not in in_kernel
        } | dict.fromkeys(shared_vars(jaxpr.outvars))
        big = [
            defined[v]
            for v in res
            if math.prod(v.aval.shape) > _MAX_HOISTED
            or np.dtype(v.aval.dtype) not in _HOISTABLE
        ]
        if not big:
            break
        in_kernel.update(big)
    res = list(res)
    if (
        not res
        or len(in_kernel) == len(eqns)
        or sum(math.prod(v.aval.shape) for v in res) > _MAX_HOISTED_TOTAL
    ):
        return None
    consts = list(flat.consts)

    def shared_fn(*a):
        it = iter(a)
        args = [next(it) if s else None for s in shared]
        env = _eval_jaxpr(
            jaxpr,
            consts,
            args,
            [e for e, m in zip(eqns, eq_mark, strict=True) if m <= 1],
        )
        return [env[v] for v in res]

    def rest_fn(*a):
        n = len(jaxpr.invars)
        kernel_eqns = [eqns[i] for i in sorted(in_kernel)]
        env = _eval_jaxpr(
            jaxpr, consts, list(a[:n]), kernel_eqns, dict(zip(res, a[n:], strict=True))
        )
        return [v.val if isinstance(v, Literal) else env[v] for v in jaxpr.outvars]

    in_avals = [_struct(v) for v in jaxpr.invars]
    return _Hoisted(
        shared=jax.make_jaxpr(shared_fn)(
            *[a for a, s in zip(in_avals, shared, strict=True) if s]
        ),
        rest=jax.make_jaxpr(rest_fn)(*in_avals, *[_struct(v) for v in res]),
        avals=tuple((tuple(v.aval.shape), np.dtype(v.aval.dtype)) for v in res),
    )
