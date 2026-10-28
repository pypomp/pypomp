"""What keeps a model off the CuTe kernel, found from its jaxprs alone.

Nothing here needs a GPU or ``nvidia-cutlass-dsl``, so :meth:`pypomp.Pomp.backend_report`
can say what the translator would reject, and where in the model's code, on any
machine. The translator (``translate.py``) runs what these checks accept.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import ClosedJaxpr  # pyright: ignore[reportPrivateImportUsage]
from jax.extend.core import Literal

from .report import Problem

if TYPE_CHECKING:
    from ..core.pomp import Pomp

# The dtypes a kernel holds in registers.
_DTYPES = (np.dtype("float32"), np.dtype("int32"), np.dtype("uint32"), np.dtype("bool"))
# Particles in the trace that finds a @vectorized rproc's particle axis: a prime that
# no other axis length is likely to equal.
_PROBE_J = 1031
# _trace_step's output: step, transform, whether vectorized, and their VJPs.
_Traced = tuple[ClosedJaxpr, ClosedJaxpr, bool, ClosedJaxpr, ClosedJaxpr]
# What the translator can't run, each with the equations that use it.
_Bad = dict[str, list[Any]]

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


def _sub_jaxpr(eqn: Any) -> ClosedJaxpr:
    """Return the jaxpr that a call primitive's equation `eqn` calls."""
    for k in ("jaxpr", "call_jaxpr", "fun_jaxpr"):
        if k in eqn.params:
            j = eqn.params[k]
            return j if hasattr(j, "consts") else ClosedJaxpr(j, ())
    raise NotImplementedError(f"{eqn.primitive.name}: no sub-jaxpr")


def _is_key(dtype: Any) -> bool:
    """Return whether `dtype` is a JAX PRNG key dtype."""
    return dtype is not None and jnp.issubdtype(dtype, jax.dtypes.prng_key)


def _note(bad: _Bad, what: str, sources: tuple[Any, ...] = ()) -> None:
    """Record `what` in `bad`, used by the equation whose source info and its
    callers' are `sources`, innermost first, if given."""
    seen = bad.setdefault(what, [])
    if sources:
        seen.append(sources)


# pypomp's own machinery, which calls the model's code; other pypomp files, such as
# pypomp.random's samplers, are code the model calls.
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
_MACHINERY = tuple(_PKG + d + os.sep for d in ("core", "cute", "functional", "panel"))
# Locations listed per problem.
_MAX_WHERE = 3


def _where(sources: tuple[Any, ...]) -> str | None:
    """Return where the model's code traced an equation, from its source info and
    its callers' (`sources`, innermost first), or None if unknown.

    That is the innermost frame outside JAX and pypomp's machinery and, if that
    is a pypomp helper, the frames that called it, up to the first outside pypomp
    or in one of its bundled models. Equations inside a jitted JAX function have
    no such frame, so they take their caller's.
    """
    try:
        from jax._src import source_info_util as siu

        for source_info in sources:
            frames = [
                f
                for f in siu.user_frames(source_info.traceback)
                if not f.file_name.startswith(_MACHINERY)
            ]
            if frames:
                break
        else:
            return None
    except Exception:  # noqa: BLE001
        return None
    shown = []
    for f in frames[:_MAX_WHERE]:
        shown.append(f"{_short(f.file_name)}:{f.start_line} ({f.function_name})")
        if not f.file_name.startswith(_PKG) or "models" in f.file_name[len(_PKG) :]:
            break
    return ", called from ".join(shown) or None


def _short(path: str) -> str:
    """Return `path` relative to the working directory, or from the package's
    parent for pypomp's own files."""
    if path.startswith(_PKG):
        return os.path.relpath(path, os.path.dirname(os.path.dirname(_PKG)))
    try:
        rel = os.path.relpath(path)
    except ValueError:
        return path
    return path if rel.startswith("..") else rel


def _problems(*bads: _Bad) -> tuple[Problem, ...]:
    """Return the problems in `bads`, each with its distinct source locations."""
    merged: dict[str, list[str]] = {}
    for bad in bads:
        for what, sources in bad.items():
            where = merged.setdefault(what, [])
            for s in sources:
                w = _where(s)
                if w is not None and w not in where:
                    where.append(w)
    return tuple(Problem(what, tuple(where)) for what, where in sorted(merged.items()))


def _merge(bads: Iterable[_Bad]) -> _Bad:
    """Return one record of what the translator can't run from several."""
    out: _Bad = {}
    for bad in bads:
        for what, sources in bad.items():
            out.setdefault(what, []).extend(sources)
    return out


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


def _untranslatable(cj: ClosedJaxpr) -> _Bad:
    """Return the primitives and uses in `cj` that the translator can't run."""
    bad: _Bad = {}
    _runtime(cj.jaxpr, [True] * len(cj.jaxpr.invars), bad)
    return bad


# The mark of a runtime value that is a table, or part of one, still in memory.
_MEM = "in memory"


def _runtime(
    jaxpr: Any, marks: list[Any], bad: _Bad, outer: tuple[Any, ...] = ()
) -> list[Any]:
    """Return which outputs of `jaxpr` depend on runtime inputs, noting misuses.

    A mark is False for a constant, True for a runtime value in registers and
    :data:`_MEM` for a table or part of one, which a runtime index may read.
    `outer` holds the source info of the equations that call `jaxpr`.
    """
    env = dict(zip(jaxpr.invars, marks, strict=True))

    def rt(v: Any) -> Any:
        return not isinstance(v, Literal) and env.get(v, False)

    def reg(ms: list[Any]) -> list[bool]:
        return [bool(m) for m in ms]

    for eqn in jaxpr.eqns:
        name, p = eqn.primitive.name, eqn.params
        ins = [rt(v) for v in eqn.invars]
        src = (eqn.source_info, *outer)
        if name not in _SUPPORTED:
            _note(bad, name, src)
        for v in eqn.outvars:
            dt = getattr(v.aval, "dtype", None)
            if dt is not None and not _is_key(dt) and np.dtype(dt) not in _DTYPES:
                _note(bad, f"{np.dtype(dt)} values", src)
        if (
            name in ("dynamic_slice", "gather") and any(ins[1:]) and ins[0] is True
        ) or (name == "dynamic_update_slice" and any(ins[2:])):
            _note(bad, f"{name} of runtime values at a runtime index", src)
        if name == "gather" and p["dimension_numbers"].operand_batching_dims:
            _note(bad, "gather with batching dimensions", src)
        if name == "dot_general" and any(p["dimension_numbers"][1]):
            _note(bad, "batched dot_general", src)
        if name == "random_bits" and p["bit_width"] != 32:
            _note(bad, "random_bits with bit_width != 32", src)
        if name in _CALLS:
            outs = _runtime(_sub_jaxpr(eqn).jaxpr, ins, bad, src)
        elif name == "cond":
            branches = [
                _runtime(b.jaxpr, reg(ins[1:]), bad, src) for b in p["branches"]
            ]
            outs = [bool(ins[0]) or any(o) for o in zip(*branches, strict=True)]
        elif name == "while":
            cn, bn = p["cond_nconsts"], p["body_nconsts"]
            body = p["body_jaxpr"].jaxpr
            carry = reg(ins[cn + bn :])
            outs = _loop(body, reg(ins[cn : cn + bn]), carry, [], bad, src)
            if any(_runtime(p["cond_jaxpr"].jaxpr, reg(ins[:cn]) + outs, bad, src)):
                _note(bad, "while loop with a runtime trip count", src)
        elif name == "scan":
            nc, ncarry = p["num_consts"], p["num_carry"]
            ins = reg(ins)
            consts, carry, xs = ins[:nc], ins[nc : nc + ncarry], ins[nc + ncarry :]
            outs = _loop(p["jaxpr"].jaxpr, consts, carry, xs, bad, src)
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
    body: Any,
    consts: list[bool],
    carry: list[bool],
    xs: list[bool],
    bad: _Bad,
    outer: tuple[Any, ...],
) -> list[bool]:
    """Return the runtime marks of a loop's carry and stacked outputs."""
    while True:
        outs = _runtime(body, consts + carry + xs, bad, outer)
        new = [a or b for a, b in zip(carry, outs, strict=False)]
        if new == carry:
            return new + outs[len(carry) :]
        carry = new


def _particle_mixing(two: Any, probe: Any, J: int) -> _Bad:
    """Return what in a ``@vectorized`` rproc couples its particles.

    `two` and `probe` are jaxprs of one Euler step on 2 and on J particles; an axis
    of length J in `probe` is the particle axis. The kernel runs each particle on
    its own, as if J were 1, so a reduction, slice or concatenation over particles,
    a random draw without particles as its last axis, or a constant that depends on
    J would compute something other than XLA does.
    """
    bad: _Bad = {}
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


def _compare(
    two: Any, probe: Any, J: int, bad: _Bad, outer: tuple[Any, ...] = ()
) -> None:
    """Walk two traces of a step together, noting uses that couple particles;
    `outer` holds the source info of the equations that call them."""
    if len(two.eqns) != len(probe.eqns):
        _note(bad, "operations that depend on the number of particles")
        return
    for e2, e in zip(two.eqns, probe.eqns, strict=True):
        name, p = e.primitive.name, e.params
        src = (e.source_info, *outer)
        if e2.primitive.name != name:
            _note(bad, "operations that depend on the number of particles", src)
            return
        for v2, v in zip(e2.invars, e.invars, strict=True):
            if isinstance(v2, Literal) != isinstance(v, Literal) or (
                isinstance(v, Literal) and not _same_constant(v2.val, v.val)
            ):
                _note(bad, "constants that depend on the number of particles", src)
        shapes = [_shape(v) for v in [*e.invars, *e.outvars]]
        if any(s.count(J) > 1 for s in shapes):
            _note(bad, "several particle axes", src)
        ax = [_particle_axis(_shape(v), J) for v in e.invars]
        if name in ("random_bits", "random_split"):
            shape = tuple(p["shape"])
            k = _particle_axis(shape, J)
            if k is not None and k != len(shape) - 1:
                _note(bad, f"{name} with particles on a non-trailing axis", src)
            if k is None and tuple(e2.params["shape"])[-1:] == (1,):
                _note(bad, f"{name} with a trailing axis of length 1", src)
        elif name == "iota" and _particle_axis(_shape(e.outvars[0]), J) is not None:
            _note(bad, "iota over particles", src)
        elif name in _REDUCTIONS:
            if ax[0] is not None and ax[0] in p["axes"]:
                _note(bad, f"{name} over particles", src)
        elif name == "dot_general":
            (lc, rc), _ = p["dimension_numbers"]
            if ax[0] in lc or ax[1] in rc:
                _note(bad, "dot_general over particles", src)
        elif name == "scan":
            if any(a == 0 for a in ax[p["num_consts"] + p["num_carry"] :]):
                _note(bad, "scan over particles", src)
        elif any(a is not None for a in ax) and name not in _PARTICLEWISE:
            if not _keeps_particles(e, ax, J):
                _note(bad, f"{name} across particles", src)
        for s2, s in zip(_sub_jaxprs(e2), _sub_jaxprs(e), strict=True):
            _compare(s2, s, J, bad, src)


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


def _trace_dmeas(model: Pomp) -> ClosedJaxpr | None:
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


def _vectorized_jaxpr(model: Pomp, J: int) -> ClosedJaxpr:
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


def _coupling(model: Pomp) -> _Bad:
    """Return what in a ``@vectorized`` rproc couples particles, which the kernel
    can't run with one particle per thread."""
    two = _vectorized_jaxpr(model, 2).jaxpr
    probe = _vectorized_jaxpr(model, _PROBE_J).jaxpr
    return _particle_mixing(two, probe, _PROBE_J)


class _Analysis(NamedTuple):
    """What the translator would reject in a model, by the part it would run."""

    traced: _Traced
    rproc: _Bad
    """In the Euler step or the parameter transform, or coupling particles."""
    grads: _Bad
    """In the VJPs of those, and not in them."""
    dm_j: ClosedJaxpr | None
    dmeasure: _Bad
    """In the measurement density, or why it couldn't be traced."""


def _analyze(model: Pomp) -> _Analysis:
    """Trace `model`'s process model and density and return what the translator
    would reject in them."""
    traced = _trace_step(model)
    step_j, prep_j, vec, step_vjp_j, prep_vjp_j = traced
    rproc = _merge([_untranslatable(step_j), _untranslatable(prep_j)])
    if vec:
        rproc = _merge([rproc, _coupling(model)])
    vjps = _merge([_untranslatable(step_vjp_j), _untranslatable(prep_vjp_j)])
    grads = {k: v for k, v in vjps.items() if k not in rproc}
    try:
        dm_j = _trace_dmeas(model)
    except Exception as e:  # noqa: BLE001
        return _Analysis(traced, rproc, grads, None, {_failed("dmeas", e): []})
    dmeasure = {} if dm_j is None else _untranslatable(dm_j)
    return _Analysis(traced, rproc, grads, dm_j, dmeasure)


def _failed(what: str, e: Exception) -> str:
    """Return the problem of tracing `what` raising `e`."""
    return f"tracing {what} failed ({type(e).__name__}: {e})"
