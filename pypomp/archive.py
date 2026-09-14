"""Computation archives in the spirit of R pomp's ``bake`` and ``stew``."""

from __future__ import annotations

import ast
import hashlib
import inspect
import operator
import os
import pickle
import textwrap
import uuid
from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path
from types import CodeType
from typing import IO, Any, SupportsIndex, TypeVar

import cloudpickle
import jax

__all__ = ["bake", "stew"]

T = TypeVar("T")

_FORMAT = "pypomp.archive/1"
# Pinned so dependency digests don't change with pickle.DEFAULT_PROTOCOL.
_PROTOCOL = 5


def bake(
    file: str | os.PathLike[str],
    fn: Callable[..., T],
    *,
    seed: SupportsIndex | None = None,
    dependson: Any = None,
) -> T:
    """Evaluate ``fn`` once and archive its value in ``file``.

    Later calls load the archived value instead of calling ``fn``, as long as
    the source of ``fn``, ``dependson``, and ``seed`` are unchanged. Otherwise
    ``fn`` is called again and the archive is replaced.

    Parameters
    ----------
    file : str or os.PathLike
        Archive path. Missing parent directories are created.
    fn : callable
        The computation. Called as ``fn(key)`` with a JAX key when ``seed`` is
        given, otherwise as ``fn()``.
    seed : int, optional
        Seed for the key passed to ``fn``.
    dependson : Any, optional
        Values the result depends on, e.g. ``(J, theta, data)``. Only the
        source of ``fn`` is tracked automatically: variables it reads from
        outside, and functions it calls, must be listed here for changes to
        invalidate the archive. Hashed via pickle, so pass parameters and
        data (e.g. ``model.theta``) rather than ``Pomp`` objects or sets,
        whose pickles vary between sessions.

    Returns
    -------
    Any
        The value returned by ``fn``.

    Notes
    -----
    Source is hashed as a syntax tree, so formatting and comment edits do not
    invalidate the archive. Only load archives you trust: unpickling can run
    arbitrary code.

    Examples
    --------
    >>> def run_pfilter(key):
    ...     model = build_model()
    ...     model.pfilter(J=J, key=key)
    ...     return model
    >>> model = pp.bake("archives/pfilter.pkl", run_pfilter, seed=123, dependson=J)
    """
    return _archive("bake", Path(file), fn, seed, dependson)


def stew(
    file: str | os.PathLike[str],
    fn: Callable[..., Mapping[str, Any]],
    *,
    seed: SupportsIndex | None = None,
    dependson: Any = None,
    namespace: MutableMapping[str, Any] | None = None,
) -> list[str]:
    """Like :func:`bake`, but restore several named objects into a namespace.

    ``fn`` returns a mapping of names to objects, which is archived and then
    written into ``namespace``. Ending ``fn`` with ``return locals()`` saves
    everything it defines.

    Parameters
    ----------
    file, fn, seed, dependson
        As in :func:`bake`.
    namespace : MutableMapping, optional
        Where to restore the objects. Defaults to the calling module's
        globals, which is the notebook namespace in Jupyter.

    Returns
    -------
    list of str
        The restored names.

    Examples
    --------
    >>> def simulate(key):
    ...     states, obs = model.simulate(nsim=10, key=key)
    ...     return {"states": states, "obs": obs}
    >>> pp.stew("archives/sims.pkl", simulate, seed=42, dependson=theta)
    ['states', 'obs']
    """
    if namespace is None:
        frame = inspect.currentframe()
        if frame is None or frame.f_back is None:  # pragma: no cover
            raise RuntimeError("cannot find the caller's globals; pass namespace")
        namespace = frame.f_back.f_globals
    objects: dict[str, Any] = _archive("stew", Path(file), fn, seed, dependson)
    namespace.update(objects)
    return list(objects)


def _archive(
    kind: str,
    file: Path,
    fn: Callable[..., Any],
    seed: SupportsIndex | None,
    dependson: Any,
) -> Any:
    if isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    seed = None if seed is None else operator.index(seed)
    ingredients = {
        "code": _code_digest(fn),
        "dependson": _digest(cloudpickle.dumps(dependson, protocol=_PROTOCOL)),
        "seed": seed,
    }
    if file.exists():
        with file.open("rb") as stream:
            if _read_header(stream, file, kind)["ingredients"] == ingredients:
                return pickle.load(stream)

    # Open the output before evaluating so an unwritable path fails early.
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(f".{file.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            value = fn() if seed is None else fn(jax.random.key(seed))
            if kind == "stew":
                value = _named_objects(value)
            header = {"format": _FORMAT, "kind": kind, "ingredients": ingredients}
            pickle.dump(header, stream, protocol=_PROTOCOL)
            cloudpickle.dump(value, stream, protocol=_PROTOCOL)
        os.replace(temporary, file)
    finally:
        temporary.unlink(missing_ok=True)
    return value


def _read_header(stream: IO[bytes], file: Path, kind: str) -> dict[str, Any]:
    # The header is a separate pickle, so a stale value is never loaded.
    try:
        header = pickle.load(stream)
    except Exception as error:
        raise ValueError(f"{file} is not a pypomp archive") from error
    if not isinstance(header, dict) or header.get("format") != _FORMAT:
        raise ValueError(f"{file} is not a pypomp archive")
    if header["kind"] != kind:
        raise ValueError(f"{file} was written by {header['kind']}, not {kind}")
    return header


def _named_objects(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(k, str) for k in value):
        raise TypeError("stew's fn must return a mapping with string keys")
    return dict(value)


def _code_digest(fn: Callable[..., Any]) -> str:
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        code = getattr(fn, "__code__", None)
        if not isinstance(code, CodeType):
            raise TypeError("fn must be a function") from None
        # No source (REPL, exec): hash the bytecode, which varies by Python version.
        return _digest(_code_bytes(code))
    try:
        return _digest(ast.dump(ast.parse(source)).encode())
    except SyntaxError:
        # A lambda inside a multi-line call can yield an unparsable fragment.
        return _digest(source.encode())


def _code_bytes(code: CodeType) -> bytes:
    # Excludes line numbers; frozensets are sorted since their order varies.
    consts = [
        _code_bytes(c)
        if isinstance(c, CodeType)
        else repr(sorted(map(repr, c)))
        if isinstance(c, frozenset)
        else repr(c)
        for c in code.co_consts
    ]
    parts = (code.co_code, code.co_names, code.co_varnames, code.co_freevars, consts)
    return repr(parts).encode()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
