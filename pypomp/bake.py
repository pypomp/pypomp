from __future__ import annotations

import ast
from contextlib import contextmanager
from hashlib import sha256
from numbers import Integral
import os
from pathlib import Path
import pickle
import sys
import tempfile
import textwrap
import time
from typing import Any
import warnings

import cloudpickle
import jax
import numpy as np

from ._bake_compat import (
    ArchiveValue,
    _directory,
    archive_directory,
    coerce_seed,
    r_uniform,
)

__all__ = ["bake", "stew", "freeze", "archive_directory", "ArchiveValue", "r_uniform"]

_FORMAT = "pypomp.recipes/1"
_MISSING = object()
_RESERVED = {"__builtins__", "_key", "_ingredients", "_system_time"}


class _Scope(dict):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    def __missing__(self, name):
        return self.parent[name]

    def __contains__(self, name):

        return super().__contains__(name) or name in self.parent


def _workspace(envir):
    if envir is not None:
        if not isinstance(envir, dict):
            raise TypeError("envir must be a dictionary")
        return envir
    frame = sys._getframe(2)
    try:
        if frame.f_locals is not frame.f_globals:
            raise ValueError("Inside a function or class, supply envir={...}")
        return frame.f_globals
    finally:
        del frame


@contextmanager
def _seeded(scope, seed, compatibility="native"):
    controlled = coerce_seed(seed) if compatibility == "R" else seed
    old = scope.get("_key", _MISSING)
    old_rng = scope.get("_rng", _MISSING)
    numpy_state = r_state = None
    if compatibility == "R":
        scope["_rng"] = r_uniform
        if controlled is not None:
            if r_uniform.get_state() is None:
                r_uniform.seed()
            r_state = r_uniform.get_state()
            numpy_state = np.random.get_state()
            np.random.seed(controlled & 0xFFFFFFFF)
            r_uniform.seed(controlled)
    if controlled is not None:
        scope["_key"] = jax.random.key(
            int(controlled) & 0xFFFFFFFF, impl="threefry2x32"
        )
    success = False
    try:
        yield
        success = True
    finally:
        if compatibility != "R" or success:
            if controlled is not None:
                if old is _MISSING:
                    scope.pop("_key", None)
                else:
                    scope["_key"] = old
            if numpy_state is not None:
                np.random.set_state(numpy_state)
                r_uniform.set_state(r_state)
        if compatibility == "R":
            if old_rng is _MISSING:
                scope.pop("_rng", None)
            else:
                scope["_rng"] = old_rng


def _wait_for_output(value):

    pending, seen = [value], {}
    while pending:
        obj = pending.pop()
        if id(obj) in seen:
            continue
        seen[id(obj)] = obj
        if isinstance(obj, dict):
            pending.extend(obj.values())
        elif isinstance(obj, (list, tuple)):
            pending.extend(obj)
        elif hasattr(obj, "block_until_ready"):
            obj.block_until_ready()
        else:
            first = True

            def child_is_leaf(_):
                nonlocal first
                result, first = not first, False
                return result

            pending.extend(jax.tree_util.tree_leaves(obj, is_leaf=child_is_leaf))


def _evaluate(operation, tree, workspace, seed, compatibility="native"):
    tail = None
    temporary = None
    if compatibility == "R" and operation != "stew" and tree.body:
        final = tree.body[-1]
        if isinstance(final, (ast.Assign, ast.AnnAssign)) and final.value is not None:
            temporary = "__recipe_assignment_value__"
            used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            while temporary in workspace or temporary in used:
                temporary += "_"
            capture = ast.Assign([ast.Name(temporary, ast.Store())], final.value)
            final.value = ast.Name(temporary, ast.Load())
            tree.body[-1:] = [capture, final, ast.Expr(ast.Name(temporary, ast.Load()))]
            ast.fix_missing_locations(tree)
    if operation != "stew" and tree.body and isinstance(tree.body[-1], ast.Expr):
        tail = compile(
            ast.Expression(tree.body.pop().value), "<bake>", "eval", dont_inherit=True
        )
    statements = compile(tree, f"<{operation}>", "exec", dont_inherit=True)
    scope = _Scope(workspace) if operation == "stew" else workspace
    cpu_start, wall_start = os.times(), time.perf_counter()
    with _seeded(scope, seed, compatibility):
        try:
            exec(statements, scope, scope)
            value = eval(tail, scope, scope) if tail is not None else None
        finally:
            if temporary is not None:
                scope.pop(temporary, None)
        if operation == "stew":
            value = {
                k: v
                for k, v in scope.items()
                if k not in _RESERVED and (compatibility != "R" or k != "_rng")
            }
            if any(not isinstance(k, str) for k in value):
                raise TypeError("stew bindings must have string names")
        _wait_for_output(value)
    elapsed, cpu_end = time.perf_counter() - wall_start, os.times()
    timing = dict(
        zip(
            ("user.self", "sys.self", "elapsed", "user.child", "sys.child"),
            (
                cpu_end.user - cpu_start.user,
                cpu_end.system - cpu_start.system,
                elapsed,
                cpu_end.children_user - cpu_start.children_user,
                cpu_end.children_system - cpu_start.children_system,
            ),
            strict=True,
        )
    )
    if operation == "bake" and value is None:
        warnings.warn(
            "bake expression evaluates to None; returning an empty list",
            UserWarning,
            stacklevel=3,
        )
        value = []
    return value, timing


def _save(path, record):

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            cloudpickle.dump(record, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parse(operation, expr, seed, compatibility, kind, normal_kind):
    if not isinstance(expr, str):
        raise TypeError("expr must be Python source text")
    if compatibility not in ("native", "R"):
        raise ValueError("compatibility must be 'native' or 'R'")
    if compatibility == "native" and (kind is not None or normal_kind is not None):
        raise ValueError("kind and normal_kind require compatibility='R'")
    if kind not in (None, "Mersenne-Twister") or normal_kind not in (None, "Inversion"):
        raise ValueError("Only R Mersenne-Twister/Inversion kind metadata is supported")
    if compatibility == "R":
        coerce_seed(seed)
    if (
        compatibility != "R"
        and seed is not None
        and (
            isinstance(seed, bool)
            or not isinstance(seed, Integral)
            or not 0 <= int(seed) < 2**32
        )
    ):
        raise ValueError("seed must be None or a non-boolean integer in [0, 2**32)")
    return ast.parse(textwrap.dedent(expr), filename=f"<{operation}>")


def _archive(
    operation,
    file,
    expr,
    seed,
    dependson,
    info,
    timing,
    directory,
    workspace,
    compatibility="native",
    kind=None,
    normal_kind=None,
):
    if not isinstance(info, bool) or not isinstance(timing, bool):
        raise TypeError("info and timing must be booleans")
    tree = _parse(operation, expr, seed, compatibility, kind, normal_kind)
    ingredients = {
        "code": sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
        "dependencies": sha256(cloudpickle.dumps(dependson)).hexdigest(),
        "seed": None
        if seed is None
        else (
            type(seed).__module__,
            type(seed).__qualname__,
            cloudpickle.dumps(seed) if compatibility == "R" else int(seed),
        ),
    }
    if compatibility == "R":
        ingredients.update(kind=kind, normal_kind=normal_kind)
    directory = _directory.get() if directory is None else directory
    if compatibility == "R":
        directory = os.getcwd() if directory is None else os.fspath(directory)
        path = (
            Path(str(directory).rstrip("/") + "/" + os.fspath(file))
            if directory
            else Path(file)
        )
    else:
        path = Path(file) if directory is None else Path(directory) / file
    if not path.parent.exists() and compatibility == "R":
        print(f"NOTE: creating archive directory '{path.parent}'.", file=sys.stderr)
    path.parent.mkdir(parents=True, exist_ok=True)
    field = "value" if operation == "bake" else "objects"
    record = None
    if path.exists():
        with path.open("rb") as stream:
            record = pickle.load(stream)
        if compatibility == "R" and isinstance(record, dict):
            if (
                operation == "bake"
                and record.get("format") == _FORMAT
                and record.get("operation") == operation
                and "value" in record
                and isinstance(record.get("system_time"), dict)
                and record.get("ingredients") is None
            ):
                record["ingredients"] = {
                    **ingredients,
                    "seed": record.get("seed"),
                    "kind": record.get("kind"),
                    "normal_kind": record.get("normal_kind"),
                }
                _save_record(path, record, compatibility)
            elif (
                operation == "stew"
                and "format" not in record
                and all(isinstance(k, str) for k in record)
            ):
                record = {
                    "format": _FORMAT,
                    "operation": operation,
                    "ingredients": ingredients,
                    "system_time": None,
                    "objects": record,
                }
                _save_record(path, record, compatibility)
        if (
            not isinstance(record, dict)
            or record.get("format") != _FORMAT
            or record.get("operation") != operation
            or not {"ingredients", "system_time", field} <= record.keys()
            or not isinstance(record["ingredients"], dict)
            or not (
                isinstance(record["system_time"], dict)
                or (compatibility == "R" and record["system_time"] is None)
            )
        ):
            raise ValueError(f"Invalid {operation} archive: {path}")
        if operation == "stew" and (
            not isinstance(record[field], dict)
            or any(not isinstance(k, str) for k in record[field])
        ):
            raise ValueError(f"Invalid stew bindings in archive: {path}")
        if record["ingredients"] != ingredients:
            record = None
    if record is None:
        value, elapsed = _evaluate(operation, tree, workspace, seed, compatibility)
        record = {
            "format": _FORMAT,
            "operation": operation,
            "ingredients": ingredients,
            "system_time": elapsed,
            field: value,
        }
        _save_record(path, record, compatibility)
    value = record[field]
    if operation == "stew":
        names = sorted(k for k in value if not k.startswith("_"))
        workspace.update((k, value[k]) for k in names)
        value = names
    if compatibility == "R" and operation == "bake":
        if info or timing:
            return ArchiveValue(
                value,
                record["ingredients"] if info else None,
                record["system_time"] if timing else None,
            )
        return value
    if info:
        workspace["_ingredients"] = record["ingredients"]
    if timing:
        workspace["_system_time"] = record["system_time"]
    return value


def bake(
    file: str | os.PathLike[str],
    expr: str,
    *,
    seed: int | np.integer | None = None,
    dependson: Any = None,
    info: bool = False,
    timing: bool = True,
    dir: str | os.PathLike[str] | None = None,
    envir: dict | None = None,
    compatibility: str = "native",
    kind: str | None = None,
    normal_kind: str | None = None,
) -> Any:

    return _archive(
        "bake",
        file,
        expr,
        seed,
        dependson,
        info,
        timing,
        dir,
        _workspace(envir),
        compatibility,
        kind,
        normal_kind,
    )


def stew(
    file: str | os.PathLike[str],
    expr: str,
    *,
    seed: int | np.integer | None = None,
    dependson: Any = None,
    info: bool = False,
    timing: bool = True,
    dir: str | os.PathLike[str] | None = None,
    envir: dict | None = None,
    compatibility: str = "native",
    kind: str | None = None,
    normal_kind: str | None = None,
) -> list[str]:

    return _archive(
        "stew",
        file,
        expr,
        seed,
        dependson,
        info,
        timing,
        dir,
        _workspace(envir),
        compatibility,
        kind,
        normal_kind,
    )


def _save_record(path, record, compatibility):
    if compatibility == "R":
        with path.open("wb") as stream:
            cloudpickle.dump(record, stream, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        _save(path, record)


def freeze(
    expr: str,
    *,
    seed=None,
    envir: dict | None = None,
    compatibility: str = "native",
    kind=None,
    normal_kind=None,
) -> Any:

    tree = _parse("freeze", expr, seed, compatibility, kind, normal_kind)
    return _evaluate("freeze", tree, _workspace(envir), seed, compatibility)[0]
