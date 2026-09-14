"""Tests for the bake/stew computation archives."""

import functools
import importlib.util
import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

import pypomp as pp
from pypomp import archive
from tests.helpers.dummy import dummy_pomp


def _counted(value: Any = 1):
    calls = []

    def fn():
        calls.append(1)
        return value

    return fn, calls


def _import(path: Path, source: str):
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hit_skips_evaluation(tmp_path):
    fn, calls = _counted([1, 2, 3])
    path = tmp_path / "nested" / "value.pkl"
    assert pp.bake(path, fn) == [1, 2, 3]
    assert pp.bake(path, fn) == [1, 2, 3]
    assert len(calls) == 1


def test_none_is_archived(tmp_path):
    fn, calls = _counted(None)
    assert pp.bake(tmp_path / "v.pkl", fn) is None
    assert pp.bake(tmp_path / "v.pkl", fn) is None
    assert len(calls) == 1


def test_source_edits(tmp_path):
    path = tmp_path / "v.pkl"
    original = "def compute():\n    return 1\n"
    reformatted = "def compute():\n    # a comment\n\n    return (1)\n"
    changed = "def compute():\n    return 2\n"
    assert pp.bake(path, _import(tmp_path / "a.py", original).compute) == 1
    assert pp.bake(path, _import(tmp_path / "b.py", reformatted).compute) == 1
    assert pp.bake(path, _import(tmp_path / "c.py", changed).compute) == 2


def test_dependson_invalidates(tmp_path):
    fn, calls = _counted()
    path = tmp_path / "v.pkl"
    for dependson in (np.array([1, 2]), np.array([1, 2]), np.array([1, 3])):
        pp.bake(path, fn, dependson=dependson)
    assert len(calls) == 2


def test_seed_supplies_key(tmp_path):
    def draw(key):
        return jax.random.normal(key, (4,))

    a = pp.bake(tmp_path / "a.pkl", draw, seed=7)
    b = pp.bake(tmp_path / "b.pkl", draw, seed=7)
    np.testing.assert_array_equal(a, draw(jax.random.key(7)))
    np.testing.assert_array_equal(a, b)
    c = pp.bake(tmp_path / "a.pkl", draw, seed=8)
    assert not np.array_equal(a, c)


def test_numpy_integer_seed_matches_int(tmp_path):
    calls = []

    def fn(key):
        calls.append(1)
        return 0

    pp.bake(tmp_path / "v.pkl", fn, seed=5)
    pp.bake(tmp_path / "v.pkl", fn, seed=np.int64(5))
    assert len(calls) == 1


@pytest.mark.parametrize("seed", [True, 1.5, "1"])
def test_invalid_seed(tmp_path, seed):
    with pytest.raises(TypeError):
        pp.bake(tmp_path / "v.pkl", lambda key: 0, seed=seed)


@pytest.mark.parametrize("content", [pickle.dumps({"x": 1}), b"not a pickle"])
def test_foreign_file_is_rejected(tmp_path, content):
    path = tmp_path / "v.pkl"
    path.write_bytes(content)
    fn, calls = _counted()
    with pytest.raises(ValueError, match="not a pypomp archive"):
        pp.stew(path, fn, namespace={})
    assert calls == [] and path.read_bytes() == content


def test_bake_archive_rejected_by_stew(tmp_path):
    path = tmp_path / "v.pkl"
    pp.bake(path, lambda: 1)
    with pytest.raises(ValueError, match="written by bake"):
        pp.stew(path, lambda: {"x": 1}, namespace={})


def test_stale_value_is_not_loaded(tmp_path):
    path = tmp_path / "v.pkl"
    header = {"format": archive._FORMAT, "kind": "bake", "ingredients": {}}
    path.write_bytes(pickle.dumps(header) + b"unloadable")
    assert pp.bake(path, lambda: 3) == 3
    assert pp.bake(path, lambda: 3) == 3


class _Unpicklable:
    def __reduce__(self):
        raise RuntimeError("cannot pickle")


def _raise():
    raise RuntimeError("evaluation failed")


@pytest.mark.parametrize(
    "fn,message",
    [(_raise, "evaluation failed"), (_Unpicklable, "cannot pickle")],
)
def test_failure_preserves_archive(tmp_path, fn, message):
    path = tmp_path / "v.pkl"
    pp.bake(path, lambda: 1)
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match=message):
        pp.bake(path, fn)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_unwritable_path_fails_before_evaluation(tmp_path):
    (tmp_path / "file").write_text("")
    fn, calls = _counted()
    with pytest.raises(OSError):
        pp.bake(tmp_path / "file" / "v.pkl", fn)
    assert calls == []


def test_stew_restores_names(tmp_path):
    calls = []

    def fn(key):
        calls.append(1)
        x = jax.random.uniform(key)
        y = 2 * x
        return {"y": y, "x": x}

    first = {}
    assert pp.stew(tmp_path / "v.pkl", fn, seed=1, namespace=first) == ["y", "x"]
    assert first["y"] == 2 * first["x"]
    second = {}
    assert pp.stew(tmp_path / "v.pkl", fn, seed=1, namespace=second) == ["y", "x"]
    assert second == first and len(calls) == 1


def test_stew_defaults_to_caller_globals(tmp_path):
    try:
        pp.stew(tmp_path / "v.pkl", lambda: {"_stewed_value": 7})
        assert globals()["_stewed_value"] == 7
    finally:
        globals().pop("_stewed_value", None)


@pytest.mark.parametrize("value", [[1, 2], {1: "x"}])
def test_stew_requires_named_objects(tmp_path, value):
    with pytest.raises(TypeError, match="string keys"):
        pp.stew(tmp_path / "v.pkl", lambda: value, namespace={})
    assert list(tmp_path.iterdir()) == []


def test_multiline_lambda(tmp_path):
    calls = []
    for _ in range(2):
        value = pp.bake(
            tmp_path / "v.pkl",
            lambda: (
                calls.append(1),  # comment
                5,
            )[1],
        )
        assert value == 5
    assert len(calls) == 1


def test_sourceless_function_uses_bytecode():
    def compile_function(source):
        namespace = {}
        exec(source, namespace)
        return namespace["f"]

    same = compile_function("def f():\n    return {'a', 'b'}\n")
    moved = compile_function("\n\n\ndef f():\n    return {'a', 'b'}\n")
    changed = compile_function("def f():\n    return {'a', 'c'}\n")
    digest = archive._code_digest(same)
    assert archive._code_digest(moved) == digest
    assert archive._code_digest(changed) != digest


def test_non_function_callable_rejected(tmp_path):
    with pytest.raises(TypeError, match="must be a function"):
        pp.bake(tmp_path / "v.pkl", functools.partial(int, "1"))


def test_pomp_round_trip(tmp_path):
    def fn(key):
        model = dummy_pomp()
        model.pfilter(J=16, key=key)
        return model

    model = pp.bake(tmp_path / "model.pkl", fn, seed=3)
    loaded = pp.bake(tmp_path / "model.pkl", fn, seed=3)
    assert loaded is not model and loaded.theta == model.theta
    np.testing.assert_array_equal(
        loaded.results_history[-1].payload.to_array(),
        model.results_history[-1].payload.to_array(),
    )


def test_hit_in_fresh_process(tmp_path, capsys):
    # The child's hash seed differs from this process's, so this checks that
    # digests are stable across sessions.
    script = _import(
        tmp_path / "script.py",
        textwrap.dedent(
            r"""
            import sys

            import numpy as np

            import pypomp as pp


            def compute(key):
                print("evaluated")
                return 42


            DEPS = {"J": 100, "theta": np.arange(3.0), "name": "dacca"}
            # No source, so this uses the bytecode digest.
            namespace = {}
            exec("def f():\n    print('evaluated')\n    return 'x' in {'x', 'y'}\n", namespace)

            if __name__ == "__main__":
                print(pp.bake(sys.argv[1], compute, seed=1, dependson=DEPS))
                print(pp.bake(sys.argv[1] + ".2", namespace["f"]))
            """
        ),
    )
    path = tmp_path / "v.pkl"
    result = subprocess.run(
        [sys.executable, str(tmp_path / "script.py"), str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "PYTHONHASHSEED": "1",
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        },
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["evaluated", "42", "evaluated", "True"]
    assert pp.bake(path, script.compute, seed=1, dependson=script.DEPS) == 42
    assert pp.bake(f"{path}.2", script.namespace["f"]) is True
    assert "evaluated" not in capsys.readouterr().out
