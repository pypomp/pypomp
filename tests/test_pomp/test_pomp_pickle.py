"""Pickling and __setstate__ back-compatibility for Pomp."""

import contextlib
from hashlib import sha256
from importlib import import_module
import io
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
from typing import Any, cast
from unittest import mock
import warnings

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import pypomp as pp
from tests.helpers.assertions import pickle_roundtrip
from tests.helpers.dummy import (
    dummy_dmeas,
    dummy_pomp,
    dummy_rinit,
    dummy_rmeas,
    dummy_rproc,
    shifted_from_est,
    shifted_to_est,
)

bake_module = import_module("pypomp.bake")


def test_pickle_setstate_fallback_warning(base_pomp):
    """Test that unpickling issues a UserWarning when a function fails to reconstruct."""
    state = base_pomp.__getstate__()

    # Corrupt the bytes of rinit so unpickling fails
    state["_rinit_func_bytes"] = b"invalid_pickle_bytes"

    with pytest.warns(UserWarning, match="Failed to reconstruct rinit function"):
        pomp_unpickled = pickle_roundtrip(base_pomp)
        del pomp_unpickled.rinit
        # Directly trigger __setstate__ with corrupted state
        pomp_unpickled.__setstate__(state)

    assert pomp_unpickled.rinit is None


def test_dprior_construction_and_pickle_roundtrip(dprior_pomp):
    """Constructing with dprior= wraps it in a _DPrior; pickling round-trips it."""
    assert dprior_pomp.dprior is not None
    unpickled = pickle_roundtrip(dprior_pomp)
    assert unpickled.dprior is not None
    assert dprior_pomp == unpickled


def test_setstate_fresh_key_reconstruction_failure(base_pomp):
    """A corrupted fresh_key payload should warn and fall back to None."""
    state = base_pomp.__getstate__()
    state["_fresh_key_data"] = jnp.zeros((3,), dtype=jnp.uint32)

    pomp_unpickled = pickle_roundtrip(base_pomp)
    with pytest.warns(UserWarning, match="Failed to reconstruct JAX fresh_key"):
        pomp_unpickled.__setstate__(state)

    assert pomp_unpickled.fresh_key is None


def test_setstate_legacy_by_reference_loading(base_pomp):
    """Legacy pickles referenced functions by module+name instead of bytes."""
    state = base_pomp.__getstate__()
    del state["_rinit_func_bytes"]
    state["_rinit_func_name"] = "dummy_rinit"
    state["_rinit_module"] = __name__

    pomp_unpickled = pickle_roundtrip(base_pomp)
    del pomp_unpickled.rinit
    pomp_unpickled.__setstate__(state)

    assert pomp_unpickled.rinit is not None


def test_setstate_prewrapped_components(dprior_pomp):
    """If the pickled bytes already contain wrapped components, reuse them directly."""
    state = dprior_pomp.__getstate__()
    state["_rinit_func_bytes"] = cloudpickle.dumps(dprior_pomp.rinit)
    state["_rproc_func_bytes"] = cloudpickle.dumps(dprior_pomp.rproc)
    state["_dmeas_func_bytes"] = cloudpickle.dumps(dprior_pomp.dmeas)
    state["_rmeas_func_bytes"] = cloudpickle.dumps(dprior_pomp.rmeas)
    state["_dprior_func_bytes"] = cloudpickle.dumps(dprior_pomp.dprior)

    pomp_unpickled = pickle_roundtrip(dprior_pomp)
    del pomp_unpickled.rinit
    del pomp_unpickled.rproc
    del pomp_unpickled.dmeas
    del pomp_unpickled.rmeas
    del pomp_unpickled.dprior
    pomp_unpickled.__setstate__(state)

    assert type(pomp_unpickled.rinit) is type(dprior_pomp.rinit)
    assert type(pomp_unpickled.rproc) is type(dprior_pomp.rproc)
    assert type(pomp_unpickled.dmeas) is type(dprior_pomp.dmeas)
    assert type(pomp_unpickled.rmeas) is type(dprior_pomp.rmeas)
    assert type(pomp_unpickled.dprior) is type(dprior_pomp.dprior)


def test_setstate_rproc_dt_and_nstep_both_present(base_pomp):
    """Cover the raw-function rproc reconstruction path when both dt and nstep are set."""
    state = base_pomp.__getstate__()
    state["_rproc_func_bytes"] = cloudpickle.dumps(base_pomp.rproc.original_func)
    state["_rproc_dt"] = 0.5
    state["_rproc_nstep"] = 3
    state["_rproc_accumvars"] = ["X"]

    pomp_unpickled = pickle_roundtrip(base_pomp)
    del pomp_unpickled.rproc
    pomp_unpickled.__setstate__(state)

    assert pomp_unpickled.rproc is not None
    assert pomp_unpickled.rproc.nstep == 3


def test_setstate_rproc_missing_defaults_to_none(base_pomp):
    """If no rproc info is present at all in the pickled state, rproc defaults to None."""
    state = base_pomp.__getstate__()
    del state["_rproc_func_bytes"]

    pomp_unpickled = pickle_roundtrip(base_pomp)
    del pomp_unpickled.rproc
    pomp_unpickled.__setstate__(state)

    assert pomp_unpickled.rproc is None


def test_setstate_dmeas_missing_defaults_to_none():
    """A Pomp with dmeas=None (rmeas-only) should pickle/unpickle with dmeas staying None."""
    pomp = pp.Pomp(
        ys=pd.DataFrame({"y": [1.0, 2.0]}, index=[1.0, 2.0]),
        theta=pp.PompParameters({"X0": 0.0, "sigma": 0.1}),
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        rmeas=dummy_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    pomp.fresh_key = jax.random.key(1)
    assert pomp.dmeas is None

    unpickled = pickle_roundtrip(pomp)
    assert unpickled.dmeas is None


def test_accumvars_success(base_pomp):
    """Valid accumvars are resolved to state-name indices at construction time."""
    pomp = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        dmeas=dummy_dmeas,
        rmeas=dummy_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
        accumvars=["X"],
    )
    assert pomp._accumvars_indices == (0,)
    assert pomp.accumvars == ["X"]


@pytest.mark.parametrize(
    "module,name",
    [("recipes", "bake"), ("recipes", "stew"), ("_recipe_compat", "ArchiveValue")],
)
def test_archives_resolve_pre_rename_modules(module, name):
    reference = f"cpypomp.{module}\n{name}\n.".encode()
    assert pickle.loads(reference) is getattr(pp, name)


def test_bake_hit_invalidation_and_normalized_code(tmp_path):
    path = tmp_path / "nested/value.pkl"
    calls = []
    env = {"calls": calls, "a": 2}
    code = "calls.append(a)\na * 10"
    assert pp.bake(path, code, dependson=2, envir=env) == 20
    original = path.read_bytes()
    assert (
        pp.bake(path, "calls.append(a)\n\n# comment\na*10", dependson=2, envir=env)
        == 20
    )
    assert calls == [2] and path.read_bytes() == original
    env["a"] = 3
    assert pp.bake(path, code, dependson=2, envir=env) == 20
    assert pp.bake(path, code, dependson=3, envir=env) == 30
    assert pp.bake(path, code + " + 1", dependson=3, envir=env) == 31
    assert calls == [2, 3, 3]


def test_array_dependency_and_failed_hash(tmp_path):
    class BadDependency:
        def __reduce__(self):
            raise RuntimeError("cannot hash")

    path = tmp_path / "value.pkl"
    env = {"calls": []}
    code = "calls.append(1)\nlen(calls)"
    assert pp.bake(path, code, dependson=np.array([1, 2]), envir=env) == 1
    assert pp.bake(path, code, dependson=np.array([1, 2]), envir=env) == 1
    assert pp.bake(path, code, dependson=np.array([1, 3]), envir=env) == 2
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="cannot hash"):
        pp.bake(path, code, dependson=BadDependency(), envir=env)
    assert path.read_bytes() == before and len(env["calls"]) == 2


@pytest.mark.parametrize("code", ["", "x = 5", "None"])
def test_bake_none_warning_only_on_miss(tmp_path, code):
    path = tmp_path / "none.pkl"
    with pytest.warns(UserWarning, match="empty list"):
        assert pp.bake(path, code, envir={}) == []
    assert pp.bake(path, code, envir={}) == []


def test_stew_capture_visibility_functions_and_hit(tmp_path):
    path = tmp_path / "bundle.pkl"
    calls = []
    env = {"a": 3, "calls": calls, "x": -1}
    code = """
        calls.append(1)
        x = a
        y = [x * i for i in range(3)]
        def f(): return x + a
        _scratch = 99
        _ingredients = "reserved"
        12345
    """
    assert pp.stew(path, code, envir=env) == ["f", "x", "y"]
    assert env["y"] == [0, 3, 6] and env["f"]() == 6
    assert "_scratch" not in env and "_ingredients" not in env
    with path.open("rb") as stream:
        record = pickle.load(stream)
    assert record["objects"]["_scratch"] == 99
    assert "a" not in record["objects"] and "calls" not in record["objects"]
    for name in ("f", "x", "y"):
        env.pop(name)
    assert pp.stew(path, code, envir=env) == ["f", "x", "y"]
    assert calls == [1] and env["f"]() == 6 and env["x"] == 3


def test_stew_import_alias_empty_and_bad_bindings(tmp_path):
    env = {}
    assert pp.stew(tmp_path / "empty.pkl", "123", envir=env) == []
    assert pp.stew(tmp_path / "alias.pkl", "import math\nx=[]\ny=x", envir=env) == [
        "math",
        "x",
        "y",
    ]
    assert env["math"].sqrt(9) == 3 and env["x"] is env["y"]
    env.clear()
    pp.stew(tmp_path / "alias.pkl", "import math\nx=[]\ny=x", envir=env)
    assert env["x"] is env["y"]
    with pytest.raises(TypeError, match="string names"):
        pp.stew(tmp_path / "bad.pkl", "globals()[1]=2", envir={})
    assert not (tmp_path / "bad.pkl").exists()


@pytest.mark.parametrize("operation", [pp.bake, pp.stew])
def test_metadata_flags_do_not_recompute(tmp_path, operation):
    path = tmp_path / "metadata.pkl"
    env: dict[str, Any] = {"calls": []}
    code = "calls.append(1)\nx=2\nx"
    operation(path, code, info=True, envir=env)
    ingredients, timing = env["_ingredients"], env["_system_time"]
    assert set(ingredients) == {"code", "dependencies", "seed"}
    assert set(timing) == {
        "user.self",
        "sys.self",
        "elapsed",
        "user.child",
        "sys.child",
    }
    assert all(t >= 0 for t in timing.values())
    before = path.read_bytes()
    operation(path, code, info=False, timing=False, envir=env)
    assert env["_ingredients"] is ingredients and env["_system_time"] is timing
    fresh = {"calls": []}
    operation(path, code, info=False, timing=False, envir=fresh)
    assert "_ingredients" not in fresh and "_system_time" not in fresh
    assert env["calls"] == [1] and fresh["calls"] == [] and path.read_bytes() == before


@pytest.mark.parametrize("operation", [pp.bake, pp.stew])
def test_seed_repeatability_key_cleanup_and_failure(tmp_path, operation):
    code = "x = jax.random.normal(_key, (8,))\nx"
    existing = jax.random.key(888)
    env = {"jax": jax, "_key": existing}
    python_state, numpy_state = random.getstate(), np.random.get_state()
    first = None
    for name in ("one", "two"):
        operation(tmp_path / name, code, seed=42, envir=env)
        if name == "one":
            first = np.asarray(env["x"])
        else:
            assert first is not None
            np.testing.assert_array_equal(env["x"], first)
        assert env["_key"] is existing
    assert first is not None
    operation(tmp_path / "one", code, seed=43, envir=env)
    assert not np.array_equal(env["x"], first)
    assert random.getstate() == python_state
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    with pytest.raises(ZeroDivisionError):
        operation(tmp_path / "failure", "x=1\n1/0", seed=42, envir=env)
    assert env["_key"] is existing and not (tmp_path / "failure").exists()
    fresh = {"jax": jax}
    operation(tmp_path / "three", code, seed=42, envir=fresh)
    assert "_key" not in fresh


def test_seed_none_and_type_tag(tmp_path):
    key = jax.random.key(5)
    env = {"_key": key, "jax": jax, "calls": []}
    code = "calls.append(1)\njax.random.normal(_key)"
    pp.bake(tmp_path / "key", code, envir=env)
    assert env["_key"] is key
    for seed in (5, 5, np.int64(5)):
        pp.bake(tmp_path / "seed", code, seed=seed, envir=env)
    assert len(env["calls"]) == 3


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.0, [1], "1"])
def test_invalid_seed(tmp_path, seed):
    with pytest.raises(ValueError, match="seed"):
        pp.bake(tmp_path / "bad", "1", seed=seed, envir={})


def test_workspace_and_source_validation(tmp_path):
    with pytest.raises(ValueError, match="envir"):
        pp.bake(tmp_path / "nested", "1")
    with pytest.raises(TypeError, match="dictionary"):
        pp.stew(tmp_path / "bad", "1", envir=cast(Any, []))
    with pytest.raises(TypeError, match="source text"):
        pp.bake(tmp_path / "bad", cast(Any, lambda: 1), envir={})
    with pytest.raises(TypeError, match="booleans"):
        pp.bake(tmp_path / "bad", "1", info=cast(Any, "false"), envir={})
    with pytest.raises(SyntaxError):
        pp.bake(tmp_path / "syntax", "!", envir={})


def test_top_level_workspace_and_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = {"pp": pp}
    exec(
        "value=pp.bake('nested/value.pkl','2+3')\nnames=pp.stew('bundle.pkl','x=7')",
        env,
        env,
    )
    assert env["value"] == 5 and env["names"] == ["x"] and env["x"] == 7
    assert pp.bake("value.pkl", "2+3", dir=tmp_path / "nested", envir={}) == 5
    assert pp.bake(tmp_path / "nested/value.pkl", "2+3", dir="ignored", envir={}) == 5


def test_corrupt_wrong_operation_and_invalid_record(tmp_path):
    path = tmp_path / "bad"
    path.write_bytes(b"not pickle data")
    with pytest.raises(pickle.UnpicklingError):
        pp.bake(path, "1", envir={})
    path.write_bytes(pickle.dumps({"format": "unknown"}))
    with pytest.raises(ValueError, match="Invalid"):
        pp.bake(path, "1", envir={})
    path.unlink()
    pp.bake(path, "1", envir={})
    with pytest.raises(ValueError, match="Invalid"):
        pp.stew(path, "x=1", envir={})


@pytest.mark.parametrize("operation", [pp.bake, pp.stew])
def test_atomic_failure_preserves_archive_and_destination(tmp_path, operation):
    class Unserializable:
        def __reduce__(self):
            raise RuntimeError("serialize failure")

    path = tmp_path / "existing.pkl"
    env = {"bad": Unserializable(), "x": "old"}
    operation(path, "x=1\nx", envir=env)
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="serialize failure"):
        operation(path, "x=bad\nx", envir=env)
    assert path.read_bytes() == before and list(tmp_path.iterdir()) == [path]
    if operation is pp.stew:
        assert env["x"] == 1


def test_replace_failure_preserves_previous_archive(tmp_path, monkeypatch):
    path = tmp_path / "value.pkl"
    pp.bake(path, "1", envir={})
    before = path.read_bytes()

    def failed_replace(*args):
        raise OSError("replace failure")

    monkeypatch.setattr(bake_module.os, "replace", failed_replace)
    with pytest.raises(OSError, match="replace failure"):
        pp.bake(path, "2", envir={})
    assert path.read_bytes() == before and list(tmp_path.iterdir()) == [path]


def test_bake_model_direct_return_and_key_history(tmp_path):
    code = "model=build()\nmodel.pfilter(J=16, key=_key)\nmodel"
    model = pp.bake(tmp_path / "model", code, seed=7, envir={"build": dummy_pomp})
    loaded = pp.bake(tmp_path / "model", code, seed=7, envir={})
    assert isinstance(loaded, pp.Pomp) and loaded.theta == model.theta
    assert loaded.fresh_key is not None and model.fresh_key is not None
    np.testing.assert_array_equal(
        jax.random.key_data(loaded.fresh_key), jax.random.key_data(model.fresh_key)
    )
    np.testing.assert_array_equal(
        loaded.results_history[-1].payload.to_array(),
        model.results_history[-1].payload.to_array(),
    )


def test_timing_excludes_serialization_and_waits_for_output(tmp_path, monkeypatch):
    delay = 0.05

    class Pending:
        def block_until_ready(self):
            time.sleep(delay)

    saved = cloudpickle.dump

    def slow_save(*args, **kwargs):
        time.sleep(0.15)
        return saved(*args, **kwargs)

    monkeypatch.setattr(bake_module.cloudpickle, "dump", slow_save)
    env: dict[str, Any] = {"pending": Pending()}
    start = time.perf_counter()
    pp.bake(tmp_path / "timing", "pending", envir=env)
    elapsed = time.perf_counter() - start
    recorded = env["_system_time"]["elapsed"]
    assert recorded >= delay and elapsed - recorded >= 0.15
    original = dict(env["_system_time"])
    pp.bake(tmp_path / "timing", "pending", envir=env)
    assert env["_system_time"] == original


def test_pomp_and_result_fresh_process_reload(tmp_path):
    archive = tmp_path / "model.pkl"
    code = (
        "model=build()\n"
        "model.par_trans=pp.ParTrans(to_est, from_est)\n"
        "model.pfilter(J=16, key=_key)\nresult=model.results_history[-1]"
    )
    env = {
        "build": dummy_pomp,
        "pp": pp,
        "to_est": shifted_to_est,
        "from_est": shifted_from_est,
    }
    assert pp.stew(archive, code, seed=7, envir=env) == ["model", "result"]
    assert len(env["model"].results_history) == 1
    child = """
import sys
import jax
import numpy as np
import pypomp as pp
env = {}
pp.stew(sys.argv[1], sys.argv[2], seed=7, envir=env)
model = env['model']
assert len(model.results_history) == 1
assert model.par_trans.to_est({'sigma': 2.0}) == {'sigma': 3.0}
assert model.par_trans.from_est({'sigma': 3.0}) == {'sigma': 2.0}
np.testing.assert_array_equal(model.results_history[-1].payload.to_array(),
                              env['result'].payload.to_array())
model.pfilter(J=16, key=jax.random.key(8))
assert len(model.results_history) == 2
print('FRESH_PROCESS_PASS')
"""
    completed = subprocess.run(
        [sys.executable, "-c", child, str(archive), code],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        timeout=90,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert "FRESH_PROCESS_PASS" in completed.stdout


class _RArchiveAPI:
    def __getattr__(self, name):
        return getattr(pp, name)

    def bake(self, *args, **kwargs):
        kwargs.setdefault("compatibility", "R")
        kwargs.setdefault("dir", "")
        kwargs.setdefault("timing", False)
        return pp.bake(*args, **kwargs)

    def stew(self, *args, **kwargs):
        kwargs.setdefault("compatibility", "R")
        kwargs.setdefault("dir", "")
        return pp.stew(*args, **kwargs)


def _archive_native(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_archive_native(x) for x in value]
    if isinstance(value, dict):
        return {k: _archive_native(v) for k, v in value.items()}
    return value


def _archive_equal(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _archive_equal(x, y) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_archive_equal(a[k], b[k]) for k in a)
    return a == b


@pytest.fixture
def archive_rng_state():
    numpy_state = np.random.get_state()
    r_state = pp.r_uniform.get_state()
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        pp.r_uniform.set_state(r_state)


@pytest.mark.parametrize(
    "source,expected",
    [
        pytest.param(
            r"""v=pp.bake(root/'v','np.arange(1, 0+1, dtype=float)',envir=env); w=pp.bake(root/'v','np.arange(1, 0+1, dtype=float)',envir=env)
result=[len(v),bool(np.array_equal(v,np.arange(1,0+1,dtype=float))),bool(np.array_equal(v,w))]""",
            [0, True, True],
            id="B-array-0: Array length 0, all elements and cache reload",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','np.arange(1, 1+1, dtype=float)',envir=env); w=pp.bake(root/'v','np.arange(1, 1+1, dtype=float)',envir=env)
result=[len(v),bool(np.array_equal(v,np.arange(1,1+1,dtype=float))),bool(np.array_equal(v,w))]""",
            [1, True, True],
            id="B-array-1: Array length 1, all elements and cache reload",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','np.arange(1, 17+1, dtype=float)',envir=env); w=pp.bake(root/'v','np.arange(1, 17+1, dtype=float)',envir=env)
result=[len(v),bool(np.array_equal(v,np.arange(1,17+1,dtype=float))),bool(np.array_equal(v,w))]""",
            [17, True, True],
            id="B-array-17: Array length 17, all elements and cache reload",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','np.arange(1, 1024+1, dtype=float)',envir=env); w=pp.bake(root/'v','np.arange(1, 1024+1, dtype=float)',envir=env)
result=[len(v),bool(np.array_equal(v,np.arange(1,1024+1,dtype=float))),bool(np.array_equal(v,w))]""",
            [1024, True, True],
            id="B-array-1024: Array length 1024, all elements and cache reload",
        ),
        pytest.param(
            r"""pp.bake(root/'v','np.array([np.nan,np.inf,-np.inf])',envir=env)
v=pp.bake(root/'v','np.array([np.nan,np.inf,-np.inf])',envir=env)
result=[bool(np.isnan(v[0])),bool(v[1]==np.inf),bool(v[2]==-np.inf)]""",
            [True, True, True],
            id="B-special: NaN and infinities survive serialization",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','np.arange(1,13,dtype=float).reshape(3,4)',envir=env)
result=[*v.shape,bool(np.array_equal(v.ravel(),np.arange(1,13,dtype=float)))]""",
            [3, 4, True],
            id="B-matrix: Matrix shape and all values",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','2+3j',envir=env)
result=[v.real,v.imag]""",
            [2, 3],
            id="B-complex: Complex values",
        ),
        pytest.param(
            r"""v=pp.bake(root/'café/值',repr('αβ café'),envir=env)
result=[v,(root/'café/值').exists()]""",
            ["αβ café", True],
            id="B-unicode: Unicode values and path",
        ),
        pytest.param(
            r"""result=pp.stew(root/'v','123',envir=env)""",
            [],
            id="B-empty-stew: No assignments yields no restored names",
        ),
        pytest.param(
            r"""env['x']=1
n=pp.stew(root/'v','x=7\n_hidden=99',envir=env)
r=pickle.loads((root/'v').read_bytes())
result=[n,env['x'],r['objects']['_hidden']]""",
            [["x"], 7, 99],
            id="B-hidden: Hidden saved bindings and visible overwrite",
        ),
        pytest.param(
            r"""with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    v=pp.bake(root/'v','None',envir=env)
    pp.bake(root/'v','None',envir=env)
result=[len(v),len(w)]""",
            [0, 1],
            id="B-none: NULL/None produces empty list and one warning",
        ),
        pytest.param(
            r"""result=pp.bake(root/'v','x=5',envir=env)""",
            5,
            id="B-assignment: Final assignment returns its assigned value",
        ),
        pytest.param(
            r"""def f():
    local_only=27
    pp.stew(root/'v','out=local_only')
try:
    f()
    result=27
except (ValueError,NameError):
    result='lookup-error'""",
            "lookup-error",
            id="B-nested: Implicit nested caller lookup follows pinned R failure behavior",
        ),
        pytest.param(
            r"""try:
    pp.bake(root/'v','1',seed=0,envir=env)
    result=False
except (ValueError,TypeError):
    result=True""",
            False,
            id="B-seed-zero: Seed boundary: zero",
        ),
        pytest.param(
            r"""try:
    pp.bake(root/'v','1',seed=2147483647,envir=env)
    result=False
except (ValueError,TypeError):
    result=True""",
            False,
            id="B-seed-max-common: Seed boundary: max-common",
        ),
        pytest.param(
            r"""try:
    pp.bake(root/'v','1',seed=2**32-1,envir=env)
    result=False
except (ValueError,TypeError):
    result=True""",
            True,
            id="B-seed-max-python: Seed boundary: max-python",
        ),
        pytest.param(
            r"""try:
    pp.bake(root/'v','1',seed=-1,envir=env)
    result=False
except (ValueError,TypeError):
    result=True""",
            False,
            id="B-seed-negative: Seed boundary: negative",
        ),
        pytest.param(
            r"""try:
    pp.bake(root/'v','1',seed=True,envir=env)
    result=False
except (ValueError,TypeError):
    result=True""",
            False,
            id="B-seed-bool: Seed boundary: bool",
        ),
        pytest.param(
            r"""try:
    pp.bake(root/'v','1',seed=1.9,envir=env)
    result=False
except (ValueError,TypeError):
    result=True""",
            False,
            id="B-seed-float: Seed boundary: float",
        ),
        pytest.param(
            r"""try:
    pp.bake(root/'v','1',seed=[1,2],envir=env)
    result=False
except (ValueError,TypeError):
    result=True""",
            False,
            id="B-seed-vector: Seed boundary: vector",
        ),
        pytest.param(
            r"""try:
    pp.bake(root/'v','1',seed='1',envir=env)
    result=False
except (ValueError,TypeError):
    result=True""",
            False,
            id="B-seed-string: Seed boundary: string",
        ),
        pytest.param(
            r"""try:
    pp.bake(root/'v','1',seed=float('nan'),envir=env)
    result=False
except (ValueError,TypeError):
    result=True""",
            True,
            id="B-seed-nan: Seed boundary: nan",
        ),
        pytest.param(
            r"""a=pp.bake(root/'a','jax.random.uniform(_key,(16,))',seed=0,envir=env)
b=pp.bake(root/'b','jax.random.uniform(_key,(16,))',seed=0,envir=env)
result=[bool(np.array_equal(a,b)),len(a)]""",
            [True, 16],
            id="B-repeat-0: Seed 0 repeats across independent files",
        ),
        pytest.param(
            r"""a=pp.bake(root/'a','jax.random.uniform(_key,(16,))',seed=7,envir=env)
b=pp.bake(root/'b','jax.random.uniform(_key,(16,))',seed=7,envir=env)
result=[bool(np.array_equal(a,b)),len(a)]""",
            [True, 16],
            id="B-repeat-7: Seed 7 repeats across independent files",
        ),
        pytest.param(
            r"""a=pp.bake(root/'a','jax.random.uniform(_key,(16,))',seed=2147483647,envir=env)
b=pp.bake(root/'b','jax.random.uniform(_key,(16,))',seed=2147483647,envir=env)
result=[bool(np.array_equal(a,b)),len(a)]""",
            [True, 16],
            id="B-repeat-2147483647: Seed 2147483647 repeats across independent files",
        ),
        pytest.param(
            r"""result=np.asarray(pp.bake(root/'v','_rng.uniform(size=4)',seed=7,envir=env)).tolist()""",
            [
                0.9889092978555709,
                0.397745453286916,
                0.11569777876138687,
                0.06974867871031165,
            ],
            id="B-raw-rng: Exact random draws for the same seed",
        ),
        pytest.param(
            r"""env['_key']=jax.random.key(99)
before=env['_key']
try:
    pp.bake(root/'v','jax.random.uniform(_key)\nraise RuntimeError()',seed=7,envir=env)
except RuntimeError:
    pass
result=env['_key'] is before""",
            False,
            id="B-seed-error-state: Pinned R seeded failure leaves the temporary RNG state active",
        ),
        pytest.param(
            r"""env['counter']=[0]
code='counter[0]+=1\ncounter[0]'
a=pp.bake(root/'v',code,dependson=np.array([1,2]),envir=env)
env['other']=99
b=pp.bake(root/'v',code,dependson=np.array([1,2]),envir=env)
d=pp.bake(root/'v',code,dependson=np.array([1,3]),envir=env)
result=[a,b,d]""",
            [1, 1, 2],
            id="B-dependencies: Same-shaped changed dependencies invalidate; undeclared changes do not",
        ),
        pytest.param(
            r"""env['counter']=[0]
code='counter[0]+=1\ncounter[0]'
result=[pp.bake(root/'v',code,dependson=float('nan'),envir=env) for _ in range(2)]""",
            [1, 1],
            id="B-nan-dependency: NaN dependency remains a hit",
        ),
        pytest.param(
            r"""env['counter']=[0]
a=pp.bake(root/'v','counter[0]+=1\n1\ncounter[0]',envir=env)
b=pp.bake(root/'v','counter[0]+=1\n1.0\ncounter[0]',envir=env)
result=[a,b]""",
            [1, 2],
            id="B-code-literal-type: Integer-to-float source literal invalidates",
        ),
        pytest.param(
            r"""pp.bake(root/'v','1',envir=env)
before=(root/'v').read_bytes()
try:
    pp.bake(root/'v','raise RuntimeError()',envir=env)
    error=False
except RuntimeError:
    error=True
result=[error,(root/'v').read_bytes()==before]""",
            [True, True],
            id="B-exception-old-file: Evaluation error preserves the prior archive",
        ),
        pytest.param(
            r"""pp.stew(root/'v',"_ingredients='user'\n_system_time='user'",envir=env,info=True)
result=isinstance(env['_ingredients'],dict) and isinstance(env['_system_time'],dict)""",
            True,
            id="B-reserved: User-written internal metadata is replaced by archive metadata",
        ),
        pytest.param(
            r"""result=pp.stew(root/'v','z=1\na=2\nm=3',envir=env)""",
            ["a", "m", "z"],
            id="B-visible-order: ASCII visible names are sorted",
        ),
        pytest.param(
            r"""path=root/'absolute'
pp.bake(path,'1',dir=root/'prefix',envir=env)
result=[path.exists(),(Path(str(root/'prefix')+'/'+str(path))).exists()]""",
            [False, True],
            id="B-path-absolute: Absolute file with explicit directory uses R concatenation",
        ),
        pytest.param(
            r"""pp.bake(root/'v','np.arange(1,1000001,dtype=float)',envir=env)
v=pp.bake(root/'v','np.arange(1,1000001,dtype=float)',envir=env)
result=[len(v),float(v.sum()),bool(np.array_equal(v,np.arange(1,1000001,dtype=float)))]""",
            [1000000, 500000500000, True],
            id="S-large-array: One million float64 values and full reload",
        ),
        pytest.param(
            r"""code='\n'.join(f'x{i:04d}={i}' for i in range(2000))
pp.stew(root/'v',code,envir=env)
n=pp.stew(root/'v',code,envir=env)
result=[len(n),sum(env[k] for k in n),n==[f'x{i:04d}' for i in range(2000)]]""",
            [2000, 1999000, True],
            id="S-many-bindings: 2,000 named bindings and overwrite on reload",
        ),
        pytest.param(
            r"""env['counter']=[0]
for _ in range(1001):
    v=pp.bake(root/'v','counter[0]+=1\ncounter[0]',envir=env)
result=[env['counter'][0],v]""",
            [1, 1],
            id="S-hits: 1,000 hits execute the expression only once",
        ),
        pytest.param(
            r"""env['counter']=[0]
for i in range(40):
    v=pp.bake(root/'v','counter[0]+=1\ncounter[0]',dependson=i,envir=env)
result=[env['counter'][0],v]""",
            [40, 40],
            id="S-stale: 40 consecutive dependency invalidations",
        ),
        pytest.param(
            r"""code="e={'value':7}\ne['self']=e\ne"
pp.bake(root/'v',code,envir=env)
v=pp.bake(root/'v',code,envir=env)
result=[v is v['self'],v['self']['value']]""",
            [True, 7],
            id="S-cycle: Self-referential object graph survives miss and reload",
        ),
        pytest.param(
            r"""code="x={'value':1}\ny=x"
pp.stew(root/'v',code,envir=env)
pp.stew(root/'v',code,envir=env)
env['x']['value']=7
result=[env['x'] is env['y'],env['y']['value']]""",
            [True, 7],
            id="S-alias: Shared binding identity survives stew reload",
        ),
        pytest.param(
            r"""pp.bake(root/'v','np.arange(1000)',envir=env)
raw=(root/'v').read_bytes()
result=0
for n in [0,1,16,len(raw)//2]:
    (root/'bad').write_bytes(raw[:n])
    try:
        pp.bake(root/'bad','1',envir=env)
    except (EOFError,pickle.UnpicklingError,ValueError):
        result+=1""",
            4,
            id="S-truncated: Truncation at four archive offsets raises errors",
        ),
        pytest.param(
            r"""pp.bake(root/'v','1',envir=env)
before=(root/'v').read_bytes()
def broken(obj,stream,**kwargs):
    stream.write(b'partial')
    raise OSError('write failure')
with mock.patch.object(cloudpickle,'dump',broken):
    try:
        pp.bake(root/'v','2',envir=env)
        error=False
    except OSError:
        error=True
result=[error,(root/'v').read_bytes()==before]""",
            [True, False],
            id="S-write-fault: Explicit R direct-write failure leaves partial destination bytes",
        ),
        pytest.param(
            r"""code='x=7\nfor i in range(128): x=[x]\nx'
pp.bake(root/'v',code,envir=env)
v=pp.bake(root/'v',code,envir=env)
n=0
while isinstance(v,list):
    n+=1
    v=v[0]
result=[n,v]""",
            [128, 7],
            id="S-deep: 128 nested containers survive reload",
        ),
        pytest.param(
            r"""env['restore_guard']=True
code='assert restore_guard\nx=7\ndef f(): return 2*x'
pp.stew(root/'v',code,envir=env)
child="import pypomp as pp,sys; e={}; pp.stew(sys.argv[1],sys.argv[2],compatibility='R',dir='',envir=e);print(e['f']())"
p=subprocess.run([sys.executable,'-c',child,str(root/'v'),code],capture_output=True,text=True,check=True,timeout=60)
result=int(p.stdout.strip())""",
            14,
            id="S-fresh-function: Function with local bindings loads in a fresh process",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','3',envir=env,info=False,timing=False)
r=pickle.loads((root/'v').read_bytes())
result=[v.value if isinstance(v,pp.ArchiveValue) else v,isinstance(v,pp.ArchiveValue) and v.ingredients is not None,isinstance(v,pp.ArchiveValue) and v.system_time is not None,'ingredients' in r,'system_time' in r]""",
            [3, False, False, True, True],
            id="A-bake-flags-00: bake: info=False, timing=False; stored versus exposed metadata",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','3',envir=env,info=False,timing=True)
r=pickle.loads((root/'v').read_bytes())
result=[v.value if isinstance(v,pp.ArchiveValue) else v,isinstance(v,pp.ArchiveValue) and v.ingredients is not None,isinstance(v,pp.ArchiveValue) and v.system_time is not None,'ingredients' in r,'system_time' in r]""",
            [3, False, True, True, True],
            id="A-bake-flags-01: bake: info=False, timing=True; stored versus exposed metadata",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','3',envir=env,info=True,timing=False)
r=pickle.loads((root/'v').read_bytes())
result=[v.value if isinstance(v,pp.ArchiveValue) else v,isinstance(v,pp.ArchiveValue) and v.ingredients is not None,isinstance(v,pp.ArchiveValue) and v.system_time is not None,'ingredients' in r,'system_time' in r]""",
            [3, True, False, True, True],
            id="A-bake-flags-10: bake: info=True, timing=False; stored versus exposed metadata",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','3',envir=env,info=True,timing=True)
r=pickle.loads((root/'v').read_bytes())
result=[v.value if isinstance(v,pp.ArchiveValue) else v,isinstance(v,pp.ArchiveValue) and v.ingredients is not None,isinstance(v,pp.ArchiveValue) and v.system_time is not None,'ingredients' in r,'system_time' in r]""",
            [3, True, True, True, True],
            id="A-bake-flags-11: bake: info=True, timing=True; stored versus exposed metadata",
        ),
        pytest.param(
            r"""pp.stew(root/'v','x=3',envir=env,info=False,timing=False)
r=pickle.loads((root/'v').read_bytes())
result=[env['x'],'_ingredients' in env,'_system_time' in env,'ingredients' in r,'system_time' in r]""",
            [3, False, False, True, True],
            id="A-stew-flags-00: stew: info=False, timing=False; stored versus exposed metadata",
        ),
        pytest.param(
            r"""pp.stew(root/'v','x=3',envir=env,info=False,timing=True)
r=pickle.loads((root/'v').read_bytes())
result=[env['x'],'_ingredients' in env,'_system_time' in env,'ingredients' in r,'system_time' in r]""",
            [3, False, True, True, True],
            id="A-stew-flags-01: stew: info=False, timing=True; stored versus exposed metadata",
        ),
        pytest.param(
            r"""pp.stew(root/'v','x=3',envir=env,info=True,timing=False)
r=pickle.loads((root/'v').read_bytes())
result=[env['x'],'_ingredients' in env,'_system_time' in env,'ingredients' in r,'system_time' in r]""",
            [3, True, False, True, True],
            id="A-stew-flags-10: stew: info=True, timing=False; stored versus exposed metadata",
        ),
        pytest.param(
            r"""pp.stew(root/'v','x=3',envir=env,info=True,timing=True)
r=pickle.loads((root/'v').read_bytes())
result=[env['x'],'_ingredients' in env,'_system_time' in env,'ingredients' in r,'system_time' in r]""",
            [3, True, True, True, True],
            id="A-stew-flags-11: stew: info=True, timing=True; stored versus exposed metadata",
        ),
        pytest.param(
            r"""(root/'v').write_bytes(pickle.dumps(3))
try:
    pp.bake(root/'v','3',envir=env)
    result=False
except ValueError:
    result=True""",
            True,
            id="A-raw-file: A raw value without recipe metadata is rejected",
        ),
        pytest.param(
            r"""pp.bake(root/'v','3',envir=env)
before=(root/'v').read_bytes()
try:
    pp.stew(root/'v','x=3',envir=env)
    error=False
except ValueError:
    error=True
result=[error,(root/'v').read_bytes()==before]""",
            [True, True],
            id="A-wrong-operation: Loading a bake file with stew fails without rewriting",
        ),
        pytest.param(
            r"""(root/'parent').write_text('keep')
try:
    pp.bake(root/'parent/v','3',envir=env)
    error=False
except OSError:
    error=True
result=[error,(root/'parent').read_text()]""",
            [True, "keep"],
            id="A-parent-file: An ordinary file cannot be used as a directory",
        ),
        pytest.param(
            r"""root.chmod(0o555)
try:
    try:
        pp.bake(root/'v','3',envir=env)
        error=False
    except OSError:
        error=True
finally:
    root.chmod(0o700)
result=[error,(root/'v').exists()]""",
            [True, False],
            id="A-readonly: A new archive cannot be created in a read-only directory",
        ),
        pytest.param(
            r"""pp.bake(root/'v','3',envir=env)
with (root/'v').open('ab') as f: f.write(b'trailing')
result=pp.bake(root/'v','3',envir=env)""",
            3,
            id="A-trailing-bytes: A valid first object followed by trailing bytes",
        ),
        pytest.param(
            r"""old=os.umask(0o022)
try:
    pp.bake(root/'v','3',envir=env)
finally:
    os.umask(old)
result=(root/'v').stat().st_mode & 0o777""",
            420,
            id="A-permissions: R-mode file permissions under umask 022",
        ),
        pytest.param(
            r"""pp.bake(root/'target','1',envir=env)
(root/'link').symlink_to(root/'target')
pp.bake(root/'link','2',envir=env)
result=[(root/'link').is_symlink(),pickle.loads((root/'target').read_bytes())['value']]""",
            [True, 2],
            id="A-symlink: Recomputation through a symbolic link",
        ),
        pytest.param(
            r"""pp.bake(root/'v',"{1:'one','a':'letter'}",envir=env)
v=pp.bake(root/'v',"{1:'one','a':'letter'}",envir=env)
result=[v[1],v['a']]""",
            ["one", "letter"],
            id="A-mixed-keys: Heterogeneous Python dictionary keys can be archived",
        ),
        pytest.param(
            r"""class Pending:
    def __init__(self): self.calls=0
    def block_until_ready(self): self.calls+=1
class Node:
    def __init__(self,children): self.children=children
jax.tree_util.register_pytree_node(Node,lambda n:(n.children,None),lambda _,c:Node(c))
pending=Pending()
node=Node([])
node.children=[node,pending,pending,7]
env['node']=node
v=pp.bake(root/'v','node',envir=env)
result=[v is v.children[0],pending.calls,v.children[3]]
""",
            [True, 1, 7],
            id="S-custom-pytree: A registered pytree containing a cycle and pending output",
        ),
        pytest.param(
            r"""pp.r_uniform.set_state(None)
v=pp.bake(root/'v','_rng.uniform(size=4)',seed=32765883,envir=env)
result=[len(v),pp.r_uniform.get_state() is not None]""",
            [4, True],
            id="A-bake-ambient-bootstrap: bake: seeded call with no ambient RNG state",
        ),
        pytest.param(
            r"""pp.r_uniform.set_state(None)
pp.stew(root/'v','v=_rng.uniform(size=4)',seed=32765883,envir=env)
result=[len(env['v']),pp.r_uniform.get_state() is not None]""",
            [4, True],
            id="A-stew-ambient-bootstrap: stew: seeded call with no ambient RNG state",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','_rng.uniform(size=1)',seed=0,envir=env)
result=(v*2**32).astype(np.uint64).tolist()""",
            {
                "uniform_words": 1,
                "sha256": "9f8c8bae7cc3edb0a9b0c4aec59609c3ec056694f837b8d37da2a0992b5741f1",
            },
            id="C-uniform-0-1: All 1 R MT uniform words, seed 0",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','_rng.uniform(size=625)',seed=0,envir=env)
result=(v*2**32).astype(np.uint64).tolist()""",
            {
                "uniform_words": 625,
                "sha256": "ac0c73eb283c38f46719c79c933daad37b90cf7e77017be7e807b41a74f2af40",
            },
            id="C-uniform-0-625: All 625 R MT uniform words, seed 0",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','_rng.uniform(size=2000)',seed=0,envir=env)
result=(v*2**32).astype(np.uint64).tolist()""",
            {
                "uniform_words": 2000,
                "sha256": "b86bd3c173c6a9b99c2ff4a83cd40a13b1eaf45e511febe1d94c51f66c943881",
            },
            id="C-uniform-0-2000: All 2000 R MT uniform words, seed 0",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','_rng.uniform(size=1)',seed=1,envir=env)
result=(v*2**32).astype(np.uint64).tolist()""",
            {
                "uniform_words": 1,
                "sha256": "8b53afb0d526b588e7be51c4982d33d9d287f7ca1e26a3e06398074e16607535",
            },
            id="C-uniform-1-1: All 1 R MT uniform words, seed 1",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','_rng.uniform(size=625)',seed=1,envir=env)
result=(v*2**32).astype(np.uint64).tolist()""",
            {
                "uniform_words": 625,
                "sha256": "a94ceb8e551784d72f4086f915737b3a3ebcde40f4a00f9c1c219183a3e25760",
            },
            id="C-uniform-1-625: All 625 R MT uniform words, seed 1",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','_rng.uniform(size=2000)',seed=1,envir=env)
result=(v*2**32).astype(np.uint64).tolist()""",
            {
                "uniform_words": 2000,
                "sha256": "9e3134d82344fc6b8b51da1367814e7a9f24703152decbf3aae55699b90c7c61",
            },
            id="C-uniform-1-2000: All 2000 R MT uniform words, seed 1",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','_rng.uniform(size=1)',seed=7,envir=env)
result=(v*2**32).astype(np.uint64).tolist()""",
            {
                "uniform_words": 1,
                "sha256": "9cb1d1c4d8cc74cba18fb877618ce575a22d13ca027791689de13c16ddbfe1bb",
            },
            id="C-uniform-7-1: All 1 R MT uniform words, seed 7",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','_rng.uniform(size=625)',seed=7,envir=env)
result=(v*2**32).astype(np.uint64).tolist()""",
            {
                "uniform_words": 625,
                "sha256": "eaa76b140a21d0d46a6a6e9d880b91eb111521d2796cdbe5fa4cb0e9461e3bc9",
            },
            id="C-uniform-7-625: All 625 R MT uniform words, seed 7",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','_rng.uniform(size=2000)',seed=7,envir=env)
result=(v*2**32).astype(np.uint64).tolist()""",
            {
                "uniform_words": 2000,
                "sha256": "bba902cad1d2c1bb34523d6c46e295ce1820e3c822bf29668dafedf45ec1f6c7",
            },
            id="C-uniform-7-2000: All 2000 R MT uniform words, seed 7",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','_rng.uniform(size=1)',seed=-1,envir=env)
result=(v*2**32).astype(np.uint64).tolist()""",
            {
                "uniform_words": 1,
                "sha256": "bbf8dc4607a3338f0a60f4cbdc21c8c0985fb983f6142048b8a468ed33a4a94c",
            },
            id="C-uniform--1-1: All 1 R MT uniform words, seed -1",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','_rng.uniform(size=625)',seed=-1,envir=env)
result=(v*2**32).astype(np.uint64).tolist()""",
            {
                "uniform_words": 625,
                "sha256": "02bd9a0c71ac51ef6f7dde41a83560ea66bf2582f3434f24899fbc284e531a9d",
            },
            id="C-uniform--1-625: All 625 R MT uniform words, seed -1",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','_rng.uniform(size=2000)',seed=-1,envir=env)
result=(v*2**32).astype(np.uint64).tolist()""",
            {
                "uniform_words": 2000,
                "sha256": "d33b5e7d988feb8a2a5fa5243d248e3af2ffe31d408227b61ea76faf93dfe980",
            },
            id="C-uniform--1-2000: All 2000 R MT uniform words, seed -1",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','_rng.uniform(size=1)',seed=5499,envir=env)
result=(v*2**32).astype(np.uint64).tolist()""",
            {
                "uniform_words": 1,
                "sha256": "3e8e23a82c039efa3b8c9e3ddde85e99eef17ba2c45736dde927568035028e8a",
            },
            id="C-uniform-5499-1: All 1 R MT uniform words, seed 5499",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','_rng.uniform(size=625)',seed=5499,envir=env)
result=(v*2**32).astype(np.uint64).tolist()""",
            {
                "uniform_words": 625,
                "sha256": "97f54be2f1076b8f656de6728662f3bdcbf1d9f021774af2a5b6eeabbb7b460c",
            },
            id="C-uniform-5499-625: All 625 R MT uniform words, seed 5499",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','_rng.uniform(size=2000)',seed=5499,envir=env)
result=(v*2**32).astype(np.uint64).tolist()""",
            {
                "uniform_words": 2000,
                "sha256": "509dadd7225596e1191255ec5ccc79d32a9652c8558205710d58e67da07353b2",
            },
            id="C-uniform-5499-2000: All 2000 R MT uniform words, seed 5499",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','_rng.uniform(size=1)',seed=2147483647,envir=env)
result=(v*2**32).astype(np.uint64).tolist()""",
            {
                "uniform_words": 1,
                "sha256": "ba3bc904875699986b38033ca31440f328b9e430e2e0861d621599f40215feaf",
            },
            id="C-uniform-2147483647-1: All 1 R MT uniform words, seed 2147483647",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','_rng.uniform(size=625)',seed=2147483647,envir=env)
result=(v*2**32).astype(np.uint64).tolist()""",
            {
                "uniform_words": 625,
                "sha256": "5c9b464af8785fc5cc6e316749bb86f228579ace3aa0e95e13937ae5e3dd3103",
            },
            id="C-uniform-2147483647-625: All 625 R MT uniform words, seed 2147483647",
        ),
        pytest.param(
            r"""v=pp.bake(root/'v','_rng.uniform(size=2000)',seed=2147483647,envir=env)
result=(v*2**32).astype(np.uint64).tolist()""",
            {
                "uniform_words": 2000,
                "sha256": "ea262d30ebdf94abecf3a7bf77ddb4e9765ec0c5ecac9401c21e0f3d9451fa30",
            },
            id="C-uniform-2147483647-2000: All 2000 R MT uniform words, seed 2147483647",
        ),
        pytest.param(
            r"""words=np.zeros(624,dtype=np.uint32)
words[-1]=1
pp.r_uniform.set_state(('MT19937',words,1,0,0.0))
result=[pp.r_uniform.uniform()==float.fromhex('0x1.00000000fffffp-33'),pp.r_uniform.uniform()>0]""",
            [True, True],
            id="C-uniform-zero-word: R endpoint correction when the MT engine produces a zero word",
        ),
        pytest.param(
            r"""pp.r_uniform.seed(7)
a=pp.r_uniform.uniform(size=8)
pp.r_uniform.seed(7)
b=pp.freeze('_rng.uniform(size=8)',seed=[],compatibility='R',envir=env)
result=bool(np.array_equal(a,b))""",
            True,
            id="C-seed-empty: Seed coercion produces the expected stream: empty",
        ),
        pytest.param(
            r"""pp.r_uniform.seed(7)
a=pp.r_uniform.uniform(size=8)
pp.r_uniform.seed(7)
b=pp.freeze('_rng.uniform(size=8)',seed=[7,99],compatibility='R',envir=env)
result=bool(np.array_equal(a,b))""",
            True,
            id="C-seed-vector: Seed coercion produces the expected stream: vector",
        ),
        pytest.param(
            r"""pp.r_uniform.seed(7)
a=pp.r_uniform.uniform(size=8)
pp.r_uniform.seed(7)
b=pp.freeze('_rng.uniform(size=8)',seed=7.9,compatibility='R',envir=env)
result=bool(np.array_equal(a,b))""",
            True,
            id="C-seed-float: Seed coercion produces the expected stream: float",
        ),
        pytest.param(
            r"""pp.r_uniform.seed(7)
a=pp.r_uniform.uniform(size=8)
pp.r_uniform.seed(7)
b=pp.freeze('_rng.uniform(size=8)',seed='7',compatibility='R',envir=env)
result=bool(np.array_equal(a,b))""",
            True,
            id="C-seed-string: Seed coercion produces the expected stream: string",
        ),
        pytest.param(
            r"""pp.r_uniform.seed(99)
a=pp.r_uniform.uniform(size=4)
b=pp.freeze('_rng.uniform(size=3)',seed=7,compatibility='R',envir=env)
d=pp.r_uniform.uniform(size=4)
pp.r_uniform.seed(99)
all_values=pp.r_uniform.uniform(size=8)
result=[bool(np.array_equal(np.concatenate([a,d]),all_values)),bool(np.array_equal(b,pp.freeze('_rng.uniform(size=3)',seed=7,compatibility='R',envir=env)))]""",
            [True, True],
            id="C-freeze-state: R-compatible uniform ambient sequence resumes after freeze",
        ),
        pytest.param(
            r"""pp.r_uniform.seed(99)
try:
    pp.freeze('_rng.uniform()\nraise RuntimeError()',seed=7,compatibility='R',envir=env)
except RuntimeError:
    pass
a=pp.r_uniform.uniform(size=3)
pp.r_uniform.seed(7)
pp.r_uniform.uniform()
result=bool(np.array_equal(a,pp.r_uniform.uniform(size=3)))""",
            True,
            id="C-freeze-error: Failure leaves R ambient generator at the failed seeded stream",
        ),
        pytest.param(
            r"""v=pp.freeze('x=5',compatibility='R',envir=env)
result=[v,env['x']]""",
            [5, 5],
            id="C-freeze-assignment: Freeze returns assignments and mutates its explicit workspace",
        ),
        pytest.param(
            r"""env['calls']=[]
env['__recipe_assignment_value__']=99
def compute():
    env['calls'].append(1)
    return 5
env['compute']=compute
v=pp.bake(root/'v','x=y=compute()',envir=env)
result=[v,env['x'],env['y'],len(env['calls']),env['__recipe_assignment_value__']]
assert not any(k.startswith('__recipe_assignment_value___') for k in env)""",
            [5, 5, 5, 1, 99],
            id="C-assignment-once: A chained final assignment evaluates its RHS exactly once",
        ),
        pytest.param(
            r"""env['x']=[0]
v=pp.bake(root/'v','x[0]=5',envir=env)
result=[v,env['x'][0]]""",
            [5, 5],
            id="C-assignment-subscript: A subscript assignment returns the RHS and mutates the target",
        ),
        pytest.param(
            r"""env['_ingredients']='keep'
v=pp.bake(root/'v','3',envir=env,info=True,timing=True)
result=[len(v.ingredients)==5,v.system_time is not None,env['_ingredients']]""",
            [True, True, "keep"],
            id="C-metadata-isolation: Bake attributes do not overwrite existing caller metadata",
        ),
        pytest.param(
            r"""env['counter']=[0]
code='counter[0]+=1\ncounter[0]'
result=[pp.bake(root/'v',code,envir=env,**k) for k in [{},{},{'kind':'Mersenne-Twister'},{'kind':'Mersenne-Twister','normal_kind':'Inversion'}]]""",
            [1, 1, 2, 3],
            id="C-kind-invalidation: Explicit kind and normal-kind changes invalidate ingredients",
        ),
        pytest.param(
            r"""direct=jax.random.uniform(jax.random.key(17),(32,))
cached=pp.bake(root/'v','jax.random.uniform(_key,(32,))',seed=17,compatibility='native',envir=env)
result=bool(np.array_equal(direct,cached))""",
            True,
            id="C-native-JAX: Native seeded cache preserves its runtime's existing generator",
        ),
        pytest.param(
            r"""pp.stew(root/'v',"_rng={'value':7}",compatibility='native',envir=env)
result=pickle.loads((root/'v').read_bytes())['objects']['_rng']['value']""",
            7,
            id="C-native-hidden-rng: Native stew still archives a user's hidden _rng binding",
        ),
    ],
)
def test_archive_r_contract(source, expected, tmp_path, archive_rng_state):
    workspace: dict[str, Any] = dict(
        pp=_RArchiveAPI(),
        np=np,
        jax=jax,
        root=tmp_path,
        env={"np": np, "jax": jax},
        pickle=pickle,
        warnings=warnings,
        cloudpickle=cloudpickle,
        mock=mock,
        Path=Path,
        subprocess=subprocess,
        sys=sys,
        os=os,
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        exec(compile(source, "<R archive contract>", "exec"), workspace, workspace)
    actual = _archive_native(workspace["result"])
    if isinstance(expected, dict) and "uniform_words" in expected:
        assert len(actual) == expected["uniform_words"]
        assert all(
            isinstance(v, (int, float)) and 0 < v < 2**32 and int(v) == v
            for v in actual
        )
        encoded = json.dumps([int(v) for v in actual], separators=(",", ":")).encode()
        assert sha256(encoded).hexdigest() == expected["sha256"]
    else:
        assert _archive_equal(actual, expected)


_UPSTREAM_ARCHIVE_IDS = [f"U{i:02d}" for i in range(1, 40)]


@pytest.fixture(scope="module")
def upstream_archive_results(tmp_path_factory):
    root = tmp_path_factory.mktemp("upstream-archives")
    numpy_state = np.random.get_state()
    r_state = pp.r_uniform.get_state()
    try:
        with pp.archive_directory(root):
            return _observe_upstream_archives(root)[0]
    finally:
        np.random.set_state(numpy_state)
        pp.r_uniform.set_state(r_state)


@pytest.mark.parametrize("case_id", _UPSTREAM_ARCHIVE_IDS)
def test_upstream_archive_predicate(case_id, upstream_archive_results):
    expected = [True, True] if case_id in ("U37", "U38") else [True]
    assert upstream_archive_results[case_id] == expected


def _observe_upstream_archives(root):
    out = {}
    env: dict[str, Any] = {"np": np}
    last_metadata = None

    def check(id, value):
        out[id] = [bool(v) for v in value] if isinstance(value, list) else [bool(value)]

    def freeze(source, **kwargs):
        return pp.freeze(source, compatibility="R", envir=env, **kwargs)

    def bake(name, source, **kwargs):
        nonlocal last_metadata
        result = pp.bake(name, source, compatibility="R", envir=env, **kwargs)
        last_metadata = result if isinstance(result, pp.ArchiveValue) else None
        return result.value if isinstance(result, pp.ArchiveValue) else result

    def stew(name, source, **kwargs):
        return pp.stew(name, source, compatibility="R", envir=env, **kwargs)

    np.random.seed(5499)
    w1 = np.random.uniform(size=2)
    w4 = freeze("np.random.uniform(size=5)", seed=[499586, 588686, 39995866])
    w2 = np.random.uniform(size=2)
    w5 = freeze("np.random.uniform(size=5)", seed=499586)
    np.random.seed(5499)
    w3 = np.random.uniform(size=4)
    check("U01", np.array_equal(np.concatenate([w1, w2]), w3))
    check("U02", np.array_equal(w4, w5))

    np.random.seed(32765883)
    x1 = bake("bake1", "np.random.uniform(size=4)", timing=False)
    x2 = bake("bake2", "np.random.uniform(size=4)", seed=32765883, timing=False)
    x3 = bake("bake1", "np.random.uniform(size=4)", timing=False)
    x3a = bake("bake1", "  np.random.uniform(size=4)", timing=False)
    pp.r_uniform.set_state(None)
    bake("bake3", "np.random.uniform(size=4)", seed=59566)
    x5 = bake("bake1", "np.random.uniform(size=5)")
    x6 = bake("bake1", "np.random.uniform(size=5)")
    env["x1"] = x1
    code = "x1+np.random.uniform(size=1)"
    x7 = bake("bake4", code)
    x8 = bake("bake4", code, dependson=x1)
    x9 = bake("bake4", code, dependson=x1)
    x10 = bake("bake4", code, dependson=[x1, x6])
    env["c"] = lambda x: x + 5
    bake("bake4", code, dependson=[x1, env["c"]])
    for dependency in ("[c,x1,x13]", "[c,x1,x13,x14]"):
        try:
            env["x12"] = bake("bake4", code, dependson=eval(dependency, env, env))
        except NameError:
            pass
    bake("bake4", code, seed=233, dependson=x1, info=True)
    assert last_metadata is not None and last_metadata.ingredients is not None
    ingredients = last_metadata.ingredients
    for id, value in zip(
        _UPSTREAM_ARCHIVE_IDS[2:13],
        [
            np.array_equal(x1, x2),
            np.array_equal(x1, x3),
            np.array_equal(x3, x3a),
            np.array_equal(x5, x6),
            not np.array_equal(x3, x5),
            not np.array_equal(x7, x8),
            np.array_equal(x8, x9),
            not np.array_equal(x9, x10),
            "x12" not in env,
            len(ingredients) == 5,
            pickle.loads(ingredients["seed"][-1]) == 233,
        ],
        strict=True,
    ):
        check(id, value)

    (root / "raw").write_bytes(pickle.dumps(x1))
    try:
        bake("raw", "np.random.uniform(size=4)")
        raw_rejected = False
    except ValueError:
        raw_rejected = True

    np.random.seed(113848)
    stew("stew1", "y1=np.random.uniform(size=4)")
    stew("stew2", "y2=np.random.uniform(size=4)", seed=113848)
    y3 = env["y1"].copy()
    names = stew("stew1", "y1=np.random.uniform(size=4)")
    pp.r_uniform.set_state(None)
    stew("stew3", "y4=np.random.uniform(size=4)", seed=59566)
    y5 = env["y1"] + env["y2"]
    stew("stew3", "y6=y1+y2", dependson=[env["y1"], env["y2"]])
    check("U14", np.array_equal(env["y1"], env["y2"]))
    check("U15", np.array_equal(env["y1"], y3))
    check("U16", np.array_equal(y5, env["y6"]))
    env["y1"] = 0
    stew("stew3", "y6=y1+y2", dependson=[env["y1"], env["y2"]])
    check("U17", np.array_equal(env["y2"], env["y6"]))
    check("U18", "_ingredients" not in env)
    check("U19", "_system_time" in env)
    stew("stew3", "y6=y1+y2", info=True)
    check("U20", np.array_equal(env["y2"], env["y6"]))
    check("U21", isinstance(env["_ingredients"], dict))
    check("U22", len(env["_ingredients"]) == 5)

    model = pp.models.sir(times=np.array([0.1, 0.2, 0.5]), key=jax.random.key(0))
    expected = model.simulate(key=jax.random.key(1347484107))
    env["model"] = model
    frozen = freeze("model.simulate(key=_key)", seed=1347484107)
    check("U23", all(a.equals(b) for a, b in zip(expected, frozen, strict=True)))
    check("U24", freeze("np.random.normal(size=5)\nNone", seed=3494995) is None)
    with warnings.catch_warnings(record=True):
        value = bake("bake4", "np.random.normal(size=5)\nNone", seed=3494995)
    check("U25", isinstance(value, list))
    pp.r_uniform.set_state(None)
    bake("b99", "np.random.uniform(size=4)", seed=32765883)
    pp.r_uniform.set_state(None)
    empty_names = stew("s99", "np.random.uniform(size=4)", seed=32765883)
    pp.r_uniform.set_state(None)
    freeze("np.random.uniform(size=4)", seed=32765883)
    freeze("np.random.uniform(size=4)")

    original = bake("old_bake", "np.random.uniform(size=5)", info=True)
    record = pickle.loads((root / "old_bake").read_bytes())
    record.pop("ingredients")
    (root / "old_bake").write_bytes(pickle.dumps(record))
    bake("old_bake", "np.random.uniform(size=5)")
    loaded = bake("old_bake", "np.random.uniform(size=5)")
    check("U26", np.array_equal(original, loaded))
    for file, ids, seed in [
        ("old_stew", ("U27", "U28"), None),
        ("old_seeded", ("U29", "U30"), 99),
    ]:
        source = "x=33\ny=np.random.uniform(size=5)"
        stew(file, source, seed=seed)
        old_x, old_y = env["x"], env["y"].copy()
        (root / file).write_bytes(pickle.dumps({"x": env["x"], "y": env["y"]}))
        stew(file, source)
        stew(file, source, seed=seed)
        check(ids[0], env["x"] == old_x)
        check(ids[1], np.array_equal(env["y"], old_y))

    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        a = bake("results/bob/mary/v", "np.random.normal(size=5)")
        b = bake("results/bob/mary/v", "np.random.normal(size=5)")
        c = bake("mary/v", "np.random.normal(size=5)", dir=root / "results/bob")
        with pp.archive_directory(root / "results"):
            d = bake("bob/mary/v", "np.random.normal(size=5)")
            e = bake("results/bob/mary/v", "np.random.normal(size=5)")
        f = bake("results/results/bob/mary/v", "np.random.normal(size=5)")
    messages = buffer.getvalue().splitlines()
    for id, value in zip(
        _UPSTREAM_ARCHIVE_IDS[30:36],
        [
            np.array_equal(a, b),
            np.array_equal(a, c),
            np.array_equal(a, d),
            not np.array_equal(a, e),
            np.array_equal(e, f),
            len(messages) == 2,
        ],
        strict=True,
    ):
        check(id, value)
    check("U37", [m.startswith("NOTE: creating archive directory") for m in messages])
    check("U38", ["results/bob/mary" in m for m in messages])
    check("U39", len(messages) >= 2 and "results/results/bob/mary" in messages[1])
    assert set(out) == set(_UPSTREAM_ARCHIVE_IDS)
    return out, {
        "raw_archive_rejected": raw_rejected,
        "stew_names": names,
        "empty_stew_names": empty_names,
        "ambient_rng_mapping": "NumPy ambient-seed predicates; R runif vectors tested separately",
        "seed_metadata_mapping": "Typed Python seed payload; compare decoded value",
        "simulation_mapping": "Complete native SIR X/Y DataFrame identity; not R/Python trajectory equality",
        "simulation_shapes": [list(x.shape) for x in expected],
    }
