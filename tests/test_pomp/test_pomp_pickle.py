"""Pickling and __setstate__ back-compatibility for Pomp."""

import cloudpickle
import jax
import jax.numpy as jnp
import pandas as pd
import pytest

import pypomp as pp
from tests.helpers.assertions import pickle_roundtrip
from tests.helpers.dummy import (
    dummy_dmeas,
    dummy_rinit,
    dummy_rmeas,
    dummy_rproc,
)


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
