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
    """With dt set, the step count is recovered from the pickled step grid."""
    state = base_pomp.__getstate__()
    state["_rproc_func_bytes"] = cloudpickle.dumps(base_pomp.rproc.original_func)
    state["_rproc_dt"] = 0.5
    state["_rproc_nstep"] = 3
    state["_rproc_accumvars"] = ["X"]

    pomp_unpickled = pickle_roundtrip(base_pomp)
    del pomp_unpickled.rproc
    pomp_unpickled.__setstate__(state)

    assert pomp_unpickled.rproc is not None
    assert pomp_unpickled.rproc.nstep == int(base_pomp._nstep_array[0])


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


def _local_model(dt: float = 0.25, scale: float = 1.0) -> pp.Pomp:
    """A model whose functions are local, so cloudpickle stores them by value
    exactly as it does for functions defined in ``__main__`` or a notebook."""

    def rinit(theta_, key, covars, t0):
        return {"X": theta_["x0"]}

    def rproc(X_, theta_, key, covars, t, dt):
        noise = theta_["s"] * jnp.sqrt(dt) * jax.random.normal(key)
        return {"X": X_["X"] - scale * theta_["a"] * X_["X"] * dt + noise}

    def dmeas(Y_, X_, theta_, covars, t):
        return jax.scipy.stats.norm.logpdf(Y_["Y"], X_["X"], 0.5)

    return pp.Pomp(
        ys=pd.DataFrame({"Y": [0.1, -0.2, 0.3, 0.0]}, index=[1.0, 2.0, 3.0, 4.0]),
        theta=pp.PompParameters({"a": 0.5, "s": 0.5, "x0": 0.0}),
        statenames=["X"],
        t0=0.0,
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        dt=dt,
    )


def test_unpickled_dt_model_can_train():
    """A dt-specified model keeps its fixed step count, which train needs."""
    model = pickle_roundtrip(_local_model())
    assert model.rproc.nstep == 4
    eta = pp.LearningRate({"a": 0.01, "s": 0.01, "x0": 0.01})
    model.train(J=10, M=1, eta=eta, key=jax.random.key(0))


def test_unpickled_local_functions_equal_and_merge():
    """Unpickling makes new function objects; the copy must still match."""
    model = _local_model()
    copy = pickle_roundtrip(model)
    assert copy == model
    merged = pp.Pomp.merge(model, copy)
    assert merged.theta.num_replicates() == 2


def test_models_differing_in_closure_are_unequal():
    assert _local_model(scale=1.0) != _local_model(scale=2.0)
    with pytest.raises(ValueError, match="same rinit and rproc"):
        pp.Pomp.merge(_local_model(scale=1.0), _local_model(scale=2.0))
