"""Result container behavior and single-unit result builders."""

from typing import Any, cast

import jax
import jax.numpy as jnp
import pandas as pd
import pytest
import xarray as xr

from pypomp.core.learning_rate import LearningRate
from pypomp.core.optimizer import Adam
from pypomp.core.parameters import PompParameters
from pypomp.core.results import (
    Result,
    build_pfilter_result,
    build_train_result,
)
from pypomp.core.rw_sigma import RWSigma
from tests.helpers.assertions import pickle_roundtrip
from tests.helpers.results import (
    KEY,
    pomp_mif_result,
    pomp_pfilter_result,
)


# =====================================================================
def test_merge_guards():
    with pytest.raises(ValueError, match="At least one"):
        Result.merge()

    r1 = pomp_pfilter_result()
    with pytest.raises(TypeError, match="must be of type"):
        Result.merge(r1, cast(Any, "not_a_result"))

    # config mismatch (different J)
    r_diff = build_pfilter_result(
        key=KEY,
        execution_time=1.0,
        theta=PompParameters({"param1": 1.0, "param2": 2.0}),
        logLiks=xr.DataArray([[1.5, 2.5]], dims=["theta_idx", "rep"]),
        J=999,
        reps=2,
        thresh=0.5,
    )
    with pytest.raises(ValueError, match="same J"):
        type(r1).merge(r1, r_diff)


def test_result_equality_and_pickle():
    r1 = pomp_pfilter_result()
    r2 = pomp_pfilter_result()
    assert r1 == r2  # equal payload/config/theta/key; timestamp ignored
    assert r1 != "not_a_result"

    # different key -> not equal
    r3 = build_pfilter_result(
        key=jax.random.key(99),
        execution_time=1.5,
        theta=PompParameters({"param1": 1.0, "param2": 2.0}),
        logLiks=xr.DataArray([[1.5, 2.5]], dims=["theta_idx", "rep"]),
        J=1000,
        reps=2,
        thresh=0.5,
    )
    assert r1 != r3

    # different payload -> not equal
    r4 = pomp_pfilter_result()
    r4.payload["logLiks"].values[:] = 0.0
    assert r1 != r4

    unpickled = pickle_roundtrip(r1)
    assert unpickled == r1
    assert jnp.array_equal(
        jax.random.key_data(unpickled.key), jax.random.key_data(r1.key)
    )
    assert type(unpickled) is type(r1)


def test_trace_result_has_empty_cll_ess():
    res = pomp_mif_result()
    assert res.CLL().empty
    assert res.ESS().empty
    # optional pfilter payload vars surface as None on a trace result
    assert res.CLL_da is None
    assert res.ESS_da is None


def test_print_summary_cooling_variants(capsys):
    sig = {"a": 0.1}

    res = pomp_mif_result(rw_sd=RWSigma(sig))
    res.print_summary()
    assert "Cooling fraction (a): 0.5" in capsys.readouterr().out

    res = pomp_mif_result(rw_sd=RWSigma(sig).hyperbolic_cooling(0.2))
    res.print_summary()
    assert "Cooling rate (s): 0.2" in capsys.readouterr().out

    res = pomp_mif_result(rw_sd=RWSigma(sig).cosine_cooling(0.1, 100))
    res.print_summary()
    out = capsys.readouterr().out
    assert "Cosine min cooling (c): 0.1" in out
    assert "Cosine duration (M): 100" in out

    def dummy_cool_fn(nt, m, ntimes):
        return 1.0

    res = pomp_mif_result(rw_sd=RWSigma(sig).custom_cooling(dummy_cool_fn))
    res.print_summary()
    assert "Cooling function: dummy_cool_fn" in capsys.readouterr().out


# =====================================================================
# 2. Single-unit result rendering + merge + accessors.
# =====================================================================
def test_pomp_pfilter_result():
    res = pomp_pfilter_result()
    assert res.method == "pfilter"
    assert res.J == 1000 and res.reps == 2 and res.thresh == 0.5
    assert isinstance(res.logLiks, xr.DataArray)
    assert res.CLL_da is not None and res.ESS_da is not None

    df = res.to_dataframe()
    assert len(df) == 1
    assert {"logLik", "se", "param1"} <= set(df.columns)
    assert not pd.isna(df.loc[0, "se"])  # reps > 1

    assert len(res.CLL(average=False)) == 4  # 1 * 2 rep * 2 time
    assert len(res.CLL(average=True)) == 2
    assert len(res.ESS(average=False)) == 4
    assert len(res.ESS(average=True)) == 2

    df_tr = res.traces()
    assert len(df_tr) == 1
    assert df_tr.loc[0, "iteration"] == 0
    assert df_tr.loc[0, "param1"] == 1.0

    merged = type(res).merge(res, pomp_pfilter_result(execution_time=2.5))
    assert merged.execution_time == 2.5
    assert merged.J == 1000
    assert merged.logLiks.sizes["theta_idx"] == 2


def test_pomp_pfilter_single_parameter_set_1d():
    theta = PompParameters({"param1": 1.0, "param2": 2.0})
    res = build_pfilter_result(
        key=KEY,
        execution_time=1.5,
        theta=theta,
        logLiks=xr.DataArray([1.5], dims=["theta_idx"]),
        J=1000,
        reps=1,
        thresh=0.5,
    )
    df = res.to_dataframe()
    assert len(df) == 1
    assert df.loc[0, "logLik"] == 1.5
    assert pd.isna(df.loc[0, "se"])
    assert df.loc[0, "param1"] == 1.0
    assert res.traces().loc[0, "logLik"] == 1.5


def test_pomp_mif_result():
    res = pomp_mif_result()
    assert res.method == "mif"
    assert res.M == 2 and res.J == 100 and res.n_monitors == 5

    df = res.to_dataframe()
    assert len(df) == 1
    assert df.loc[0, "logLik"] == 2.5
    assert df.loc[0, "param1"] == 1.1
    assert pd.isna(df.loc[0, "se"])

    assert res.traces()["se"].isna().all()

    merged = type(res).merge(res, pomp_mif_result(execution_time=1.5))
    assert merged.M == 2
    assert merged.traces_da.sizes["theta_idx"] == 2


def test_pomp_train_result():
    theta = PompParameters({"param1": 1.0})
    traces = xr.DataArray(
        [[[1.5, 1.0]]],
        dims=["theta_idx", "iteration", "variable"],
        coords={"theta_idx": [0], "iteration": [0], "variable": ["logLik", "param1"]},
    )
    opt = Adam()
    lr = LearningRate({"param1": 0.01})

    def _make(execution_time):
        return build_train_result(
            key=KEY,
            execution_time=execution_time,
            theta=theta,
            traces=traces,
            optimizer=opt,
            J=500,
            M=1,
            eta=lr,
            alpha=0.95,
            thresh=0.0,
            alpha_cooling=0.99,
        )

    res = _make(1.0)
    assert res.method == "train"
    assert res.optimizer == opt

    merged = type(res).merge(res, _make(2.0))
    assert merged.optimizer == opt
    assert merged.execution_time == 2.0
