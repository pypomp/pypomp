"""Tests for the unified :class:`pypomp.core.results.result.Result` container,
its builders, rendering, merge, equality/pickling, and :class:`ResultsHistory`.
"""

import pickle
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pypomp.core.results import (
    Result,
    ResultsHistory,
    build_pfilter_result,
    build_mif_result,
    build_train_result,
    build_panel_pfilter_result,
    build_panel_mif_result,
    build_panel_train_result,
)
from pypomp.core.parameters import PompParameters, PanelParameters
from pypomp.core.rw_sigma import RWSigma
from pypomp.core.learning_rate import LearningRate
from pypomp.core.optimizer import Adam

KEY = jax.random.key(0)


# =====================================================================
# Fixtures / helpers building concrete results via the public builders.
# =====================================================================
def _pomp_pfilter(execution_time=1.5, with_diag=True):
    theta = PompParameters({"param1": 1.0, "param2": 2.0})
    logLiks = xr.DataArray([[1.5, 2.5]], dims=["theta_idx", "rep"])
    cll = ess = None
    if with_diag:
        cll = xr.DataArray(
            [[[0.5, 0.6], [0.7, 0.8]]], dims=["theta_idx", "rep", "time"]
        )
        ess = xr.DataArray(
            [[[10.0, 20.0], [30.0, 40.0]]], dims=["theta_idx", "rep", "time"]
        )
    return build_pfilter_result(
        key=KEY,
        execution_time=execution_time,
        theta=theta,
        logLiks=logLiks,
        J=1000,
        reps=2,
        thresh=0.5,
        CLL=cll,
        ESS=ess,
    )


def _pomp_mif(execution_time=1.0, rw_sd=None):
    theta = PompParameters({"param1": 1.0})
    traces = xr.DataArray(
        [[[1.5, 1.0], [2.5, 1.1]]],
        dims=["theta_idx", "iteration", "variable"],
        coords={
            "theta_idx": [0],
            "iteration": [0, 1],
            "variable": ["logLik", "param1"],
        },
    )
    return build_mif_result(
        key=KEY,
        execution_time=execution_time,
        theta=theta,
        traces=traces,
        J=100,
        M=2,
        rw_sd=rw_sd,
        thresh=0.8,
        n_monitors=5,
    )


def _panel_shared_unit(shared_val=1.0, unit_val=2.0):
    shared_traces = xr.DataArray(
        [[[shared_val]]],
        dims=["theta_idx", "iteration", "variable"],
        coords={"theta_idx": [0], "iteration": [0], "variable": ["logLik"]},
    )
    unit_traces = xr.DataArray(
        [[[[unit_val]]]],
        dims=["theta_idx", "iteration", "unit", "variable"],
        coords={
            "theta_idx": [0],
            "iteration": [0],
            "unit": ["u1"],
            "variable": ["unitLogLik"],
        },
    )
    logLiks = xr.DataArray(
        [[np.nan, np.nan]],
        dims=["theta_idx", "unit"],
        coords={"theta_idx": [0], "unit": ["shared", "u1"]},
    )
    return shared_traces, unit_traces, logLiks


# =====================================================================
# 1. Result: merge guards, equality, pickling.
# =====================================================================
def test_merge_guards():
    with pytest.raises(ValueError, match="At least one"):
        Result.merge()

    r1 = _pomp_pfilter()
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
    r1 = _pomp_pfilter()
    r2 = _pomp_pfilter()
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
    r4 = _pomp_pfilter()
    r4.payload["logLiks"].values[:] = 0.0
    assert r1 != r4

    unpickled = pickle.loads(pickle.dumps(r1))
    assert unpickled == r1
    assert jnp.array_equal(
        jax.random.key_data(unpickled.key), jax.random.key_data(r1.key)
    )
    assert type(unpickled) is type(r1)


def test_n_chains_no_payload():
    r = Result(
        method="pmcmc",
        kind="trace",
        panel=False,
        execution_time=None,
        key=KEY,
        payload=None,  # type: ignore
    )
    assert r.n_chains == 0


def test_acceptance_rate_edge_cases():
    # 1. No "accepts" variable at all (e.g. a plain pfilter result).
    r_pfilter = _pomp_pfilter()
    assert r_pfilter.acceptance_rate.shape == (0,)

    # 2. "accepts" present but config lacks a (truthy) "M" to divide by.
    r_no_M = Result(
        method="pmcmc",
        kind="trace",
        panel=False,
        execution_time=None,
        key=KEY,
        config={},
        payload=xr.Dataset({"accepts": (["theta_idx"], np.array([3, 5]))}),
    )
    rate = r_no_M.acceptance_rate
    assert np.array_equal(rate, np.zeros(2))

    # 3. Normal success path: accepts present and M > 0.
    r_with_M = Result(
        method="pmcmc",
        kind="trace",
        panel=False,
        execution_time=None,
        key=KEY,
        config={"M": 10},
        payload=xr.Dataset({"accepts": (["theta_idx"], np.array([3, 5]))}),
    )
    assert np.allclose(r_with_M.acceptance_rate, np.array([0.3, 0.5]))


def test_eq_theta_mismatch():
    r1 = _pomp_pfilter()
    r2 = _pomp_pfilter()
    r2.theta = PompParameters({"param1": 99.0, "param2": 2.0})
    assert r1 != r2


def test_eq_method_kind_panel_mismatch():
    r_pfilter = _pomp_pfilter()
    r_mif = _pomp_mif()
    assert r_pfilter != r_mif


def test_eq_config_mismatch():
    r1 = _pomp_pfilter()
    r2 = build_pfilter_result(
        key=KEY,
        execution_time=1.5,
        theta=PompParameters({"param1": 1.0, "param2": 2.0}),
        logLiks=xr.DataArray([[1.5, 2.5]], dims=["theta_idx", "rep"]),
        J=999,  # different J -> config mismatch
        reps=2,
        thresh=0.5,
    )
    assert r1 != r2


def test_n_chains_and_attribute_errors_positive_paths():
    r = _pomp_pfilter()
    assert r.n_chains == 1

    with pytest.raises(AttributeError):
        r.__some_nonexistent_dunder__
    with pytest.raises(AttributeError):
        r.totally_made_up_attribute


def test_merge_method_kind_panel_mismatch():
    r_pfilter = _pomp_pfilter()
    r_mif = _pomp_mif()
    assert type(r_pfilter) is type(r_mif)
    with pytest.raises(ValueError, match="share method/kind/panel"):
        type(r_pfilter).merge(r_pfilter, r_mif)


def test_merge_all_empty_payloads():
    empty1 = Result(
        method="pfilter",
        kind="table",
        panel=False,
        execution_time=1.0,
        key=KEY,
        config={"J": 10, "reps": 1, "thresh": 0.0},
        payload=xr.Dataset(),
    )
    empty2 = Result(
        method="pfilter",
        kind="table",
        panel=False,
        execution_time=1.0,
        key=KEY,
        config={"J": 10, "reps": 1, "thresh": 0.0},
        payload=xr.Dataset(),
    )
    merged = Result.merge(empty1, empty2)
    assert merged.payload.equals(empty1.payload)


def test_values_equal_and_config_mismatch_helpers():
    from pypomp.core.results.result import (
        _values_equal,
        _first_config_mismatch,
        _merge_theta,
    )

    assert _values_equal(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert not _values_equal(np.array([1.0, 2.0]), np.array([1.0, 3.0]))

    assert _first_config_mismatch({"a": 1}, {"a": 1, "b": 2}) == "b"

    assert _merge_theta([None, None]) is None
    assert _merge_theta([[1, 2], [3]]) == [1, 2, 3]
    assert _merge_theta([5, 5]) == 5


def test_print_summary_skips_unlabeled_config_key():
    res = _pomp_pfilter()
    res.config["some_unrecognized_key"] = 123
    res.print_summary()  # should not raise, and should just skip the unknown key


def test_empty_table_result_pomp():
    """A table-kind pomp result with no theta yields empty to_dataframe()/traces()."""
    res = Result(
        method="pfilter",
        kind="table",
        panel=False,
        execution_time=None,
        key=KEY,
        theta=None,
        payload=xr.Dataset(
            {"logLiks": xr.DataArray([[1.0]], dims=["theta_idx", "rep"])}
        ),
    )
    assert res.to_dataframe().empty
    assert res.traces().empty


def test_empty_trace_result_pomp():
    """A trace-kind pomp result with no 'traces' variable yields empty output."""
    res = Result(
        method="mif",
        kind="trace",
        panel=False,
        execution_time=None,
        key=KEY,
        payload=xr.Dataset(),
    )
    assert res.to_dataframe().empty
    assert res.traces().empty


def test_empty_trace_result_panel():
    """A trace-kind panel result with no shared/unit traces yields empty output."""
    res = Result(
        method="mif",
        kind="trace",
        panel=True,
        execution_time=None,
        key=KEY,
        payload=xr.Dataset(),
    )
    assert res.to_dataframe().empty
    assert res.traces().empty


def test_empty_mcmc_result():
    """An MCMC-family result with no 'traces' variable yields empty output."""
    res = Result(
        method="pmcmc",
        kind="trace",
        panel=False,
        execution_time=None,
        key=KEY,
        payload=xr.Dataset(),
    )
    assert res.to_dataframe().empty
    assert res.traces().empty


def test_mcmc_to_dataframe_ignore_nan():
    traces = xr.DataArray(
        [[[1.0, np.nan], [2.0, 0.5]]],
        dims=["theta_idx", "iteration", "variable"],
        coords={
            "theta_idx": [0],
            "iteration": [0, 1],
            "variable": ["logLik", "beta"],
        },
    )
    res = Result(
        method="pmcmc",
        kind="trace",
        panel=False,
        execution_time=None,
        key=KEY,
        payload=xr.Dataset({"traces": traces}),
    )
    df_with_nan = res.to_dataframe(ignore_nan=False)
    df_no_nan = res.to_dataframe(ignore_nan=True)
    assert len(df_no_nan) < len(df_with_nan)


def test_theta_count_list():
    from pypomp.core.results.render import _theta_count

    assert _theta_count([1, 2, 3]) == 3


def test_trace_result_has_empty_cll_ess():
    res = _pomp_mif()
    assert res.CLL().empty
    assert res.ESS().empty
    # optional pfilter payload vars surface as None on a trace result
    assert res.CLL_da is None
    assert res.ESS_da is None


def test_print_summary_cooling_variants(capsys):
    sig = {"a": 0.1}

    res = _pomp_mif(rw_sd=RWSigma(sig))
    res.print_summary()
    assert "Cooling fraction (a): 0.5" in capsys.readouterr().out

    res = _pomp_mif(rw_sd=RWSigma(sig).hyperbolic_cooling(0.2))
    res.print_summary()
    assert "Cooling rate (s): 0.2" in capsys.readouterr().out

    res = _pomp_mif(rw_sd=RWSigma(sig).cosine_cooling(0.1, 100))
    res.print_summary()
    out = capsys.readouterr().out
    assert "Cosine min cooling (c): 0.1" in out
    assert "Cosine duration (M): 100" in out

    def dummy_cool_fn(nt, m, ntimes):
        return 1.0

    res = _pomp_mif(rw_sd=RWSigma(sig).custom_cooling(dummy_cool_fn))
    res.print_summary()
    assert "Cooling function: dummy_cool_fn" in capsys.readouterr().out


# =====================================================================
# 2. Single-unit result rendering + merge + accessors.
# =====================================================================
def test_pomp_pfilter_result():
    res = _pomp_pfilter()
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

    merged = type(res).merge(res, _pomp_pfilter(execution_time=2.5))
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
    res = _pomp_mif()
    assert res.method == "mif"
    assert res.M == 2 and res.J == 100 and res.n_monitors == 5

    df = res.to_dataframe()
    assert len(df) == 1
    assert df.loc[0, "logLik"] == 2.5
    assert df.loc[0, "param1"] == 1.1
    assert pd.isna(df.loc[0, "se"])

    assert res.traces()["se"].isna().all()

    merged = type(res).merge(res, _pomp_mif(execution_time=1.5))
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


# =====================================================================
# 3. Panel result rendering + merge + accessors.
# =====================================================================
def test_panel_pfilter_result():
    shared_df = pd.DataFrame({"shared": [1.0]}, index=["s1"])
    unit_df = pd.DataFrame({"u1": [2.0], "u2": [3.0]}, index=["up1"])
    theta = PanelParameters(
        theta=cast(Any, {"shared": shared_df, "unit_specific": unit_df})
    )
    coords_ll = {"theta_idx": [0], "unit": ["u1", "u2"], "rep": [0, 1]}
    logLiks = xr.DataArray(
        [[[1.0, 2.0], [3.0, 4.0]]], dims=["theta_idx", "unit", "rep"], coords=coords_ll
    )
    cll = xr.DataArray(
        [[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]],
        dims=["theta_idx", "unit", "rep", "time"],
        coords={**coords_ll, "time": [0, 1]},
    )
    ess = xr.DataArray(
        [[[[10.0, 11.0], [20.0, 21.0]], [[30.0, 31.0], [40.0, 41.0]]]],
        dims=["theta_idx", "unit", "rep", "time"],
        coords={**coords_ll, "time": [0, 1]},
    )

    def _make(execution_time):
        return build_panel_pfilter_result(
            key=KEY,
            execution_time=execution_time,
            theta=theta,
            logLiks=logLiks,
            J=200,
            reps=2,
            thresh=0.1,
            CLL=cll,
            ESS=ess,
        )

    res = _make(1.5)
    df = res.to_dataframe()
    assert len(df) == 2
    assert {
        "shared logLik",
        "shared logLik se",
        "unit logLik",
        "unit logLik se",
        "s1",
        "up1",
    } <= set(df.columns)
    assert not pd.isna(df["shared logLik se"].iloc[0])

    assert len(res.CLL(average=False)) == 8
    assert len(res.CLL(average=True)) == 4
    assert len(res.ESS(average=False)) == 8
    assert len(res.ESS(average=True)) == 4

    df_tr = res.traces()
    assert len(df_tr) == 3
    assert set(df_tr["unit"]) == {"shared", "u1", "u2"}
    assert not df_tr["se"].isna().any()

    merged = type(res).merge(res, _make(2.0))
    assert merged.execution_time == 2.0
    assert merged.logLiks.sizes["theta_idx"] == 2


def test_panel_mif_result():
    theta = PanelParameters(None)
    shared_traces, unit_traces, logLiks = _panel_shared_unit()

    def _make(execution_time):
        return build_panel_mif_result(
            key=KEY,
            execution_time=execution_time,
            theta=theta,
            shared_traces=shared_traces,
            unit_traces=unit_traces,
            logLiks=logLiks,
            J=50,
            M=1,
            rw_sd=None,
            thresh=0.0,
            n_monitors=1,
            block=True,
        )

    res = _make(1.0)
    assert res.block is True
    # accessors restore public dims
    assert res.unit_traces.dims == ("theta_idx", "iteration", "unit", "variable")
    assert res.logLiks.dims == ("theta_idx", "unit")

    df = res.to_dataframe()
    assert len(df) == 1
    assert df.loc[0, "shared logLik"] == 1.0
    assert df.loc[0, "unit logLik"] == 2.0
    assert pd.isna(df.loc[0, "shared logLik se"])

    assert res.traces()["se"].isna().all()

    merged = type(res).merge(res, _make(3.0))
    assert merged.block is True
    assert merged.execution_time == 3.0
    assert merged.shared_traces.sizes["theta_idx"] == 2
    assert merged.unit_traces.sizes["theta_idx"] == 2
    assert merged.logLiks.sizes["theta_idx"] == 2


def test_panel_train_result():
    theta = PanelParameters(None)
    shared_traces, unit_traces, logLiks = _panel_shared_unit()
    opt = Adam()

    def _make(execution_time):
        return build_panel_train_result(
            key=KEY,
            execution_time=execution_time,
            theta=theta,
            shared_traces=shared_traces,
            unit_traces=unit_traces,
            logLiks=logLiks,
            optimizer=opt,
            J=100,
            M=1,
            eta=None,
            alpha=0.9,
            alpha_cooling=1.0,
        )

    res = _make(1.0)
    assert res.alpha == 0.9
    merged = type(res).merge(res, _make(1.2))
    assert merged.alpha == 0.9
    assert merged.execution_time == 1.2


# =====================================================================
# 4. ResultsHistory.
# =====================================================================
def test_results_history(capsys):
    hist = ResultsHistory()
    assert len(hist) == 0
    assert hist.time().empty
    assert hist.results().empty
    assert hist.CLL().empty
    assert hist.ESS().empty
    assert hist.traces().empty
    with pytest.raises(ValueError, match="History is empty"):
        hist.last()

    r1 = _pomp_pfilter(execution_time=10.0)
    r2 = _pomp_mif(execution_time=20.0)
    hist.append(r1)
    hist.add(r2)
    assert len(hist) == 2
    assert hist.last() == r2
    assert list(hist) == [r1, r2]
    assert hist[0] == r1 and hist[-1] == r2

    sub = hist[0:1]
    assert isinstance(sub, ResultsHistory) and len(sub) == 1 and sub[0] == r1

    df_t = hist.time()
    assert list(df_t["method"]) == ["pfilter", "mif"]
    assert list(df_t["time"]) == [10.0, 20.0]

    hist.print_summary()
    out = capsys.readouterr().out
    assert "Results History:" in out
    assert "[0] PFILTER Result:" in out
    assert "[1] MIF Result:" in out

    assert hist == ResultsHistory([r1, r2])
    assert hist != ResultsHistory([r1])
    assert hist != "not_a_history"

    assert not hist.results(0).empty
    assert not hist.CLL(0).empty  # pfilter has CLL
    assert hist.CLL(1).empty  # mif has none

    # traces across entries are concatenated with a shared iteration axis
    df_tr = hist.traces()
    assert not df_tr.empty
    assert {"theta_idx", "iteration", "method", "logLik"} <= set(df_tr.columns)

    # merge histories: pairwise merge of compatible entries, max execution time
    h_a = ResultsHistory([_pomp_pfilter(execution_time=5.0)])
    h_b = ResultsHistory([_pomp_pfilter(execution_time=12.0)])
    with pytest.raises(ValueError, match="same number of entries"):
        ResultsHistory.merge(h_a, ResultsHistory([]))
    assert len(ResultsHistory.merge()) == 0
    h_merged = ResultsHistory.merge(h_a, h_b)
    assert len(h_merged) == 1
    assert h_merged[0].execution_time == 12.0
    assert h_merged[0].logLiks.sizes["theta_idx"] == 2

    hist.clear()
    assert len(hist) == 0
