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

from pypomp.core.learning_rate import LearningRate
from pypomp.core.optimizer import Adam
from pypomp.core.parameters import PanelParameters, PompParameters
from pypomp.core.results import (
    Result,
    ResultsHistory,
    build_mif_result,
    build_panel_mif_result,
    build_panel_pfilter_result,
    build_panel_train_result,
    build_pfilter_result,
    build_pmcmc_result,
    build_train_result,
)
import pypomp.core.results.result as result_mod
from pypomp.core.rw_sigma import RWSigma

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


# =====================================================================
# 5. Result: accessor/diagnostic/equality/merge edge cases.
# =====================================================================
def test_getattr_raises_on_unknown_name():
    res = _pomp_pfilter()
    with pytest.raises(AttributeError, match="no attribute 'totally_bogus'"):
        _ = res.totally_bogus


def test_n_chains_zero_when_payload_none():
    res = _pomp_pfilter()
    res.__dict__["payload"] = None
    assert res.n_chains == 0


def test_acceptance_rate_no_accepts_var():
    # pfilter results carry no "accepts" payload variable.
    res = _pomp_pfilter()
    rate = res.acceptance_rate
    assert isinstance(rate, np.ndarray)
    assert rate.shape == (0,)


def test_acceptance_rate_zero_denominator():
    theta = PompParameters({"param1": 1.0})
    traces = xr.DataArray(
        [[[0.0]]],
        dims=["theta_idx", "iteration", "variable"],
        coords={"theta_idx": [0], "iteration": [0], "variable": ["logLik"]},
    )
    res = build_pmcmc_result(
        key=KEY,
        execution_time=1.0,
        theta=theta,
        traces=traces,
        M=0,  # falsy denominator
        J=10,
        accepts=np.array([3]),
    )
    rate = res.acceptance_rate
    assert np.array_equal(rate, np.zeros_like(np.array([3.0])))


def test_eq_mismatch_branches():
    r_pfilter = _pomp_pfilter()
    r_mif = _pomp_mif()
    # method/kind/panel differ
    assert r_pfilter != r_mif

    # config differs (different thresh), method/kind/panel/theta/payload same shape
    r_a = build_pfilter_result(
        key=KEY,
        execution_time=1.0,
        theta=PompParameters({"param1": 1.0, "param2": 2.0}),
        logLiks=xr.DataArray([[1.5, 2.5]], dims=["theta_idx", "rep"]),
        J=1000,
        reps=2,
        thresh=0.5,
    )
    r_b = build_pfilter_result(
        key=KEY,
        execution_time=1.0,
        theta=PompParameters({"param1": 1.0, "param2": 2.0}),
        logLiks=xr.DataArray([[1.5, 2.5]], dims=["theta_idx", "rep"]),
        J=1000,
        reps=2,
        thresh=0.9,  # different
    )
    assert r_a != r_b

    # theta differs, everything else identical
    r_c = build_pfilter_result(
        key=KEY,
        execution_time=1.0,
        theta=PompParameters({"param1": 1.0, "param2": 2.0}),
        logLiks=xr.DataArray([[1.5, 2.5]], dims=["theta_idx", "rep"]),
        J=1000,
        reps=2,
        thresh=0.5,
    )
    r_d = build_pfilter_result(
        key=KEY,
        execution_time=1.0,
        theta=PompParameters({"param1": 9.0, "param2": 2.0}),  # different
        logLiks=xr.DataArray([[1.5, 2.5]], dims=["theta_idx", "rep"]),
        J=1000,
        reps=2,
        thresh=0.5,
    )
    assert r_c != r_d


def test_merge_rejects_mismatched_method_kind_panel():
    with pytest.raises(ValueError, match="must share method/kind/panel"):
        Result.merge(_pomp_pfilter(), _pomp_mif())


def test_merge_falls_back_to_first_payload_when_no_theta_idx():
    a = Result(
        method="custom",
        kind="table",
        panel=False,
        execution_time=1.0,
        key=KEY,
        config={},
        payload=xr.Dataset(),
    )
    b = Result(
        method="custom",
        kind="table",
        panel=False,
        execution_time=2.0,
        key=KEY,
        config={},
        payload=xr.Dataset(),
    )
    merged = Result.merge(a, b)
    assert merged.execution_time == 2.0
    assert merged.payload.equals(a.payload)
    assert merged.theta is None  # both thetas were None


def test_values_equal_array_branch():
    assert result_mod._values_equal(np.array([1, 2]), np.array([1, 2])) is True
    assert result_mod._values_equal(np.array([1, 2]), np.array([1, 3])) is False


def test_first_config_mismatch_differing_key_sets():
    assert result_mod._first_config_mismatch({"a": 1, "b": 2}, {"a": 1}) == "b"
    assert result_mod._first_config_mismatch({"a": 1}, {"a": 1}) is None


def test_merge_theta_branches():
    # No present (non-None) thetas -> None.
    assert result_mod._merge_theta([None, None]) is None

    # First present theta is a list -> extend semantics, None entries treated
    # as empty.
    merged = result_mod._merge_theta([["a", "b"], None, ["c"]])
    assert merged == ["a", "b", "c"]

    # First present theta has no `.merge` classmethod -> fall back to it as-is.
    merged = result_mod._merge_theta(["only_first_kept", "ignored"])
    assert merged == "only_first_kept"


# =====================================================================
# 6. Rendering (render.py) edge cases.
# =====================================================================
def test_print_summary_skips_unlabeled_config_keys(capsys):
    """A config key with no entry in _SUMMARY_LABELS is silently skipped."""
    res = build_pfilter_result(
        key=KEY,
        execution_time=1.0,
        theta=PompParameters({"param1": 1.0}),
        logLiks=xr.DataArray([1.5], dims=["theta_idx"]),
        J=10,
        reps=1,
        thresh=0.5,
    )
    res.config["not_a_known_label"] = "surprise"
    res.print_summary()
    out = capsys.readouterr().out
    assert "surprise" not in out
    assert "Method: pfilter" in out


def test_pomp_table_result_with_no_theta_renders_empty():
    res = build_pfilter_result(
        key=KEY,
        execution_time=1.0,
        theta=None,
        logLiks=xr.DataArray([1.5], dims=["theta_idx"]),
        J=10,
        reps=1,
        thresh=0.5,
    )
    assert res.to_dataframe().empty
    assert res.traces().empty


def test_pomp_trace_result_missing_traces_var_renders_empty():
    res = Result(
        method="mif",
        kind="trace",
        panel=False,
        execution_time=1.0,
        key=KEY,
        theta=PompParameters({"param1": 1.0}),
        config={"J": 10, "M": 1, "thresh": 0.5, "n_monitors": 1},
        payload=xr.Dataset(),  # no "traces" variable
    )
    assert res.to_dataframe().empty
    assert res.traces().empty


def test_panel_trace_result_missing_unit_traces_renders_empty():
    """Payload has shared_traces but not unit_traces: `_var` returns None for
    the missing variable and both to_dataframe/traces short-circuit empty."""
    shared_traces = xr.DataArray(
        [[[1.0]]],
        dims=["theta_idx", "iteration", "variable"],
        coords={"theta_idx": [0], "iteration": [0], "variable": ["logLik"]},
    )
    res = Result(
        method="mif",
        kind="trace",
        panel=True,
        execution_time=1.0,
        key=KEY,
        theta=None,
        config={"J": 10, "M": 1, "thresh": 0.5, "n_monitors": 1, "block": False},
        payload=xr.Dataset({"shared_traces": shared_traces}),
    )
    assert res.to_dataframe().empty
    assert res.traces().empty


def test_attach_panel_params_none_theta_leaves_df_unchanged():
    logLiks = xr.DataArray(
        [[[1.0, 2.0], [3.0, 4.0]]],
        dims=["theta_idx", "unit", "rep"],
        coords={"theta_idx": [0], "unit": ["u1", "u2"], "rep": [0, 1]},
    )
    res = build_panel_pfilter_result(
        key=KEY,
        execution_time=1.0,
        theta=None,
        logLiks=logLiks,
        J=10,
        reps=2,
        thresh=0.5,
    )
    df = res.to_dataframe()
    assert not df.empty
    # No theta -> no shared/unit parameter columns attached.
    assert set(df.columns) == {
        "theta_idx",
        "shared logLik",
        "shared logLik se",
        "unit",
        "unit logLik",
        "unit logLik se",
    }


def test_panel_trace_traces_merges_shared_params_onto_unit_rows():
    shared_traces = xr.DataArray(
        [[[1.0, 0.25]]],
        dims=["theta_idx", "iteration", "variable"],
        coords={"theta_idx": [0], "iteration": [0], "variable": ["logLik", "beta"]},
    )
    unit_traces = xr.DataArray(
        [[[[2.0]]]],
        dims=["theta_idx", "iteration", "unit", "variable"],
        coords={
            "theta_idx": [0],
            "iteration": [0],
            "unit": ["u1"],
            "variable": ["unitLogLik"],
        },
    )
    res = build_panel_mif_result(
        key=KEY,
        execution_time=1.0,
        theta=None,
        shared_traces=shared_traces,
        unit_traces=unit_traces,
        logLiks=xr.DataArray(
            [[np.nan, np.nan]],
            dims=["theta_idx", "unit"],
            coords={"theta_idx": [0], "unit": ["shared", "u1"]},
        ),
        J=10,
        M=1,
        rw_sd=None,
        thresh=0.5,
        n_monitors=1,
        block=False,
    )
    df = res.traces()
    # "beta" (a shared, non-standard column) should be merged onto the u1 row.
    u1_row = df[df["unit"] == "u1"].iloc[0]
    assert u1_row["beta"] == 0.25


def test_mcmc_to_df_and_traces_empty_when_traces_missing():
    res = Result(
        method="pmcmc",
        kind="trace",
        panel=False,
        execution_time=1.0,
        key=KEY,
        theta=PompParameters({"param1": 1.0}),
        config={"M": 5, "J": 10},
        payload=xr.Dataset(),  # no "traces" variable
    )
    assert res.to_dataframe().empty
    assert res.traces().empty


def test_mcmc_to_df_ignore_nan_drops_rows():
    traces = xr.DataArray(
        [[[1.0, np.nan], [1.0, 2.0]]],
        dims=["theta_idx", "iteration", "variable"],
        coords={
            "theta_idx": [0],
            "iteration": [0, 1],
            "variable": ["logLik", "param1"],
        },
    )
    res = build_pmcmc_result(
        key=KEY,
        execution_time=1.0,
        theta=PompParameters({"param1": 1.0}),
        traces=traces,
        M=2,
        J=10,
        accepts=np.array([1]),
    )
    df_with_nan = res.to_dataframe(ignore_nan=False)
    assert df_with_nan["param1"].isna().any()

    df_no_nan = res.to_dataframe(ignore_nan=True)
    assert len(df_no_nan) < len(df_with_nan)
    assert not df_no_nan["param1"].isna().any()


def test_theta_count_list_branch():
    """`_theta_count` special-cases a plain-list `theta` (len, not
    `.num_replicates()`).

    A bare-list `theta` isn't otherwise renderable end-to-end (the
    to_dataframe paths for both single-unit and panel results assume
    `theta` exposes `.get_param_names()` / `.num_replicates()`), so this
    exercises the private helper directly rather than through
    `print_summary`.
    """
    from pypomp.core.results import render as render_mod

    thetas = [PompParameters({"param1": 1.0}), PompParameters({"param1": 2.0})]
    assert render_mod._theta_count(thetas) == 2
    assert render_mod._theta_count([]) == 0
