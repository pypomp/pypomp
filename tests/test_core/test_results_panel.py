"""Panel result builders and ResultsHistory."""

from typing import Any, cast

import pandas as pd
import pytest
import xarray as xr

from pypomp.core.optimizer import Adam
from pypomp.core.parameters import PanelParameters
from pypomp.core.results import (
    ResultsHistory,
    build_panel_mif_result,
    build_panel_pfilter_result,
    build_panel_train_result,
)
from tests.helpers.results import (
    KEY,
    panel_shared_unit,
    pomp_mif_result,
    pomp_pfilter_result,
)


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
    shared_traces, unit_traces, logLiks = panel_shared_unit()

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
    shared_traces, unit_traces, logLiks = panel_shared_unit()
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

    r1 = pomp_pfilter_result(execution_time=10.0)
    r2 = pomp_mif_result(execution_time=20.0)
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
    h_a = ResultsHistory([pomp_pfilter_result(execution_time=5.0)])
    h_b = ResultsHistory([pomp_pfilter_result(execution_time=12.0)])
    with pytest.raises(ValueError, match="same number of entries"):
        ResultsHistory.merge(h_a, ResultsHistory([]))
    assert len(ResultsHistory.merge()) == 0
    h_merged = ResultsHistory.merge(h_a, h_b)
    assert len(h_merged) == 1
    assert h_merged[0].execution_time == 12.0
    assert h_merged[0].logLiks.sizes["theta_idx"] == 2

    hist.clear()
    assert len(hist) == 0
