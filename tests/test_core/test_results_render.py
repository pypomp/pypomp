"""Accessor edge cases and result rendering."""

import numpy as np
import pytest
import xarray as xr

import pypomp.core.results.result as result_mod
from pypomp.core.parameters import PompParameters
from pypomp.core.results import (
    Result,
    build_panel_mif_result,
    build_panel_pfilter_result,
    build_pfilter_result,
    build_pmcmc_result,
)
from tests.helpers.results import (
    KEY,
    pomp_mif_result,
    pomp_pfilter_result,
)


# =====================================================================
# 5. Result: accessor/diagnostic/equality/merge edge cases.
# =====================================================================
def test_getattr_raises_on_unknown_name():
    res = pomp_pfilter_result()
    with pytest.raises(AttributeError, match="no attribute 'totally_bogus'"):
        _ = res.totally_bogus


def test_n_chains_zero_when_payload_none():
    res = pomp_pfilter_result()
    res.__dict__["payload"] = None
    assert res.n_chains == 0


def test_acceptance_rate_no_accepts_var():
    # pfilter results carry no "accepts" payload variable.
    res = pomp_pfilter_result()
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
    r_pfilter = pomp_pfilter_result()
    r_mif = pomp_mif_result()
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
        Result.merge(pomp_pfilter_result(), pomp_mif_result())


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
