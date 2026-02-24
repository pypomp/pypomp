import xarray as xr
import jax.numpy as jnp
import jax
import pandas as pd
import pypomp as pp
from copy import deepcopy


def check_mif_result(result, panel, J, M, a, rw_sd, theta_orig):
    """Helper to verify common mif result attributes."""
    assert result.method == "mif"
    assert hasattr(result, "shared_traces")
    assert hasattr(result, "unit_traces")
    assert hasattr(result, "logLiks")

    theta_list1, theta_list2 = result.theta.to_list(), theta_orig.to_list()
    assert len(theta_list1) == len(theta_list2)
    for d1, d2 in zip(theta_list1, theta_list2):
        for k in ["shared", "unit_specific"]:
            v1, v2 = d1.get(k), d2.get(k)
            if v1 is None or v2 is None:
                assert v1 is v2
            else:
                pd.testing.assert_frame_equal(v1, v2)

    assert result.J == J
    assert result.M == M
    assert result.a == a
    assert result.rw_sd == rw_sd
    assert isinstance(result.shared_traces, xr.DataArray)
    assert isinstance(result.unit_traces, xr.DataArray)
    assert isinstance(result.logLiks, xr.DataArray)
    assert "iteration" in result.shared_traces.dims
    assert "variable" in result.shared_traces.dims
    assert "iteration" in result.unit_traces.dims
    assert "variable" in result.unit_traces.dims
    assert "unit" in result.unit_traces.dims
    assert result.shared_traces.shape[1] == M + 1
    assert result.unit_traces.shape[1] == M + 1
    assert set(result.logLiks.coords["unit"].values) == set(
        ["shared"] + list(panel.unit_objects.keys())
    )


def test_mif(measles_panel_setup_some_shared):
    panel, rw_sd, key = measles_panel_setup_some_shared
    J, M, a = 2, 2, 0.5
    theta_orig = deepcopy(panel.theta)
    panel.mif(J=J, rw_sd=rw_sd, M=M, a=a, key=key)

    check_mif_result(panel.results_history[-1], panel, J, M, a, rw_sd, theta_orig)


def test_mif_parameter_order_consistency(measles_panel_setup_some_shared):
    """
    Test that MIF produces consistent results regardless of parameter order in parameter dataframes.
    """
    panel, rw_sd, key = measles_panel_setup_some_shared
    J = 2
    M = 3
    a = 0.5
    panel.theta = panel.theta * 2

    original_theta = deepcopy(panel.theta)
    reordered_theta = deepcopy(panel.theta)
    reordered_theta.theta = list(reversed(reordered_theta.theta))
    
    for t_dict in reordered_theta.theta:
        if t_dict["shared"] is not None:
            t_dict["shared"] = t_dict["shared"].iloc[::-1]
        if t_dict["unit_specific"] is not None:
            t_dict["unit_specific"] = t_dict["unit_specific"].iloc[::-1, ::-1]

    panel.mif(
        J=J,
        rw_sd=rw_sd,
        M=M,
        a=a,
        key=key,
        theta=original_theta,
    )
    result_original = panel.results_history[-1]

    panel.results_history.clear()
    panel.mif(
        J=J,
        rw_sd=rw_sd,
        M=M,
        a=a,
        key=key,
        theta=reordered_theta,
    )
    result_reordered = panel.results_history[-1]

    traces_orig = result_original.shared_traces
    traces_reordered = result_reordered.shared_traces

    if traces_orig is not None and traces_reordered is not None:
        assert jnp.allclose(
            traces_orig.values, traces_reordered.values, equal_nan=True
        ), (
            f"Shared traces differed after reordering parameter order:\n"
            f"original: {traces_orig.values}\n"
            f"reordered: {traces_reordered.values}"
        )

    unit_traces_orig = result_original.unit_traces
    unit_traces_reordered = result_reordered.unit_traces

    assert jnp.allclose(
        unit_traces_orig.values, unit_traces_reordered.values, equal_nan=True
    ), (
        f"Unit traces differed after reordering parameter order:\n"
        f"original shape: {unit_traces_orig.shape}\n"
        f"reordered shape: {unit_traces_reordered.shape}\n"
        f"original values: {unit_traces_orig.values}\n"
        f"reordered values: {unit_traces_reordered.values}"
    )

    logliks_orig = result_original.logLiks
    logliks_reordered = result_reordered.logLiks

    nan_mask_orig = jnp.isnan(logliks_orig.values)
    nan_mask_reordered = jnp.isnan(logliks_reordered.values)

    assert jnp.array_equal(nan_mask_orig, nan_mask_reordered), (
        f"NaN positions differed after reordering parameter columns:\n"
        f"original NaN mask: {nan_mask_orig}\n"
        f"reordered NaN mask: {nan_mask_reordered}"
    )

    if not jnp.all(nan_mask_orig):
        non_nan_mask = ~nan_mask_orig
        assert jnp.allclose(
            logliks_orig.values[non_nan_mask], logliks_reordered.values[non_nan_mask]
        ), (
            f"Log-likelihoods differed after reordering parameter columns:\n"
            f"original: {logliks_orig.values}\n"
            f"reordered: {logliks_reordered.values}"
        )


def test_mif_shared_vs_unit_specific_single_unit_consistency(measles_panel_setup_pomps_module, measles_rw_sd):
    """
    Test that MIF produces equivalent results for a single-unit panel whether
    parameters are marked as shared or unit-specific.
    """
    london, _, AK_mles = measles_panel_setup_pomps_module
    
    # Force London to have a different canonical parameter order from the panel
    # by re-initializing its parameters with reversed key order
    london_params_orig = AK_mles["London"].to_dict()
    reversed_london_params = {k: london_params_orig[k] for k in reversed(list(london_params_orig.keys()))}
    london.theta = reversed_london_params
    
    # Define some parameters to toggle between shared and unit-specific
    toggled_params = ["gamma", "cohort"]
    
    # Use original order for Panel DataFrames to ensure mismatch with London
    london_params = london_params_orig
    shared_df = pd.DataFrame(
        {p: [london_params[p]] for p in toggled_params},
        index=pd.Index(toggled_params),
        columns=pd.Index(["shared"])
    )
    specific_params = [p for p in london_params if p not in toggled_params]
    specific_df = pd.DataFrame(
        {p: [london_params[p]] for p in specific_params},
        index=pd.Index(specific_params),
        columns=pd.Index(["London"])
    )
    
    panel_shared = pp.PanelPomp(
        Pomp_dict={"London": london},
        theta={"shared": shared_df, "unit_specific": specific_df}
    )
    
    # 2. Setup Panel with toggled parameters as UNIT-SPECIFIC
    all_specific_df = pd.DataFrame(
        {p: [london_params[p]] for p in london_params},
        index=pd.Index(list(london_params.keys())),
        columns=pd.Index(["London"])
    )
    
    panel_specific = pp.PanelPomp(
        Pomp_dict={"London": london},
        theta={"shared": None, "unit_specific": all_specific_df}
    )
    
    J, M, a = 2, 3, 0.5
    key = jax.random.key(42)
    
    # Run MIF on both
    panel_shared.mif(J=J, M=M, rw_sd=measles_rw_sd, a=a, key=key)
    res_shared = panel_shared.results_history[-1]
    
    panel_specific.mif(J=J, M=M, rw_sd=measles_rw_sd, a=a, key=key)
    res_specific = panel_specific.results_history[-1]
    
    # Verify log-likelihoods match
    assert jnp.allclose(res_shared.logLiks.values, res_specific.logLiks.values), (
        f"Log-likelihoods differed:\n"
        f"shared: {res_shared.logLiks.values}\n"
        f"specific: {res_specific.logLiks.values}"
    )
    
    # Verify traces match for toggled parameters
    # In res_shared, they are in shared_traces
    # In res_specific, they are in unit_traces
    
    for p in toggled_params:
        trace_shared = res_shared.shared_traces.sel(variable=p).values
        trace_specific = res_specific.unit_traces.sel(variable=p, unit="London").values
        
        assert jnp.allclose(trace_shared, trace_specific, equal_nan=True), (
            f"Traces for parameter '{p}' differed:\n"
            f"shared_traces version: {trace_shared}\n"
            f"unit_traces version: {trace_specific}"
        )
