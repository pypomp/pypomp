import xarray as xr
import jax.numpy as jnp
from copy import deepcopy


def test_mif(measles_panel_setup_some_shared):
    panel, rw_sd, key = measles_panel_setup_some_shared
    J = 2
    M = 2
    a = 0.5
    theta_orig = deepcopy(panel.theta)
    panel.mif(J=J, rw_sd=rw_sd, M=M, a=a, key=key)
    result = panel.results_history[-1]

    assert result.method == "mif"
    assert hasattr(result, "shared_traces")
    assert hasattr(result, "unit_traces")
    assert hasattr(result, "logLiks")
    theta_list1, theta_list2 = result.theta.to_list(), theta_orig.to_list()
    assert len(theta_list1) == len(theta_list2) and all(
        (d1.get(k) is None and d2.get(k) is None)
        or (
            d1.get(k) is not None
            and d2.get(k) is not None
            and d1.get(k).equals(d2.get(k))
        )
        for d1, d2 in zip(theta_list1, theta_list2)
        for k in ["shared", "unit_specific"]
    )
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
