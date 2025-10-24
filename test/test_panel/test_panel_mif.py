import xarray as xr
import jax.numpy as jnp


def test_mif(measles_panel_setup_some_shared):
    panel, rw_sd, key = measles_panel_setup_some_shared
    J = 2
    M = 2
    a = 0.5
    shared_orig = panel.shared
    unit_specific_orig = panel.unit_specific
    panel.mif(J=J, rw_sd=rw_sd, M=M, a=a, key=key)
    result = panel.results_history[-1]

    assert result["method"] == "mif"
    assert "shared_traces" in result
    assert "unit_traces" in result
    assert "logLiks" in result
    assert result["shared"] is shared_orig
    assert result["unit_specific"] is unit_specific_orig
    assert result["J"] == J
    assert result["M"] == M
    assert result["a"] == a
    assert result["rw_sd"] == rw_sd
    assert isinstance(result["shared_traces"], xr.DataArray)
    assert isinstance(result["unit_traces"], xr.DataArray)
    assert isinstance(result["logLiks"], xr.DataArray)
    assert "iteration" in result["shared_traces"].dims
    assert "variable" in result["shared_traces"].dims
    assert "iteration" in result["unit_traces"].dims
    assert "variable" in result["unit_traces"].dims
    assert "unit" in result["unit_traces"].dims
    assert result["shared_traces"].shape[1] == M + 1
    assert result["unit_traces"].shape[1] == M + 1
    assert set(result["logLiks"].coords["unit"].values) == set(
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
    panel.unit_specific = panel.unit_specific * 2
    panel.shared = panel.shared * 2

    original_unit_specific = [df.copy() for df in panel.unit_specific]
    original_shared = [df.copy() for df in panel.shared]

    reordered_unit_specific = []
    for df in original_unit_specific:
        reordered_index = list(reversed(df.index))
        reordered_df = df.reindex(reordered_index)
        reordered_unit_specific.append(reordered_df)
    reordered_shared = []
    for df in original_shared:
        reordered_index = list(reversed(df.index))
        reordered_df = df.reindex(reordered_index)
        reordered_shared.append(reordered_df)

    panel.mif(
        J=J,
        rw_sd=rw_sd,
        M=M,
        a=a,
        key=key,
        unit_specific=original_unit_specific,
        shared=original_shared,
    )
    result_original = panel.results_history[-1]

    panel.results_history.clear()
    panel.mif(
        J=J,
        rw_sd=rw_sd,
        M=M,
        a=a,
        key=key,
        unit_specific=reordered_unit_specific,
        shared=reordered_shared,
    )
    result_reordered = panel.results_history[-1]

    traces_orig = result_original["shared_traces"]
    traces_reordered = result_reordered["shared_traces"]

    if traces_orig is not None and traces_reordered is not None:
        assert jnp.allclose(
            traces_orig.values, traces_reordered.values, equal_nan=True
        ), (
            f"Shared traces differed after reordering parameter order:\n"
            f"original: {traces_orig.values}\n"
            f"reordered: {traces_reordered.values}"
        )

    unit_traces_orig = result_original["unit_traces"]
    unit_traces_reordered = result_reordered["unit_traces"]

    assert jnp.allclose(
        unit_traces_orig.values, unit_traces_reordered.values, equal_nan=True
    ), (
        f"Unit traces differed after reordering parameter order:\n"
        f"original shape: {unit_traces_orig.shape}\n"
        f"reordered shape: {unit_traces_reordered.shape}\n"
        f"original values: {unit_traces_orig.values}\n"
        f"reordered values: {unit_traces_reordered.values}"
    )

    logliks_orig = result_original["logLiks"]
    logliks_reordered = result_reordered["logLiks"]

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
