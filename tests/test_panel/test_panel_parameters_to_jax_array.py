import jax
import numpy as np
import pandas as pd
import pytest

import pypomp as pp


def _mixed_panel():
    """Two replicates, two units, mix of shared and unit-specific params."""
    shared_df = pd.DataFrame(
        {"shared": [10.0, 20.0]}, index=pd.Index(["s1", "s2"])
    )
    unit_specific_df = pd.DataFrame(
        {"unit1": [1.0, 2.0, 3.0], "unit2": [4.0, 5.0, 6.0]},
        index=pd.Index(["u1", "u2", "u3"]),
    )
    return (
        pp.PanelParameters(
            theta=[{"shared": shared_df, "unit_specific": unit_specific_df}]
        )
        * 2
    )


def _specific_only_panel():
    unit_specific_df = pd.DataFrame(
        {"unit1": [1.0, 2.0], "unit2": [3.0, 4.0]},
        index=pd.Index(["a", "b"]),
    )
    return pp.PanelParameters(
        theta=[{"shared": None, "unit_specific": unit_specific_df}]
    )


def _shared_only_panel():
    shared_df = pd.DataFrame(
        {"shared": [10.0, 20.0]}, index=pd.Index(["s1", "s2"])
    )
    return pp.PanelParameters(theta=[{"shared": shared_df, "unit_specific": None}])


def test_to_jax_array_mixed_shape_and_values():
    panel = _mixed_panel()
    param_names = ["s1", "u1", "u2", "s2", "u3"]
    arr = panel.to_jax_array(param_names, unit_names=["unit1", "unit2"])

    assert isinstance(arr, jax.Array)
    # (reps, n_units, n_params)
    assert arr.shape == (2, 2, 5)

    # Replicate 0, unit "unit1": s1=10, u1=1, u2=2, s2=20, u3=3
    np.testing.assert_allclose(np.asarray(arr[0, 0]), [10.0, 1.0, 2.0, 20.0, 3.0])
    # Replicate 0, unit "unit2": shared values are the same; unit-specific differ.
    np.testing.assert_allclose(np.asarray(arr[0, 1]), [10.0, 4.0, 5.0, 20.0, 6.0])
    # Replicate 1 mirrors replicate 0 (constructed via `* 2`).
    np.testing.assert_allclose(np.asarray(arr[1]), np.asarray(arr[0]))


def test_to_jax_array_infers_unit_names_from_unit_specific():
    panel = _mixed_panel()
    arr_with = panel.to_jax_array(["u1"], unit_names=["unit1", "unit2"])
    arr_inferred = panel.to_jax_array(["u1"])
    np.testing.assert_allclose(np.asarray(arr_inferred), np.asarray(arr_with))


def test_to_jax_array_specific_only_panel():
    panel = _specific_only_panel()
    arr = panel.to_jax_array(["a", "b"])
    assert arr.shape == (1, 2, 2)
    np.testing.assert_allclose(np.asarray(arr[0, 0]), [1.0, 2.0])
    np.testing.assert_allclose(np.asarray(arr[0, 1]), [3.0, 4.0])


def test_to_jax_array_shared_only_panel_requires_unit_names():
    panel = _shared_only_panel()
    arr = panel.to_jax_array(["s1", "s2"], unit_names=["u1", "u2"])
    assert arr.shape == (1, 2, 2)
    # Shared values should be broadcast across all units.
    np.testing.assert_allclose(np.asarray(arr[0, 0]), [10.0, 20.0])
    np.testing.assert_allclose(np.asarray(arr[0, 1]), [10.0, 20.0])


def test_to_jax_array_shared_only_no_unit_names_raises():
    panel = _shared_only_panel()
    with pytest.raises(ValueError, match="unit_names required"):
        panel.to_jax_array(["s1", "s2"])


def test_to_jax_array_unknown_param_raises():
    panel = _mixed_panel()
    with pytest.raises(KeyError, match="not found"):
        panel.to_jax_array(["s1", "nonexistent"], unit_names=["unit1", "unit2"])


def test_to_jax_array_unknown_unit_raises():
    """Asking for a unit that isn't a column in unit_specific re-raises the
    underlying KeyError (the 'missing' list stays empty because all
    parameter names *are* present)."""
    panel = _mixed_panel()
    with pytest.raises(KeyError):
        panel.to_jax_array(
            ["s1", "u1"], unit_names=["unit1", "unit_does_not_exist"]
        )
