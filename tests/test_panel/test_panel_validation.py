from copy import deepcopy
import jax
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import pypomp as pp


def test_validate_unit_objects_not_dict(lg_panel_setup_some_shared):
    panel, _, _ = lg_panel_setup_some_shared
    panel.unit_objects = "not a dict"
    with pytest.raises(TypeError, match="unit_objects must be a dictionary"):
        panel._validate_unit_objects()


def test_validate_unit_objects_not_pomp(lg_panel_setup_some_shared):
    panel, _, _ = lg_panel_setup_some_shared
    panel.unit_objects = {"unit1": "not a pomp object"}
    with pytest.raises(
        TypeError,
        match="Every element of unit_objects must be an instance of the class Pomp",
    ):
        panel._validate_unit_objects()


def test_validate_unit_objects_mismatched_t0(lg_panel_setup_some_shared):
    panel, _, _ = lg_panel_setup_some_shared
    panel.unit_objects["unit2"].t0 = 999.0
    with pytest.raises(ValueError, match="All units must have the same t0"):
        panel._validate_unit_objects()


def test_validate_unit_objects_mismatched_dt(lg_panel_setup_some_shared):
    panel, _, _ = lg_panel_setup_some_shared
    panel.unit_objects["unit2"]._dt_array_extended = (
        panel.unit_objects["unit2"]._dt_array_extended + 1.0
    )
    with pytest.raises(
        ValueError, match="All units must have the same _dt_array_extended"
    ):
        panel._validate_unit_objects()


def test_validate_unit_objects_mismatched_nstep(lg_panel_setup_some_shared):
    panel, _, _ = lg_panel_setup_some_shared
    panel.unit_objects["unit2"]._nstep_array = (
        panel.unit_objects["unit2"]._nstep_array + 1
    )
    with pytest.raises(ValueError, match="All units must have the same _nstep_array"):
        panel._validate_unit_objects()


def test_validate_unit_objects_mismatched_ys_index(lg_panel_setup_some_shared):
    panel, _, _ = lg_panel_setup_some_shared
    new_ys = panel.unit_objects["unit2"].ys.copy()
    new_ys.index = pd.Index([9.0] * len(new_ys))
    panel.unit_objects["unit2"].ys = new_ys
    with pytest.raises(ValueError, match="All units must have the same ys index"):
        panel._validate_unit_objects()


def test_validate_unit_objects_mismatched_ys_columns(lg_panel_setup_some_shared):
    panel, _, _ = lg_panel_setup_some_shared
    new_ys = panel.unit_objects["unit2"].ys.copy()
    new_ys.columns = pd.Index(["other_col1", "other_col2"])
    panel.unit_objects["unit2"].ys = new_ys
    with pytest.raises(ValueError, match="All units must have the same ys columns"):
        panel._validate_unit_objects()


def test_validate_params_and_units_mismatched_unit_names(
    lg_panel_setup_some_shared,
):
    panel, _, _ = lg_panel_setup_some_shared
    panel.unit_objects = {
        "mismatched_unit": panel.unit_objects["unit1"],
        "unit2": panel.unit_objects["unit2"],
    }
    with pytest.raises(
        ValueError,
        match="The unit names in the unit_objects dictionary must match the unit names in the theta object",
    ):
        panel._validate_params_and_units()


def test_validate_params_and_units_mismatched_param_names(
    lg_panel_setup_some_shared,
):
    panel, _, _ = lg_panel_setup_some_shared
    panel.canonical_param_names = panel.canonical_param_names + [
        "extra_nonexistent_param"
    ]
    with pytest.raises(
        ValueError,
        match="The canonical parameter names must match the canonical parameter names in the theta object",
    ):
        panel._validate_params_and_units()


def test_validate_params_and_units_unit_mismatch_across_units(
    lg_panel_setup_some_shared,
):
    panel, _, _ = lg_panel_setup_some_shared
    panel.unit_objects["unit2"].canonical_param_names = ["mismatched_param"]
    with pytest.raises(
        ValueError,
        match="The canonical parameter names in the unit objects must match the canonical parameter names in the first unit for all units",
    ):
        panel._validate_params_and_units()


def test_validate_params_and_units_panel_vs_units_mismatch(
    lg_panel_setup_some_shared,
):
    panel, _, _ = lg_panel_setup_some_shared
    panel.unit_objects["unit1"].canonical_param_names = ["dummy_param"]
    panel.unit_objects["unit2"].canonical_param_names = ["dummy_param"]
    with pytest.raises(
        ValueError,
        match="The canonical parameter names must match the canonical parameter names in the unit objects",
    ):
        panel._validate_params_and_units()


def test_panel_pomp_eq_type_mismatch(lg_panel_setup_some_shared):
    panel, _, _ = lg_panel_setup_some_shared
    assert panel != 42
    assert panel != "not a panel"


def test_panel_pomp_eq_canonical_param_mismatch(lg_panel_setup_some_shared):
    panel1, _, _ = lg_panel_setup_some_shared
    panel2 = deepcopy(panel1)
    panel2.canonical_param_names = list(reversed(panel2.canonical_param_names))
    assert panel1 != panel2


def test_panel_pomp_eq_canonical_shared_param_mismatch(
    lg_panel_setup_some_shared,
):
    panel1, _, _ = lg_panel_setup_some_shared
    panel2 = deepcopy(panel1)
    panel2.canonical_shared_param_names = list(
        reversed(panel2.canonical_shared_param_names)
    )
    assert panel1 != panel2


def test_panel_pomp_eq_canonical_unit_param_mismatch(
    lg_panel_setup_some_shared,
):
    panel1, _, _ = lg_panel_setup_some_shared
    panel2 = deepcopy(panel1)
    panel2.canonical_unit_param_names = list(
        reversed(panel2.canonical_unit_param_names)
    )
    assert panel1 != panel2


def test_panel_pomp_eq_theta_mismatch(lg_panel_setup_some_shared):
    panel1, _, _ = lg_panel_setup_some_shared
    panel2 = deepcopy(panel1)
    theta_params = panel2.theta.params()
    theta_params[0]["shared"].iloc[0, 0] += 1.0
    panel2.theta.set_params(theta_params)
    assert panel1 != panel2


def test_panel_pomp_eq_unit_names_mismatch(lg_panel_setup_some_shared):
    panel1, _, _ = lg_panel_setup_some_shared
    panel2 = deepcopy(panel1)
    panel2.unit_objects = {
        "unit2": panel2.unit_objects["unit2"],
        "unit1": panel2.unit_objects["unit1"],
    }
    assert panel1 != panel2


def test_panel_pomp_eq_unit_objects_mismatch(lg_panel_setup_some_shared):
    panel1, _, _ = lg_panel_setup_some_shared
    panel2 = deepcopy(panel1)
    panel2.unit_objects["unit1"].t0 += 1.0
    assert panel1 != panel2


def test_panel_pomp_eq_results_history_mismatch(lg_panel_setup_some_shared):
    panel1, _, _ = lg_panel_setup_some_shared
    panel2 = deepcopy(panel1)
    from pypomp.core.results import PanelPompPFilterResult

    dummy_result = PanelPompPFilterResult(
        method="pfilter",
        execution_time=0.1,
        key=jax.random.key(1),
        theta=panel2.theta,
        logLiks=xr.DataArray([1.0]),
        J=2,
        reps=1,
        thresh=0.0,
    )
    panel2.results_history.add(dummy_result)
    assert panel1 != panel2


def test_panel_pomp_eq_fresh_key_presence_mismatch(lg_panel_setup_some_shared):
    panel1, _, _ = lg_panel_setup_some_shared
    panel2 = deepcopy(panel1)
    panel1.fresh_key = jax.random.key(1)
    panel2.fresh_key = None
    assert panel1 != panel2


def test_panel_pomp_eq_fresh_key_value_mismatch(lg_panel_setup_some_shared):
    panel1, _, _ = lg_panel_setup_some_shared
    panel2 = deepcopy(panel1)
    panel1.fresh_key = jax.random.key(1)
    panel2.fresh_key = jax.random.key(2)
    assert panel1 != panel2


def test_panel_pomp_merge_empty():
    with pytest.raises(
        ValueError, match="At least one PanelPomp object must be provided."
    ):
        pp.PanelPomp.merge()


def test_panel_pomp_merge_invalid_type(lg_panel_setup_some_shared):
    panel, _, _ = lg_panel_setup_some_shared
    with pytest.raises(
        TypeError, match="All merged objects must be of type PanelPomp."
    ):
        pp.PanelPomp.merge(panel, "not a panel pomp")  # type: ignore


def test_panel_pomp_merge_mismatched_params(lg_panel_setup_some_shared):
    panel1, _, _ = lg_panel_setup_some_shared
    panel2 = deepcopy(panel1)
    panel2.canonical_param_names = panel2.canonical_param_names + ["extra"]
    with pytest.raises(
        ValueError,
        match="All PanelPomp objects must have the same canonical_param_names.",
    ):
        pp.PanelPomp.merge(panel1, panel2)


def test_panel_pomp_merge_mismatched_shared_params(lg_panel_setup_some_shared):
    panel1, _, _ = lg_panel_setup_some_shared
    panel2 = deepcopy(panel1)
    panel2.canonical_shared_param_names = panel2.canonical_shared_param_names + [
        "extra"
    ]
    with pytest.raises(
        ValueError,
        match="All PanelPomp objects must have the same canonical_shared_param_names.",
    ):
        pp.PanelPomp.merge(panel1, panel2)


def test_panel_pomp_merge_mismatched_unit_params(lg_panel_setup_some_shared):
    panel1, _, _ = lg_panel_setup_some_shared
    panel2 = deepcopy(panel1)
    panel2.canonical_unit_param_names = panel2.canonical_unit_param_names + ["extra"]
    with pytest.raises(
        ValueError,
        match="All PanelPomp objects must have the same canonical_unit_param_names.",
    ):
        pp.PanelPomp.merge(panel1, panel2)


def test_panel_pomp_merge_mismatched_units(lg_panel_setup_some_shared):
    panel1, _, _ = lg_panel_setup_some_shared
    panel2 = deepcopy(panel1)
    panel2.unit_objects = {
        "other_unit": panel2.unit_objects["unit1"],
        "unit2": panel2.unit_objects["unit2"],
    }
    with pytest.raises(
        ValueError, match="All PanelPomp objects must have the same unit names."
    ):
        pp.PanelPomp.merge(panel1, panel2)


def test_update_fresh_key_both_none(lg_panel_setup_some_shared):
    panel, _, _ = lg_panel_setup_some_shared
    panel.fresh_key = None
    with pytest.raises(
        ValueError,
        match="Both the key argument and the fresh_key attribute are None",
    ):
        panel._update_fresh_key(key=None)


def test_get_covars_per_unit_partial_covariates(lg_panel_setup_some_shared):
    panel, _, _ = lg_panel_setup_some_shared
    panel.unit_objects["unit1"]._covars_extended = np.array([1, 2, 3])
    panel.unit_objects["unit2"]._covars_extended = None
    with pytest.raises(
        NotImplementedError,
        match="Some units have covariates, but not all units have covariates",
    ):
        panel._get_covars_per_unit(list(panel.unit_objects.keys()))


def test_prepare_theta_input_both_none(lg_panel_setup_some_shared):
    panel, _, _ = lg_panel_setup_some_shared
    panel._theta = None  # type: ignore
    with pytest.raises(
        ValueError, match="theta must be provided or self.theta must exist"
    ):
        panel._prepare_theta_input(theta=None)


def test_prepare_theta_input_invalid_type(lg_panel_setup_some_shared):
    panel, _, _ = lg_panel_setup_some_shared
    with pytest.raises(
        TypeError, match="theta must be a PanelParameters instance or None"
    ):
        panel._prepare_theta_input(theta="not a PanelParameters object")  # type: ignore


def test_get_unit_param_permutation_missing_param(lg_panel_setup_some_shared):
    panel, _, _ = lg_panel_setup_some_shared
    panel.unit_objects["unit1"].canonical_param_names = panel.unit_objects[
        "unit1"
    ].canonical_param_names + ["nonexistent"]
    with pytest.raises(ValueError, match="not in the panel's parameter list"):
        panel._get_unit_param_permutation("unit1")


def test_pfilter_dmeas_none(lg_panel_setup_some_shared):
    panel, _, key = lg_panel_setup_some_shared
    panel.unit_objects["unit1"].dmeas = None
    with pytest.raises(ValueError, match="dmeas cannot be None in PanelPomp units"):
        panel.pfilter(J=2, key=key)


def test_mif_invalid_args(lg_panel_setup_some_shared):
    panel, rw_sd, key = lg_panel_setup_some_shared
    with pytest.raises(ValueError, match="J and M must be greater than 0."):
        panel.mif(J=0, M=2, rw_sd=rw_sd, key=key)
    with pytest.raises(ValueError, match="J and M must be greater than 0."):
        panel.mif(J=2, M=0, rw_sd=rw_sd, key=key)


def test_mif_dmeas_none(lg_panel_setup_some_shared):
    panel, rw_sd, key = lg_panel_setup_some_shared
    panel.unit_objects["unit1"].dmeas = None
    with pytest.raises(ValueError, match="dmeas cannot be None in PanelPomp units"):
        panel.mif(J=2, M=2, rw_sd=rw_sd, key=key)


def test_train_invalid_args(lg_panel_setup_some_shared):
    panel, _, key = lg_panel_setup_some_shared
    eta = pp.LearningRate({n: 0.01 for n in panel.canonical_param_names})
    with pytest.raises(ValueError, match="J and M must be greater than 0."):
        panel.train(J=0, M=2, eta=eta, key=key)
    with pytest.raises(ValueError, match="J and M must be greater than 0."):
        panel.train(J=2, M=0, eta=eta, key=key)


def test_train_dmeas_none(lg_panel_setup_some_shared):
    panel, _, key = lg_panel_setup_some_shared
    eta = pp.LearningRate({n: 0.01 for n in panel.canonical_param_names})
    panel.unit_objects["unit1"].dmeas = None
    with pytest.raises(ValueError, match="dmeas cannot be None in PanelPomp units"):
        panel.train(J=2, M=2, eta=eta, key=key)


def test_train_invalid_eta(lg_panel_setup_some_shared):
    panel, _, key = lg_panel_setup_some_shared
    with pytest.raises(TypeError, match="eta must be a LearningRate object"):
        panel.train(J=2, M=2, eta="not a LearningRate", key=key)  # type: ignore


def test_mif_traces_both_none(lg_panel_setup_some_shared, monkeypatch):
    panel, rw_sd, key = lg_panel_setup_some_shared
    import pypomp.functional as F

    monkeypatch.setattr(
        F,
        "panel_mif",
        lambda *args, **kwargs: (None, None, None, None),
    )
    with pytest.raises(ValueError, match="Both shared_traces and unit_traces are None"):
        panel.mif(J=2, M=2, rw_sd=rw_sd, key=key)


def test_train_traces_both_none(lg_panel_setup_some_shared, monkeypatch):
    panel, _, key = lg_panel_setup_some_shared
    eta = pp.LearningRate({n: 0.01 for n in panel.canonical_param_names})
    import pypomp.functional as F

    monkeypatch.setattr(
        F,
        "panel_train",
        lambda *args, **kwargs: (None, None, None),
    )
    with pytest.raises(ValueError, match="Both shared_traces and unit_traces are None"):
        panel.train(J=2, M=2, eta=eta, key=key)


def test_plot_traces_no_shared_rows(lg_panel_setup_some_shared, monkeypatch):
    panel, _, _ = lg_panel_setup_some_shared
    dummy_traces = pd.DataFrame(
        {
            "theta_idx": [0],
            "unit": ["unit1"],
            "iteration": [0],
            "method": ["mif"],
            "logLik": [1.0],
        }
    )
    monkeypatch.setattr(panel, "traces", lambda: dummy_traces)
    with pytest.warns(UserWarning, match="No shared rows to plot."):
        res = panel.plot_traces(which="shared", show=False)
    assert res is None


def test_plot_simulations_invalid_theta(lg_panel_setup_some_shared):
    panel, _, key = lg_panel_setup_some_shared
    with pytest.raises(TypeError, match="theta must be a PanelParameters instance"):
        panel.plot_simulations(key=key, theta="not a PanelParameters", show=False)  # type: ignore
