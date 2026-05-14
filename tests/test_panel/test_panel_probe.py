import pandas as pd
import numpy as np


def test_panel_pomp_probe(lg_panel_setup_some_shared):
    panel, rw_sd, key = lg_panel_setup_some_shared

    probes = {"mean_first_col": lambda df: df.iloc[:, 0].mean()}

    nsim = 3
    probe_df = panel.probe(probes=probes, nsim=nsim, key=key)

    assert isinstance(probe_df, pd.DataFrame)
    expected_cols = ["probe", "value", "is_real_data", "theta_idx", "sim", "unit"]
    assert all(col in probe_df.columns for col in expected_cols)

    unit_names = list(panel.unit_objects.keys())
    num_units = len(unit_names)
    num_reps = panel.theta.num_replicates()

    expected_real_rows = num_units * 1
    expected_sim_rows = num_units * num_reps * nsim * 1
    assert len(probe_df) == expected_real_rows + expected_sim_rows

    assert probe_df["is_real_data"].sum() == expected_real_rows

    assert set(probe_df["unit"]) == set(unit_names)

    for unit in unit_names:
        row = probe_df[(probe_df["is_real_data"]) & (probe_df["unit"] == unit)]
        unit_real_val = row["value"].iloc[0]
        expected_val = panel.unit_objects[unit].ys.iloc[:, 0].mean()
        assert np.isclose(unit_real_val, expected_val)


def test_panel_pomp_probe_with_subset_theta(lg_panel_setup_some_shared):
    panel, rw_sd, key = lg_panel_setup_some_shared

    panel.theta = panel.theta.subset([0])

    probes = {"max_first_col": lambda df: df.iloc[:, 0].max()}

    nsim = 2
    probe_df = panel.probe(probes=probes, nsim=nsim, key=key)

    num_units = len(panel.unit_objects)
    assert len(probe_df) == num_units + (num_units * 2)
    assert set(probe_df[~probe_df["is_real_data"]]["theta_idx"]) == {0}
    assert set(probe_df[~probe_df["is_real_data"]]["sim"]) == {0, 1}
