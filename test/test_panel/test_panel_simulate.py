import pandas as pd


def test_simulate(measles_panel_setup_some_shared):
    panel, rw_sd, key = measles_panel_setup_some_shared
    X_sim_order = ["unit", "replicate", "sim", "time"] + [
        f"state_{i}" for i in range(0, 6)
    ]
    Y_sim_order = ["unit", "replicate", "sim", "time", "obs_0"]

    X_sims, Y_sims = panel.simulate(key=key)

    assert isinstance(X_sims, pd.DataFrame)
    assert isinstance(Y_sims, pd.DataFrame)
    assert list(X_sims.columns) == X_sim_order
    assert list(Y_sims.columns) == Y_sim_order


def test_simulate_as_pomp(measles_panel_setup_some_shared):
    import pytest
    import pypomp as pp

    panel, rw_sd, key = measles_panel_setup_some_shared

    # Test normal as_pomp
    new_panel = panel.simulate(key=key, as_pomp=True)
    assert hasattr(new_panel, "unit_objects")
    assert len(new_panel.unit_objects) == len(panel.unit_objects)

    # Check that unit objects are also Pomp objects with updated data
    for unit, obj in new_panel.unit_objects.items():
        assert isinstance(obj, pp.Pomp)
        assert obj.ys.shape == panel.unit_objects[unit].ys.shape
        assert not obj.ys.equals(panel.unit_objects[unit].ys)
        assert obj.theta.num_replicates() == 1

    # Check that replicate count was pruned
    assert new_panel.theta.num_replicates() == 1

    # Test as_pomp with nsim > 1 (should warn)
    with pytest.warns(UserWarning, match="as_pomp is True, but nsim > 1"):
        panel.simulate(key=key, nsim=5, as_pomp=True)
