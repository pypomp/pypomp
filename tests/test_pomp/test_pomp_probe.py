import jax
import pytest
import pypomp as pp
import pandas as pd
import numpy as np


@pytest.fixture(scope="function")
def simple_pomp():
    return pp.models.LG()


def test_pomp_probe_structure(simple_pomp):
    pomp = simple_pomp
    key = jax.random.key(42)

    probes = {
        "mean_y1": lambda df: df["Y1"].mean(),
        "max_y2": lambda df: df["Y2"].max(),
    }

    nsim = 5
    probe_df = pomp.probe(probes=probes, nsim=nsim, key=key)

    assert isinstance(probe_df, pd.DataFrame)
    expected_cols = ["probe", "value", "is_real_data", "theta_idx", "sim"]
    assert all(col in probe_df.columns for col in expected_cols)

    assert len(probe_df) == 2 + (1 * nsim * 2)

    assert probe_df["is_real_data"].sum() == 2
    assert (~probe_df["is_real_data"]).sum() == 10


def test_pomp_probe_values(simple_pomp):
    pomp = simple_pomp
    key = jax.random.key(42)

    probes = {"sum_y1": lambda df: df["Y1"].sum()}

    probe_df = pomp.probe(probes=probes, nsim=2, key=key)

    real_val = probe_df[probe_df["is_real_data"]]["value"].iloc[0]
    expected_real = pomp.ys["Y1"].sum()
    assert np.isclose(real_val, expected_real)

    sim_vals = probe_df[~probe_df["is_real_data"]]["value"]
    assert len(sim_vals) == 2
    assert not all(np.isclose(sim_vals, expected_real))


def test_pomp_probe_with_list_theta(simple_pomp):
    pomp = simple_pomp
    key = jax.random.key(42)

    theta_list = pomp.theta * 3

    probes = {"mean_y1": lambda df: df["Y1"].mean()}

    nsim = 2
    probe_df = pomp.probe(probes=probes, nsim=nsim, key=key, theta=theta_list)

    assert len(probe_df) == 1 + (3 * nsim)

    assert set(probe_df[~probe_df["is_real_data"]]["theta_idx"]) == {0, 1, 2}
    assert set(probe_df[~probe_df["is_real_data"]]["sim"]) == {0, 1}
