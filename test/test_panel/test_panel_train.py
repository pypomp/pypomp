from copy import deepcopy
import jax
import pandas as pd
import pytest
from typing import cast
import numpy as np
import pypomp as pp
from pypomp.results import PanelPompTrainResult


def _get_measles_003_panel():
    AK_mles = cast(pd.DataFrame, pp.UKMeasles.AK_mles())
    london_theta = AK_mles["London"].to_dict()
    hastings_theta = AK_mles["Hastings"].to_dict()
    london = pp.UKMeasles.Pomp(
        unit=["London"],
        theta=london_theta,
        model="003",
    )
    hastings = pp.UKMeasles.Pomp(
        unit=["Hastings"],
        theta=hastings_theta,
        model="003",
    )
    unit_specific = cast(
        pd.DataFrame, AK_mles[["London", "Hastings"]].drop(labels=["gamma", "cohort"])
    )
    shared = cast(
        pd.DataFrame,
        AK_mles[["London", "Hastings"]]
        .loc[["gamma", "cohort"], :]
        .mean(axis=1)
        .to_frame(name="shared"),
    )
    theta = pp.PanelParameters(
        theta=[{"shared": shared, "unit_specific": unit_specific}]
    )
    panel = pp.PanelPomp(
        Pomp_dict={"London": london, "Hastings": hastings},
        theta=theta,
    )
    return panel


@pytest.mark.parametrize("chunk_size", [1, 2], ids=["chunk1", "chunk2"])
@pytest.mark.parametrize("optimizer", ["Adam", "FullMatrixAdam"])
def test_panel_train(chunk_size, optimizer):
    panel = _get_measles_003_panel()
    J, M = 2, 2
    panel.train(
        J=J,
        M=M,
        eta=0.01,
        theta=deepcopy(panel.theta),
        chunk_size=chunk_size,
        optimizer=optimizer,
        key=jax.random.key(0),
    )

    res = panel.results_history[-1]
    assert isinstance(res, PanelPompTrainResult)
    assert res.shared_traces.shape[0] == 1  # n_reps
    assert res.shared_traces.shape[1] == M + 1
    assert res.unit_traces.shape[0] == 1  # n_reps
    assert res.unit_traces.shape[1] == M + 1
    assert res.unit_traces.shape[3] == len(panel.get_unit_names())  # U
    df = res.to_dataframe()
    assert "shared logLik" in df.columns
    assert "unit logLik" in df.columns
    assert "R0" in df.columns


def test_panel_train_clipping():
    panel = _get_measles_003_panel()
    J, M = 2, 1
    eta = 10.0
    key = jax.random.key(0)
    theta_init = deepcopy(panel.theta)

    panel.train(J=J, M=M, eta=eta, key=key, theta=deepcopy(theta_init), clip_norm=None)
    res_no_clip = panel.results_history[-1]
    shared_vars = panel.canonical_shared_param_names
    p0 = res_no_clip.shared_traces.sel(
        replicate=0, iteration=0, variable=shared_vars
    ).values
    p1_no_clip = res_no_clip.shared_traces.sel(
        replicate=0, iteration=1, variable=shared_vars
    ).values
    diff_no_clip = np.linalg.norm(p1_no_clip - p0)

    panel.train(J=J, M=M, eta=eta, key=key, theta=deepcopy(theta_init), clip_norm=1e-5)
    res_clip = panel.results_history[-1]
    p1_clip = res_clip.shared_traces.sel(
        replicate=0, iteration=1, variable=shared_vars
    ).values
    diff_clip = np.linalg.norm(p1_clip - p0)

    assert diff_clip < diff_no_clip
