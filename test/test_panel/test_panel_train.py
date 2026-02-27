from copy import deepcopy
import jax
import pandas as pd
from typing import cast
import pypomp as pp


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


def test_panel_train():
    panel = _get_measles_003_panel()

    J = 2
    M = 2

    theta = deepcopy(panel.theta)

    panel.train(J=J, M=M, eta=0.01, theta=theta, chunk_size=1, key=jax.random.key(0))

    res = panel.results_history[-1]

    from pypomp.results import PanelPompTrainResult

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
