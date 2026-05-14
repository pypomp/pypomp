from copy import deepcopy
import jax
import pandas as pd
import pytest
from typing import cast
import numpy as np
import pypomp as pp
from pypomp.core.results import PanelPompTrainResult


def _get_lg_panel():
    lg1 = pp.models.LG()
    lg2 = pp.models.LG()
    # Create PanelParameters with some shared and some unit-specific
    shared_names = ["A1", "C1"]
    unit_specific_names = [
        n for n in lg1.canonical_param_names if n not in shared_names
    ]

    p1, p2 = lg1.theta[0], lg2.theta[0]
    shared_df = pd.DataFrame(
        {"shared": [(p1[n] + p2[n]) / 2 for n in shared_names]},
        index=pd.Index(shared_names),
    )

    unit_specific_df = pd.DataFrame(
        {
            "unit1": [p1[n] for n in unit_specific_names],
            "unit2": [p2[n] for n in unit_specific_names],
        },
        index=pd.Index(unit_specific_names),
    )

    theta = pp.PanelParameters(
        theta=[{"shared": shared_df, "unit_specific": unit_specific_df}]
    )
    panel = pp.PanelPomp(
        Pomp_dict={"unit1": lg1, "unit2": lg2},
        theta=theta,
    )
    return panel


@pytest.mark.parametrize("chunk_size", [1, 2], ids=["chunk1", "chunk2"])
@pytest.mark.parametrize("optimizer", ["Adam", "FullMatrixAdam"])
def test_panel_train(chunk_size, optimizer):
    panel = _get_lg_panel()
    J, M = 2, 2
    panel.train(
        J=J,
        M=M,
        eta=0.01,
        theta=deepcopy(panel.theta),
        chunk_size=chunk_size,
        optimizer=optimizer,
        key=jax.random.key(1),
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
    assert "A1" in df.columns


def test_panel_train_clipping():
    panel = _get_lg_panel()
    J, M = 2, 1
    eta = 0.5
    key = jax.random.key(0)
    theta_init = deepcopy(panel.theta)

    panel.train(
        J=J,
        M=M,
        eta=eta,
        key=key,
        theta=deepcopy(theta_init),
        clip_norm=None,
        optimizer="SGD",
    )
    res_no_clip = panel.results_history[-1]
    shared_vars = panel.canonical_shared_param_names
    p0 = res_no_clip.shared_traces.sel(
        theta_idx=0, iteration=0, variable=shared_vars
    ).values
    p1_no_clip = res_no_clip.shared_traces.sel(
        theta_idx=0, iteration=1, variable=shared_vars
    ).values
    diff_no_clip = np.linalg.norm(p1_no_clip - p0)

    panel.train(
        J=J,
        M=M,
        eta=eta,
        key=key,
        theta=deepcopy(theta_init),
        clip_norm=1e-5,
        optimizer="SGD",
    )
    res_clip = panel.results_history[-1]
    p1_clip = res_clip.shared_traces.sel(
        theta_idx=0, iteration=1, variable=shared_vars
    ).values
    diff_clip = np.linalg.norm(p1_clip - p0)

    assert diff_clip < diff_no_clip
