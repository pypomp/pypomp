from copy import deepcopy

import jax
import numpy as np
import pandas as pd
import pytest

import pypomp as pp
from pypomp.core.results import Result


def _get_lg_panel():
    lg1 = pp.models.LG()
    lg2 = pp.models.LG()
    # Create PanelParameters with some shared and some unit-specific
    shared_names = ["A11", "C11"]
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


def _get_lg_panel_specific_only():
    """Build a 2-unit LG panel with no shared parameters at all."""
    lg1 = pp.models.LG()
    lg2 = pp.models.LG()
    names = lg1.canonical_param_names
    p1, p2 = lg1.theta[0], lg2.theta[0]
    unit_specific_df = pd.DataFrame(
        {"unit1": [p1[n] for n in names], "unit2": [p2[n] for n in names]},
        index=pd.Index(names),
    )
    theta = pp.PanelParameters(
        theta=[{"shared": None, "unit_specific": unit_specific_df}]
    )
    panel = pp.PanelPomp(Pomp_dict={"unit1": lg1, "unit2": lg2}, theta=theta)
    return panel


def _get_lg_panel_shared_only():
    """Build a 2-unit LG panel with no unit-specific parameters at all."""
    lg1 = pp.models.LG()
    lg2 = pp.models.LG()
    names = lg1.canonical_param_names
    p1, p2 = lg1.theta[0], lg2.theta[0]
    shared_df = pd.DataFrame(
        {"shared": [(p1[n] + p2[n]) / 2 for n in names]}, index=pd.Index(names)
    )
    empty_unit_specific = pd.DataFrame(index=pd.Index([]), columns=["unit1", "unit2"])
    theta = pp.PanelParameters(
        theta=[{"shared": shared_df, "unit_specific": empty_unit_specific}]
    )
    panel = pp.PanelPomp(Pomp_dict={"unit1": lg1, "unit2": lg2}, theta=theta)
    return panel


def test_panel_train_unit_specific_only():
    """No shared parameters: shared_traces should reduce to logLik only.

    Exercises the ``n_shared == 0`` setup branch as well as the
    ``shared_traces is None`` fallback used when assembling the shared trace.
    """
    panel = _get_lg_panel_specific_only()
    J, M = 2, 2
    panel.train(
        J=J,
        M=M,
        eta=pp.LearningRate({n: 0.01 for n in panel.canonical_param_names}),
        key=jax.random.key(11),
    )
    res = panel.results_history[-1]
    assert isinstance(res, Result)
    assert res.shared_traces.shape == (1, M + 1, 1)
    assert list(res.shared_traces.coords["variable"].values) == ["logLik"]
    assert res.unit_traces.shape[-1] == 1 + len(panel.canonical_unit_param_names)
    # All but the first (pre-training) iteration should have a finite logLik.
    assert np.all(np.isfinite(np.asarray(res.shared_traces)[:, 1:, :]))


def test_panel_train_shared_only():
    """No unit-specific parameters: unit_traces should reduce to unitLogLik only.

    Exercises the ``n_spec == 0`` setup branch as well as the
    ``unit_traces is None`` fallback used when assembling the unit trace.
    """
    panel = _get_lg_panel_shared_only()
    J, M = 2, 2
    panel.train(
        J=J,
        M=M,
        eta=pp.LearningRate({n: 0.01 for n in panel.canonical_param_names}),
        key=jax.random.key(13),
    )
    res = panel.results_history[-1]
    assert isinstance(res, Result)
    assert list(res.unit_traces.coords["variable"].values) == ["unitLogLik"]
    assert res.unit_traces.shape[-1] == 1
    assert res.shared_traces.shape[-1] == 1 + len(panel.canonical_shared_param_names)
    # unit_traces is filled with zeros as a placeholder since there are no
    # unit-specific parameters to trace.
    assert np.all(np.asarray(res.unit_traces) == 0.0)


@pytest.mark.parametrize("chunk_size", [1, 2], ids=["chunk1", "chunk2"])
@pytest.mark.parametrize(
    "opt_instance",
    [pp.Adam(), pp.FullMatrixAdam()],
    ids=["Adam", "FullMatrixAdam"],
)
def test_panel_train(chunk_size, opt_instance):
    panel = _get_lg_panel()
    J, M = 2, 2
    panel.train(
        J=J,
        M=M,
        eta=pp.LearningRate({n: 0.01 for n in panel.canonical_param_names}),
        theta=deepcopy(panel.theta),
        chunk_size=chunk_size,
        optimizer=opt_instance,
        key=jax.random.key(1),
    )

    res = panel.results_history[-1]
    assert isinstance(res, Result)
    assert res.method == "train"
    assert res.shared_traces.shape[0] == 1  # n_reps
    assert res.shared_traces.shape[1] == M + 1
    assert res.unit_traces.shape[0] == 1  # n_reps
    assert res.unit_traces.shape[1] == M + 1
    assert res.unit_traces.shape[2] == len(panel.get_unit_names())  # U
    df = res.to_dataframe()
    assert "shared logLik" in df.columns
    assert "unit logLik" in df.columns
    assert "A11" in df.columns


def test_panel_train_clipping():
    panel = _get_lg_panel()
    J, M = 2, 1
    eta = 0.5
    key = jax.random.key(0)
    theta_init = deepcopy(panel.theta)

    panel.train(
        J=J,
        M=M,
        eta=pp.LearningRate({n: eta for n in panel.canonical_param_names}),
        key=key,
        theta=deepcopy(theta_init),
        optimizer=pp.SGD(clip_norm=None),
    )
    res_no_clip = panel.results_history[-1]
    assert isinstance(res_no_clip, Result)
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
        eta=pp.LearningRate({n: eta for n in panel.canonical_param_names}),
        key=key,
        theta=deepcopy(theta_init),
        optimizer=pp.SGD(clip_norm=1e-5),
    )
    res_clip = panel.results_history[-1]
    assert isinstance(res_clip, Result)
    p1_clip = res_clip.shared_traces.sel(
        theta_idx=0, iteration=1, variable=shared_vars
    ).values
    diff_clip = np.linalg.norm(p1_clip - p0)

    assert diff_clip < diff_no_clip
