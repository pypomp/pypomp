import jax
import numpy as np
import pytest

import pypomp as pp
from tests.helpers.models import lg_panel


@pytest.fixture
def simple_panel():
    # Every transition/measurement entry shared, every noise and initial-state
    # entry unit-specific.
    panel = lg_panel(
        sharing="some",
        shared_names=[
            n for n in pp.models.LG().canonical_param_names if n.startswith(("A", "C"))
        ],
    )

    key = jax.random.key(456)
    J = 5
    reps = 2

    panel.pfilter(J=J, key=key, reps=reps, CLL=True, ESS=True)
    return panel, reps, list(panel.get_unit_names())


def test_panel_cll(simple_panel):
    panel, reps, unit_names = simple_panel
    times = list(panel.unit_objects.values())[0].ys.index

    # Raw CLL
    cll_df = panel.CLL()
    assert "CLL" in cll_df.columns
    assert "unit" in cll_df.columns
    assert "theta_idx" in cll_df.columns
    assert "rep" in cll_df.columns
    expected_rows = len(unit_names) * reps * len(times)
    assert cll_df.shape[0] == expected_rows
    assert np.all(np.isfinite(cll_df["CLL"]))

    # Averaged CLL
    cll_avg_df = panel.CLL(average=True)
    assert "CLL" in cll_avg_df.columns
    assert "unit" in cll_avg_df.columns
    assert "theta_idx" in cll_avg_df.columns
    assert "rep" not in cll_avg_df.columns
    expected_avg_rows = len(unit_names) * len(times)
    assert cll_avg_df.shape[0] == expected_avg_rows


def test_panel_ess(simple_panel):
    panel, reps, unit_names = simple_panel
    times = list(panel.unit_objects.values())[0].ys.index

    # Raw ESS
    ess_df = panel.ESS()
    assert "ESS" in ess_df.columns
    assert "unit" in ess_df.columns
    assert "theta_idx" in ess_df.columns
    assert "rep" in ess_df.columns
    expected_rows = len(unit_names) * reps * len(times)
    assert ess_df.shape[0] == expected_rows
    assert np.all(np.isfinite(ess_df["ESS"]))
    # ESS should be between 0 and J=5
    assert np.all((ess_df["ESS"] >= 0) & (ess_df["ESS"] <= 5))

    # Averaged ESS
    ess_avg_df = panel.ESS(average=True)
    assert "ESS" in ess_avg_df.columns
    assert "unit" in ess_avg_df.columns
    assert "theta_idx" in ess_avg_df.columns
    assert "rep" not in ess_avg_df.columns
    expected_avg_rows = len(unit_names) * len(times)
    assert ess_avg_df.shape[0] == expected_avg_rows
