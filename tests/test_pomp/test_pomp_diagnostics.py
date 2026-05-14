import jax
import pytest
import pypomp as pp
import numpy as np


@pytest.fixture
def simple_pomp():
    LG_obj = pp.models.LG()
    key = jax.random.key(123)
    J = 10
    reps = 3
    LG_obj.pfilter(J=J, key=key, reps=reps, CLL=True, ESS=True)
    return LG_obj, reps


def test_pomp_cll(simple_pomp):
    LG, reps = simple_pomp

    # Raw CLL
    cll_df = LG.CLL()
    assert "CLL" in cll_df.columns
    assert "theta_idx" in cll_df.columns
    assert "rep" in cll_df.columns
    assert cll_df.shape[0] == reps * len(LG.ys)
    assert np.all(np.isfinite(cll_df["CLL"]))

    # Averaged CLL
    cll_avg_df = LG.CLL(average=True)
    assert "CLL" in cll_avg_df.columns
    assert "theta_idx" in cll_avg_df.columns
    assert "rep" not in cll_avg_df.columns
    assert cll_avg_df.shape[0] == len(LG.ys)
    # theta_idx should be 0 for averaged results (first parameter set)
    assert np.all(cll_avg_df["theta_idx"] == 0)


def test_pomp_ess(simple_pomp):
    LG, reps = simple_pomp

    # Raw ESS
    ess_df = LG.ESS()
    assert "ESS" in ess_df.columns
    assert "theta_idx" in ess_df.columns
    assert "rep" in ess_df.columns
    assert ess_df.shape[0] == reps * len(LG.ys)
    assert np.all(np.isfinite(ess_df["ESS"]))
    # ESS should be between 0 and J=10
    assert np.all((ess_df["ESS"] >= 0) & (ess_df["ESS"] <= 10))

    # Averaged ESS
    ess_avg_df = LG.ESS(average=True)
    assert "ESS" in ess_avg_df.columns
    assert "theta_idx" in ess_avg_df.columns
    assert "rep" not in ess_avg_df.columns
    assert ess_avg_df.shape[0] == len(LG.ys)
    assert np.all(ess_avg_df["theta_idx"] == 0)
