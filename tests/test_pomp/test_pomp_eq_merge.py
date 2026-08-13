"""Equality and merge semantics for Pomp."""

from copy import deepcopy

import jax
import pandas as pd
import pytest

import pypomp as pp
from tests.helpers.assertions import pickle_roundtrip
from tests.helpers.dummy import (
    dummy_dmeas,
    dummy_rinit,
    dummy_rmeas,
    dummy_rproc,
)


def test_eq_comparisons(base_pomp):
    """Test all inequality paths in __eq__."""
    assert base_pomp != "not_pomp"

    # 1. canonical_param_names mismatch
    p2_diff_params = pp.Pomp(
        ys=base_pomp.ys,
        theta=pp.PompParameters({"diff_name": 0.0, "sigma": 0.1}),
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        dmeas=dummy_dmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    assert base_pomp != p2_diff_params

    # 2. one theta is None
    p2 = pickle_roundtrip(base_pomp)
    p2._theta = None
    assert base_pomp != p2

    # 3. different theta values
    p3 = pickle_roundtrip(base_pomp)
    p3.theta = pp.PompParameters({"X0": 1.0, "sigma": 0.1})
    assert base_pomp != p3

    # 4. different ys
    p4 = pickle_roundtrip(base_pomp)
    p4.ys = pd.DataFrame({"y": [3.0, 4.0]}, index=[1.0, 2.0])
    assert base_pomp != p4

    # 5. different statenames
    p5 = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=lambda theta_, key, covars, t0: {"Y": theta_["X0"]},
        rproc=lambda X_, theta_, key, covars, t, dt: {"Y": X_["Y"]},
        dmeas=lambda Y_, X_, theta_, covars, t: 0.0,
        statenames=["Y"],
        t0=0.0,
        nstep=1,
    )
    assert base_pomp != p5

    # 6. different t0
    p6 = pickle_roundtrip(base_pomp)
    p6.t0 = 10.0
    assert base_pomp != p6

    # 7. different fresh_key
    p7 = pickle_roundtrip(base_pomp)
    p7.fresh_key = jax.random.key(99)
    assert base_pomp != p7


def test_eq_covars_and_derived_arrays(base_pomp):
    """Cover __eq__ branches for covars, covars_extended, nstep/dt arrays."""
    covars = pd.DataFrame({"c": [1.0, 2.0]}, index=[1.0, 2.0])

    p_with_covars_a = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        dmeas=dummy_dmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
        covars=covars,
    )
    # one has covars, the other does not
    assert base_pomp != p_with_covars_a

    # both have covars, but different values
    p_with_covars_b = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        dmeas=dummy_dmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
        covars=pd.DataFrame({"c": [3.0, 4.0]}, index=[1.0, 2.0]),
    )
    assert p_with_covars_a != p_with_covars_b

    # differing _covars_extended directly (bypassing normal construction)
    p_covars_extended_diff = deepcopy(p_with_covars_a)
    p_covars_extended_diff._covars_extended = (
        p_with_covars_a._covars_extended + 1.0
        if p_with_covars_a._covars_extended is not None
        else None
    )
    assert p_with_covars_a != p_covars_extended_diff

    p_covars_extended_none = deepcopy(p_with_covars_a)
    p_covars_extended_none._covars_extended = None
    assert p_with_covars_a != p_covars_extended_none

    # differing nstep_array / dt_array_extended / max_steps_per_interval
    p_nstep_diff = deepcopy(base_pomp)
    p_nstep_diff._nstep_array = base_pomp._nstep_array + 1
    assert base_pomp != p_nstep_diff

    p_dt_diff = deepcopy(base_pomp)
    p_dt_diff._dt_array_extended = base_pomp._dt_array_extended + 1.0
    assert base_pomp != p_dt_diff

    p_max_steps_diff = deepcopy(base_pomp)
    p_max_steps_diff._max_steps_per_interval = base_pomp._max_steps_per_interval + 1
    assert base_pomp != p_max_steps_diff


def test_eq_model_components(base_pomp):
    """Cover __eq__ branches for rinit, rproc, dmeas, rmeas, results_history, par_trans."""

    def other_rinit(theta_, key, covars, t0):
        val = next(iter(theta_.values()))
        return {"X": val + 1.0}

    p_diff_rinit = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=other_rinit,
        rproc=dummy_rproc,
        dmeas=dummy_dmeas,
        rmeas=dummy_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    assert base_pomp != p_diff_rinit

    def other_rproc(X_, theta_, key, covars, t, dt):
        return {"X": X_["X"] + 1.0}

    p_diff_rproc = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=other_rproc,
        dmeas=dummy_dmeas,
        rmeas=dummy_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    assert base_pomp != p_diff_rproc

    # dmeas: one None, other not
    p_no_dmeas = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        rmeas=dummy_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    assert base_pomp != p_no_dmeas

    # dmeas: both present but different
    def other_dmeas(Y_, X_, theta_, covars, t):
        return jax.scipy.stats.norm.logpdf(Y_["y"], loc=X_["X"], scale=1.0)

    p_diff_dmeas = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        dmeas=other_dmeas,
        rmeas=dummy_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    assert base_pomp != p_diff_dmeas

    # rmeas: one None, other not
    p_no_rmeas = deepcopy(base_pomp)
    p_no_rmeas.rmeas = None
    assert base_pomp != p_no_rmeas

    # rmeas: both present but different
    def other_rmeas(X_, theta_, key, covars, t):
        return {"y": X_["X"] + 1.0}

    p_diff_rmeas = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        dmeas=dummy_dmeas,
        rmeas=other_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    assert base_pomp != p_diff_rmeas

    # results_history mismatch (pfilter also mutates theta, so build the
    # comparison history separately and graft it on to isolate this branch)
    p_diff_history = deepcopy(base_pomp)
    p_with_results = deepcopy(base_pomp)
    p_with_results.pfilter(J=5, reps=1, key=jax.random.key(0))
    p_diff_history.results_history = p_with_results.results_history
    assert base_pomp != p_diff_history

    # par_trans mismatch
    def to_est(theta_):
        return {k: v + 1.0 for k, v in theta_.items()}

    def from_est(theta_):
        return {k: v - 1.0 for k, v in theta_.items()}

    p_diff_par_trans = deepcopy(base_pomp)
    p_diff_par_trans.par_trans = pp.ParTrans(to_est, from_est)
    assert base_pomp != p_diff_par_trans

    # fresh_key: one is None
    p_no_key = deepcopy(base_pomp)
    p_no_key.fresh_key = None
    assert base_pomp != p_no_key


def test_merge_component_mismatches(base_pomp):
    """Cover merge() validation branches for rinit/rproc, dmeas, rmeas, par_trans."""

    def other_rproc(X_, theta_, key, covars, t, dt):
        return {"X": X_["X"] + 1.0}

    p_diff_rproc = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=other_rproc,
        dmeas=dummy_dmeas,
        rmeas=dummy_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    with pytest.raises(ValueError, match="same rinit and rproc"):
        pp.Pomp.merge(base_pomp, p_diff_rproc)

    def other_dmeas(Y_, X_, theta_, covars, t):
        return jax.scipy.stats.norm.logpdf(Y_["y"], loc=X_["X"], scale=1.0)

    p_diff_dmeas = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        dmeas=other_dmeas,
        rmeas=dummy_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    with pytest.raises(ValueError, match="same dmeas"):
        pp.Pomp.merge(base_pomp, p_diff_dmeas)

    p_no_rmeas = deepcopy(base_pomp)
    p_no_rmeas.rmeas = None
    with pytest.raises(ValueError, match="same rmeas \\(both None"):
        pp.Pomp.merge(base_pomp, p_no_rmeas)

    def other_rmeas(X_, theta_, key, covars, t):
        return {"y": X_["X"] + 1.0}

    p_diff_rmeas = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        dmeas=dummy_dmeas,
        rmeas=other_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    with pytest.raises(ValueError, match="same rmeas\\."):
        pp.Pomp.merge(base_pomp, p_diff_rmeas)

    def to_est(theta_):
        return {k: v + 1.0 for k, v in theta_.items()}

    def from_est(theta_):
        return {k: v - 1.0 for k, v in theta_.items()}

    p_diff_par_trans = deepcopy(base_pomp)
    p_diff_par_trans.par_trans = pp.ParTrans(to_est, from_est)
    with pytest.raises(ValueError, match="same par_trans"):
        pp.Pomp.merge(base_pomp, p_diff_par_trans)
