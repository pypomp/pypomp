"""Tests for Pomp.pmcmc() — Particle MCMC (PMMH)."""

from copy import deepcopy
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pypomp as pp
from pypomp.results import PompPMCMCResult
from pypomp.proposals import mvn_diag_rw, mvn_rw, mvn_rw_adaptive, MVNRWAdaptive


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _get_sir():
    """Return a fresh SIR Pomp for testing."""
    return pp.sir(seed=42)


def _flat_dprior(theta_dict):
    """Flat (improper) prior — always returns 0."""
    return 0.0


def _informative_dprior(theta_dict):
    """Simple log-normal prior on gamma and rho."""
    lp = 0.0
    for p in ["gamma", "rho"]:
        v = theta_dict.get(p, 1.0)
        if v <= 0:
            return -np.inf
        lp += -0.5 * (np.log(v)) ** 2  # log-normal(0, 1) prior
    return lp


# ---------------------------------------------------------------
# Proposal tests
# ---------------------------------------------------------------

class TestProposals:
    def test_mvn_diag_rw_basic(self):
        prop = mvn_diag_rw({"beta1": 0.1, "gamma": 0.1})
        theta = {"beta1": 400.0, "gamma": 26.0, "mu": 0.02}
        key = jax.random.key(0)
        theta_new = prop(theta, key)
        assert set(theta_new.keys()) == set(theta.keys())
        # mu should not change (not in rw_sd)
        assert theta_new["mu"] == theta["mu"]
        # beta1 and gamma should change
        assert theta_new["beta1"] != theta["beta1"]
        assert theta_new["gamma"] != theta["gamma"]

    def test_mvn_diag_rw_zero_sd_dropped(self):
        prop = mvn_diag_rw({"beta1": 0.1, "gamma": 0.0})
        theta = {"beta1": 400.0, "gamma": 26.0}
        key = jax.random.key(1)
        theta_new = prop(theta, key)
        # gamma sd=0 → dropped → gamma unchanged
        assert theta_new["gamma"] == theta["gamma"]

    def test_mvn_diag_rw_empty_raises(self):
        with pytest.raises(ValueError, match="at least one positive"):
            mvn_diag_rw({"a": 0.0, "b": -1.0})

    def test_mvn_rw_basic(self):
        rw_var = np.array([[1.0, 0.0], [0.0, 0.1]])
        prop = mvn_rw(rw_var, param_names=["beta1", "gamma"])
        theta = {"beta1": 400.0, "gamma": 26.0, "mu": 0.02}
        key = jax.random.key(2)
        theta_new = prop(theta, key)
        assert theta_new["mu"] == theta["mu"]
        assert theta_new["beta1"] != theta["beta1"]

    def test_mvn_rw_nonsquare_raises(self):
        with pytest.raises(ValueError, match="square matrix"):
            mvn_rw(np.ones((2, 3)), ["a", "b"])

    def test_mvn_rw_adaptive_basic(self):
        prop = mvn_rw_adaptive(rw_sd={"beta1": 1.0, "gamma": 0.1})
        theta = {"beta1": 400.0, "gamma": 26.0, "mu": 0.02}
        key = jax.random.key(3)
        theta_new = prop(theta, key, n=1, accepts=0)
        assert theta_new["mu"] == theta["mu"]
        assert isinstance(prop, MVNRWAdaptive)

    def test_mvn_rw_adaptive_reset(self):
        prop = mvn_rw_adaptive(rw_sd={"beta1": 1.0})
        theta = {"beta1": 400.0}
        key = jax.random.key(4)
        prop(theta, key, n=1, accepts=0)
        prop.reset()
        assert prop._theta_mean is None
        assert prop._scaling == 1.0

    def test_mvn_rw_adaptive_rw_var(self):
        rw_var = np.array([[1.0]])
        prop = mvn_rw_adaptive(rw_var=rw_var, param_names=["beta1"])
        theta = {"beta1": 400.0}
        key = jax.random.key(5)
        theta_new = prop(theta, key, n=1, accepts=0)
        assert theta_new["beta1"] != theta["beta1"]

    def test_mvn_rw_adaptive_both_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            mvn_rw_adaptive(rw_sd={"a": 1.0}, rw_var=np.eye(1), param_names=["a"])


# ---------------------------------------------------------------
# PMCMC tests
# ---------------------------------------------------------------

class TestPMCMC:
    def test_basic_run(self):
        """PMCMC should run and produce a PompPMCMCResult."""
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0, "gamma": 0.1})
        sir.pmcmc(
            J=10, Nmcmc=5, proposal=prop,
            key=jax.random.key(0),
        )
        res = sir.results_history[-1]
        assert isinstance(res, PompPMCMCResult)
        assert res.method == "pmcmc"
        assert res.Nmcmc == 5
        assert res.J == 10
        assert res.traces_arr.shape == (6, 2 + len(sir.canonical_param_names))

    def test_with_dprior(self):
        """PMCMC should accept a dprior function."""
        sir = _get_sir()
        prop = mvn_diag_rw({"gamma": 0.1, "rho": 0.01})
        sir.pmcmc(
            J=10, Nmcmc=5, proposal=prop,
            dprior=_informative_dprior,
            key=jax.random.key(1),
        )
        res = sir.results_history[-1]
        assert isinstance(res, PompPMCMCResult)
        # log_prior column should have non-zero values
        assert not np.allclose(res.traces_arr[:, 1], 0.0)

    def test_flat_prior_default(self):
        """Default dprior=None should give flat prior (log_prior=0)."""
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.pmcmc(
            J=10, Nmcmc=3, proposal=prop,
            key=jax.random.key(2),
        )
        res = sir.results_history[-1]
        # All log_prior values should be 0
        np.testing.assert_array_equal(res.traces_arr[:, 1], 0.0)

    def test_reps_gt_1(self):
        """reps > 1 should use logmeanexp for more stable loglik."""
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.pmcmc(
            J=10, Nmcmc=3, proposal=prop,
            reps=3,
            key=jax.random.key(3),
        )
        res = sir.results_history[-1]
        assert res.reps == 3
        assert res.traces_arr.shape[0] == 4

    def test_acceptance_rate(self):
        """Acceptance rate should be in [0, 1]."""
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 0.01, "gamma": 0.001})
        sir.pmcmc(
            J=20, Nmcmc=10, proposal=prop,
            key=jax.random.key(4),
        )
        res = sir.results_history[-1]
        assert 0 <= res.acceptance_rate <= 1

    def test_reproducibility(self):
        """Same key should produce identical results."""
        kwargs = dict(
            J=10, Nmcmc=5,
            proposal=mvn_diag_rw({"beta1": 1.0}),
            key=jax.random.key(99),
        )
        sir1 = _get_sir()
        sir1.pmcmc(**kwargs)
        res1 = sir1.results_history[-1]

        sir2 = _get_sir()
        sir2.pmcmc(**kwargs)
        res2 = sir2.results_history[-1]

        np.testing.assert_array_equal(res1.traces_arr, res2.traces_arr)
        assert res1.accepts == res2.accepts

    def test_to_dataframe(self):
        """to_dataframe should return a well-formed DataFrame."""
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.pmcmc(J=10, Nmcmc=3, proposal=prop, key=jax.random.key(5))
        res = sir.results_history[-1]
        df = res.to_dataframe()
        assert "iteration" in df.columns
        assert "chain" in df.columns
        assert "loglik" in df.columns
        assert "log_prior" in df.columns
        assert len(df) == 4  # Nmcmc + 1
        assert all(df["chain"] == 0)

    def test_traces_method(self):
        """traces() should return a DataFrame with replicate, chain, and method columns."""
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.pmcmc(J=10, Nmcmc=3, proposal=prop, key=jax.random.key(6))
        res = sir.results_history[-1]
        df = res.traces()
        assert "replicate" in df.columns
        assert "chain" in df.columns
        assert "method" in df.columns
        assert all(df["method"] == "pmcmc")
        assert all(df["chain"] == 0)

    def test_print_summary(self, capsys):
        """print_summary should not error."""
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.pmcmc(J=10, Nmcmc=3, proposal=prop, key=jax.random.key(7))
        res = sir.results_history[-1]
        res.print_summary()
        captured = capsys.readouterr()
        assert "pmcmc" in captured.out

    def test_theta_updated(self):
        """self.theta should be updated to the final accepted parameters."""
        sir = _get_sir()
        original_theta = deepcopy(sir.theta.to_list()[0])
        prop = mvn_diag_rw({"beta1": 10.0, "gamma": 1.0})
        sir.pmcmc(J=20, Nmcmc=20, proposal=prop, key=jax.random.key(8))
        final_theta = sir.theta.to_list()[0]
        res = sir.results_history[-1]
        if res.accepts > 0:
            changed = any(
                original_theta[p] != final_theta[p]
                for p in original_theta
            )
            assert changed

    def test_with_mvn_rw(self):
        """Should work with full-covariance MVN proposal."""
        sir = _get_sir()
        rw_var = np.diag([1.0, 0.1])
        prop = mvn_rw(rw_var, param_names=["beta1", "gamma"])
        sir.pmcmc(J=10, Nmcmc=3, proposal=prop, key=jax.random.key(9))
        res = sir.results_history[-1]
        assert isinstance(res, PompPMCMCResult)

    def test_with_adaptive_proposal(self):
        """Should work with adaptive MVN proposal."""
        sir = _get_sir()
        prop = mvn_rw_adaptive(
            rw_sd={"beta1": 1.0, "gamma": 0.1},
            scale_start=2, shape_start=3,
        )
        sir.pmcmc(J=10, Nmcmc=5, proposal=prop, key=jax.random.key(10))
        res = sir.results_history[-1]
        assert isinstance(res, PompPMCMCResult)

    def test_verbose(self, capsys):
        """verbose=True should print progress."""
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.pmcmc(
            J=10, Nmcmc=3, proposal=prop,
            key=jax.random.key(11), verbose=True,
        )
        captured = capsys.readouterr()
        assert "PMCMC iteration" in captured.out

    # --- Validation tests ---

    def test_invalid_J(self):
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        with pytest.raises(ValueError, match="J must be"):
            sir.pmcmc(J=0, Nmcmc=5, proposal=prop, key=jax.random.key(20))

    def test_invalid_Nmcmc(self):
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        with pytest.raises(ValueError, match="Nmcmc must be"):
            sir.pmcmc(J=10, Nmcmc=0, proposal=prop, key=jax.random.key(21))

    def test_multi_replicate_raises(self):
        sir = _get_sir()
        theta = pp.PompParameters([sir.theta.to_list()[0], sir.theta.to_list()[0]])
        prop = mvn_diag_rw({"beta1": 1.0})
        with pytest.raises(ValueError, match="exactly one parameter replicate"):
            sir.pmcmc(J=10, Nmcmc=5, proposal=prop, theta=theta, key=jax.random.key(22))

    def test_no_dmeas_raises(self):
        """pmcmc should raise if dmeas is None."""
        sir = _get_sir()
        sir.dmeas = None
        prop = mvn_diag_rw({"beta1": 1.0})
        with pytest.raises(ValueError, match="dmeas"):
            sir.pmcmc(J=10, Nmcmc=5, proposal=prop, key=jax.random.key(23))

    def test_prior_rejects_invalid(self):
        """A prior that returns -inf should cause proposals to be rejected."""
        sir = _get_sir()

        def strict_prior(theta_dict):
            if theta_dict.get("rho", 0) < 0 or theta_dict.get("rho", 1) > 1:
                return -np.inf
            return 0.0

        # Use a huge step so proposals often violate the constraint
        prop = mvn_diag_rw({"rho": 5.0})
        sir.pmcmc(
            J=10, Nmcmc=10, proposal=prop,
            dprior=strict_prior,
            key=jax.random.key(24),
        )
        res = sir.results_history[-1]
        # All accepted rho should be in [0, 1]
        rho_idx = res.trace_names.index("rho")
        assert np.all(res.traces_arr[:, rho_idx] >= 0)
        assert np.all(res.traces_arr[:, rho_idx] <= 1)


# ---------------------------------------------------------------
# Merge test
# ---------------------------------------------------------------

class TestPMCMCMerge:
    def test_merge_two_chains(self):
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})

        sir.pmcmc(J=10, Nmcmc=3, proposal=prop, key=jax.random.key(30))
        res1 = sir.results_history[-1]

        sir.pmcmc(J=10, Nmcmc=3, proposal=prop, key=jax.random.key(31))
        res2 = sir.results_history[-1]

        merged = PompPMCMCResult.merge(res1, res2)
        assert merged.traces_arr.shape[0] == res1.traces_arr.shape[0] + res2.traces_arr.shape[0]
        assert merged.Nmcmc == 6
        assert merged.accepts == res1.accepts + res2.accepts

        # Chain column should distinguish the two chains
        df = merged.to_dataframe()
        assert "chain" in df.columns
        assert set(df["chain"].unique()) == {0, 1}
        assert (df["chain"] == 0).sum() == res1.traces_arr.shape[0]
        assert (df["chain"] == 1).sum() == res2.traces_arr.shape[0]

        # traces() should also have chain column
        df_traces = merged.traces()
        assert "chain" in df_traces.columns
        assert set(df_traces["chain"].unique()) == {0, 1}
