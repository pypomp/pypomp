"""Tests for Pomp.abc() — Approximate Bayesian Computation (ABC-MCMC)."""

from copy import deepcopy
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pypomp as pp
from pypomp.results import PompABCResult
from pypomp.proposals import mvn_diag_rw, mvn_rw, mvn_rw_adaptive


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _get_sir():
    """Return a fresh SIR Pomp for testing."""
    return pp.sir(seed=42)


def _default_probes():
    """Simple probes for the SIR model."""
    return {
        "mean_reports": lambda ys: float(ys["reports"].mean()),
        "sd_reports": lambda ys: float(ys["reports"].std()),
    }


def _default_scale():
    """Scale dict matching _default_probes."""
    return {"mean_reports": 100.0, "sd_reports": 100.0}


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
        lp += -0.5 * (np.log(v)) ** 2
    return lp


# ---------------------------------------------------------------
# ABC tests
# ---------------------------------------------------------------

class TestABC:
    def test_basic_run(self):
        """ABC should run and produce a PompABCResult."""
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0, "gamma": 0.1})
        sir.abc(
            Nabc=5, probes=_default_probes(), scale=_default_scale(),
            epsilon=1e6, proposal=prop, key=jax.random.key(0),
        )
        res = sir.results_history[-1]
        assert isinstance(res, PompABCResult)
        assert res.method == "abc"
        assert res.Nabc == 5
        assert res.epsilon == 1e6
        assert res.traces_arr.shape == (6, 2 + len(sir.canonical_param_names))

    def test_with_dprior(self):
        """ABC should accept a dprior function."""
        sir = _get_sir()
        prop = mvn_diag_rw({"gamma": 0.1, "rho": 0.01})
        sir.abc(
            Nabc=5, probes=_default_probes(), scale=_default_scale(),
            epsilon=1e6, proposal=prop,
            dprior=_informative_dprior, key=jax.random.key(1),
        )
        res = sir.results_history[-1]
        assert isinstance(res, PompABCResult)
        assert not np.allclose(res.traces_arr[:, 1], 0.0)

    def test_flat_prior_default(self):
        """Default dprior=None should give flat prior (log_prior=0)."""
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.abc(
            Nabc=3, probes=_default_probes(), scale=_default_scale(),
            epsilon=1e6, proposal=prop, key=jax.random.key(2),
        )
        res = sir.results_history[-1]
        np.testing.assert_array_equal(res.traces_arr[:, 1], 0.0)

    def test_acceptance_rate(self):
        """Acceptance rate should be in [0, 1]."""
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 0.01, "gamma": 0.001})
        sir.abc(
            Nabc=10, probes=_default_probes(), scale=_default_scale(),
            epsilon=1e6, proposal=prop, key=jax.random.key(3),
        )
        res = sir.results_history[-1]
        assert 0 <= res.acceptance_rate <= 1

    def test_reproducibility(self):
        """Same key should produce identical results."""
        kwargs = dict(
            Nabc=5, probes=_default_probes(), scale=_default_scale(),
            epsilon=1e6, proposal=mvn_diag_rw({"beta1": 1.0}),
            key=jax.random.key(99),
        )
        sir1 = _get_sir()
        sir1.abc(**kwargs)
        res1 = sir1.results_history[-1]

        sir2 = _get_sir()
        sir2.abc(**kwargs)
        res2 = sir2.results_history[-1]

        np.testing.assert_array_equal(res1.traces_arr, res2.traces_arr)
        assert res1.accepts == res2.accepts

    def test_to_dataframe(self):
        """to_dataframe should return a well-formed DataFrame."""
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.abc(
            Nabc=3, probes=_default_probes(), scale=_default_scale(),
            epsilon=1e6, proposal=prop, key=jax.random.key(5),
        )
        res = sir.results_history[-1]
        df = res.to_dataframe()
        assert "iteration" in df.columns
        assert "chain" in df.columns
        assert "distance" in df.columns
        assert "log_prior" in df.columns
        assert len(df) == 4  # Nabc + 1
        assert all(df["chain"] == 0)

    def test_traces_method(self):
        """traces() should return a DataFrame with replicate, chain, and method columns."""
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.abc(
            Nabc=3, probes=_default_probes(), scale=_default_scale(),
            epsilon=1e6, proposal=prop, key=jax.random.key(6),
        )
        res = sir.results_history[-1]
        df = res.traces()
        assert "replicate" in df.columns
        assert "chain" in df.columns
        assert "method" in df.columns
        assert all(df["method"] == "abc")
        assert all(df["chain"] == 0)

    def test_print_summary(self, capsys):
        """print_summary should not error."""
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.abc(
            Nabc=3, probes=_default_probes(), scale=_default_scale(),
            epsilon=1e6, proposal=prop, key=jax.random.key(7),
        )
        res = sir.results_history[-1]
        res.print_summary()
        captured = capsys.readouterr()
        assert "abc" in captured.out

    def test_theta_updated(self):
        """self.theta should be updated to the final accepted parameters."""
        sir = _get_sir()
        original_theta = deepcopy(sir.theta.to_list()[0])
        prop = mvn_diag_rw({"beta1": 10.0, "gamma": 1.0})
        sir.abc(
            Nabc=20, probes=_default_probes(), scale=_default_scale(),
            epsilon=1e10, proposal=prop, key=jax.random.key(8),
        )
        final_theta = sir.theta.to_list()[0]
        res = sir.results_history[-1]
        if res.accepts > 0:
            changed = any(
                original_theta[p] != final_theta[p]
                for p in original_theta
            )
            assert changed

    def test_distance_column(self):
        """First trace column should be 'distance', not 'loglik'."""
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.abc(
            Nabc=3, probes=_default_probes(), scale=_default_scale(),
            epsilon=1e6, proposal=prop, key=jax.random.key(9),
        )
        res = sir.results_history[-1]
        assert res.trace_names[0] == "distance"
        assert res.trace_names[1] == "log_prior"
        # Distance should be non-negative
        assert np.all(res.traces_arr[:, 0] >= 0)

    def test_with_mvn_rw(self):
        """Should work with full-covariance MVN proposal."""
        sir = _get_sir()
        rw_var = np.diag([1.0, 0.1])
        prop = mvn_rw(rw_var, param_names=["beta1", "gamma"])
        sir.abc(
            Nabc=3, probes=_default_probes(), scale=_default_scale(),
            epsilon=1e6, proposal=prop, key=jax.random.key(10),
        )
        res = sir.results_history[-1]
        assert isinstance(res, PompABCResult)

    def test_with_adaptive_proposal(self):
        """Should work with adaptive MVN proposal."""
        sir = _get_sir()
        prop = mvn_rw_adaptive(
            rw_sd={"beta1": 1.0, "gamma": 0.1},
            scale_start=2, shape_start=3,
        )
        sir.abc(
            Nabc=5, probes=_default_probes(), scale=_default_scale(),
            epsilon=1e6, proposal=prop, key=jax.random.key(11),
        )
        res = sir.results_history[-1]
        assert isinstance(res, PompABCResult)

    def test_verbose(self, capsys):
        """verbose=True should print progress."""
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.abc(
            Nabc=3, probes=_default_probes(), scale=_default_scale(),
            epsilon=1e6, proposal=prop,
            key=jax.random.key(12), verbose=True,
        )
        captured = capsys.readouterr()
        assert "ABC iteration" in captured.out

    def test_tight_epsilon_low_acceptance(self):
        """Very small epsilon should lead to low acceptance rate."""
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 10.0})
        sir.abc(
            Nabc=10, probes=_default_probes(), scale=_default_scale(),
            epsilon=1e-20, proposal=prop, key=jax.random.key(13),
        )
        res = sir.results_history[-1]
        assert res.accepts == 0


# ---------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------

class TestABCValidation:
    def test_invalid_Nabc(self):
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        with pytest.raises(ValueError, match="Nabc must be"):
            sir.abc(
                Nabc=0, probes=_default_probes(), scale=_default_scale(),
                epsilon=1.0, proposal=prop, key=jax.random.key(20),
            )

    def test_invalid_epsilon(self):
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        with pytest.raises(ValueError, match="epsilon must be positive"):
            sir.abc(
                Nabc=5, probes=_default_probes(), scale=_default_scale(),
                epsilon=0.0, proposal=prop, key=jax.random.key(21),
            )

    def test_empty_probes(self):
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        with pytest.raises(ValueError, match="probes must be"):
            sir.abc(
                Nabc=5, probes={}, scale={},
                epsilon=1.0, proposal=prop, key=jax.random.key(22),
            )

    def test_scale_keys_mismatch(self):
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        with pytest.raises(ValueError, match="scale keys must match"):
            sir.abc(
                Nabc=5, probes=_default_probes(),
                scale={"wrong_key": 1.0},
                epsilon=1.0, proposal=prop, key=jax.random.key(23),
            )

    def test_negative_scale(self):
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        with pytest.raises(ValueError, match="must be positive"):
            sir.abc(
                Nabc=5, probes=_default_probes(),
                scale={"mean_reports": -1.0, "sd_reports": 1.0},
                epsilon=1.0, proposal=prop, key=jax.random.key(24),
            )

    def test_multi_replicate_raises(self):
        sir = _get_sir()
        theta = pp.PompParameters([sir.theta.to_list()[0], sir.theta.to_list()[0]])
        prop = mvn_diag_rw({"beta1": 1.0})
        with pytest.raises(ValueError, match="exactly one parameter replicate"):
            sir.abc(
                Nabc=5, probes=_default_probes(), scale=_default_scale(),
                epsilon=1.0, proposal=prop, theta=theta, key=jax.random.key(25),
            )

    def test_no_rmeas_raises(self):
        """abc should raise if rmeas is None."""
        sir = _get_sir()
        sir.rmeas = None
        prop = mvn_diag_rw({"beta1": 1.0})
        with pytest.raises(ValueError, match="rmeas"):
            sir.abc(
                Nabc=5, probes=_default_probes(), scale=_default_scale(),
                epsilon=1.0, proposal=prop, key=jax.random.key(26),
            )


# ---------------------------------------------------------------
# Merge test
# ---------------------------------------------------------------

class TestABCMerge:
    def test_merge_two_chains(self):
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        kwargs = dict(
            probes=_default_probes(), scale=_default_scale(),
            epsilon=1e6, proposal=prop,
        )

        sir.abc(Nabc=3, key=jax.random.key(30), **kwargs)
        res1 = sir.results_history[-1]

        sir.abc(Nabc=3, key=jax.random.key(31), **kwargs)
        res2 = sir.results_history[-1]

        merged = PompABCResult.merge(res1, res2)
        assert merged.traces_arr.shape[0] == res1.traces_arr.shape[0] + res2.traces_arr.shape[0]
        assert merged.Nabc == 6
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

    def test_merge_different_epsilon_raises(self):
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        kwargs = dict(
            probes=_default_probes(), scale=_default_scale(), proposal=prop,
        )

        sir.abc(Nabc=3, epsilon=1e6, key=jax.random.key(40), **kwargs)
        res1 = sir.results_history[-1]

        sir.abc(Nabc=3, epsilon=1e3, key=jax.random.key(41), **kwargs)
        res2 = sir.results_history[-1]

        with pytest.raises(ValueError, match="same epsilon"):
            PompABCResult.merge(res1, res2)
