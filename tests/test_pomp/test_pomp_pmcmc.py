"""Tests for Pomp.pmcmc() — JIT-compiled Particle MCMC (PMMH)."""

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pypomp as pp
from pypomp.core.results import PompPMCMCResult
from pypomp.proposals import (
    mvn_diag_rw,
    mvn_rw,
    mvn_rw_adaptive,
    MVNDiagRW,
    MVNRWFull,
    MVNRWAdaptive,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _get_sir():
    """Return a fresh SIR Pomp for testing."""
    return pp.models.sir(seed=42)


def _flat_dprior(theta_arr):
    """Flat (improper) prior — always returns 0."""
    return jnp.zeros((), dtype=theta_arr.dtype)


def _informative_dprior(theta_arr):
    """Toy log-normal prior on the first parameter (gamma) using a JAX function."""
    v = jnp.maximum(theta_arr[0], 1e-10)
    return -0.5 * (jnp.log(v)) ** 2


# ---------------------------------------------------------------
# Proposal tests
# ---------------------------------------------------------------

class TestProposals:
    def test_mvn_diag_rw_basic(self):
        prop = mvn_diag_rw({"beta1": 0.1, "gamma": 0.1})
        assert isinstance(prop, MVNDiagRW)
        assert prop.param_names == ("beta1", "gamma")
        theta = jnp.array([400.0, 26.0])
        state = prop.init_state(theta)
        new_theta, new_state = prop.step(
            state, theta, jax.random.key(0), jnp.int32(1), jnp.int32(0)
        )
        assert new_theta.shape == theta.shape
        assert not jnp.array_equal(new_theta, theta)

    def test_mvn_diag_rw_zero_sd_dropped(self):
        prop = mvn_diag_rw({"beta1": 0.1, "gamma": 0.0})
        assert prop.param_names == ("beta1",)

    def test_mvn_diag_rw_empty_raises(self):
        with pytest.raises(ValueError, match="at least one positive"):
            mvn_diag_rw({"a": 0.0, "b": -1.0})

    def test_mvn_rw_basic(self):
        rw_var = np.array([[1.0, 0.0], [0.0, 0.1]])
        prop = mvn_rw(rw_var, param_names=["beta1", "gamma"])
        assert isinstance(prop, MVNRWFull)
        theta = jnp.array([400.0, 26.0])
        state = prop.init_state(theta)
        new_theta, _ = prop.step(
            state, theta, jax.random.key(0), jnp.int32(1), jnp.int32(0)
        )
        assert not jnp.array_equal(new_theta, theta)

    def test_mvn_rw_nonsquare_raises(self):
        with pytest.raises(ValueError, match="square matrix"):
            mvn_rw(np.ones((2, 3)), ["a", "b"])

    def test_mvn_rw_size_mismatch_raises(self):
        with pytest.raises(ValueError, match="dimensions must match"):
            mvn_rw(np.eye(2), ["a", "b", "c"])

    def test_mvn_rw_adaptive_basic(self):
        prop = mvn_rw_adaptive(rw_sd={"beta1": 1.0, "gamma": 0.1})
        assert isinstance(prop, MVNRWAdaptive)
        theta = jnp.array([400.0, 26.0])
        state = prop.init_state(theta)
        new_theta, new_state = prop.step(
            state, theta, jax.random.key(0), jnp.int32(1), jnp.int32(0)
        )
        assert new_theta.shape == theta.shape
        assert float(new_state.initialized) == 1.0

    def test_mvn_rw_adaptive_with_rw_var(self):
        rw_var = np.array([[1.0]])
        prop = mvn_rw_adaptive(rw_var=rw_var, param_names=["beta1"])
        theta = jnp.array([400.0])
        state = prop.init_state(theta)
        new_theta, _ = prop.step(
            state, theta, jax.random.key(0), jnp.int32(1), jnp.int32(0)
        )
        assert new_theta[0] != theta[0]

    def test_mvn_rw_adaptive_both_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            mvn_rw_adaptive(rw_sd={"a": 1.0}, rw_var=np.eye(1), param_names=["a"])

    def test_proposal_jit_scan_compatible(self):
        """All proposals must work inside jax.lax.scan."""
        prop = mvn_rw_adaptive(rw_sd={"beta1": 0.5, "gamma": 0.05})
        theta = jnp.array([400.0, 26.0])
        state = prop.init_state(theta)

        def body(carry, n):
            theta, st, key = carry
            key, sub = jax.random.split(key)
            new_t, new_st = prop.step(st, theta, sub, n, n // 2)
            return (new_t, new_st, key), new_t

        init = (theta, state, jax.random.key(0))
        _, traj = jax.lax.scan(jax.jit(body), init, jnp.arange(5, dtype=jnp.int32))
        assert traj.shape == (5, 2)


# ---------------------------------------------------------------
# PMCMC tests
# ---------------------------------------------------------------

class TestPMCMC:
    def test_basic_run(self):
        """PMCMC should run and produce a PompPMCMCResult."""
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0, "gamma": 0.1})
        sir.pmcmc(J=20, Nmcmc=5, proposal=prop, key=jax.random.key(0))
        res = sir.results_history[-1]
        assert isinstance(res, PompPMCMCResult)
        assert res.method == "pmcmc"
        assert res.Nmcmc == 5
        assert res.J == 20
        assert res.n_chains == 1
        assert res.traces_da.sizes == {
            "theta_idx": 1,
            "iteration": 6,
            "variable": 2 + len(sir.canonical_param_names),
        }
        var_list = list(res.traces_da.coords["variable"].values)
        assert var_list[0] == "logLik"
        assert var_list[1] == "log_prior"

    def test_with_dprior(self):
        sir = _get_sir()
        prop = mvn_diag_rw({"gamma": 0.1})
        sir.pmcmc(
            J=20, Nmcmc=3, proposal=prop,
            dprior=_informative_dprior, key=jax.random.key(1),
        )
        res = sir.results_history[-1]
        log_priors = res.traces_da.sel(variable="log_prior").values
        # Informative prior should not be identically zero.
        assert not np.allclose(log_priors, 0.0)

    def test_flat_prior_default(self):
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.pmcmc(J=20, Nmcmc=3, proposal=prop, key=jax.random.key(2))
        res = sir.results_history[-1]
        log_priors = res.traces_da.sel(variable="log_prior").values
        np.testing.assert_array_equal(log_priors, 0.0)

    def test_acceptance_rate_in_range(self):
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 0.01, "gamma": 0.001})
        sir.pmcmc(J=20, Nmcmc=10, proposal=prop, key=jax.random.key(3))
        res = sir.results_history[-1]
        for r in res.acceptance_rate:
            assert 0.0 <= r <= 1.0

    def test_reproducibility(self):
        prop = mvn_diag_rw({"beta1": 1.0})
        sir1 = _get_sir()
        sir1.pmcmc(J=20, Nmcmc=5, proposal=prop, key=jax.random.key(99))
        res1 = sir1.results_history[-1]

        sir2 = _get_sir()
        sir2.pmcmc(J=20, Nmcmc=5, proposal=prop, key=jax.random.key(99))
        res2 = sir2.results_history[-1]

        np.testing.assert_array_equal(
            np.asarray(res1.traces_da.values),
            np.asarray(res2.traces_da.values),
        )
        np.testing.assert_array_equal(res1.accepts, res2.accepts)

    def test_to_dataframe(self):
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.pmcmc(J=20, Nmcmc=3, proposal=prop, key=jax.random.key(4))
        df = sir.results_history[-1].to_dataframe()
        assert "logLik" in df.columns
        assert "log_prior" in df.columns
        assert "chain" in df.columns
        assert "iteration" in df.columns
        assert len(df) == 4  # Nmcmc + 1

    def test_traces_method(self):
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.pmcmc(J=20, Nmcmc=3, proposal=prop, key=jax.random.key(5))
        df = sir.results_history[-1].traces()
        assert "method" in df.columns
        assert "replicate" in df.columns
        assert (df["method"] == "pmcmc").all()

    def test_print_summary(self, capsys):
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.pmcmc(J=20, Nmcmc=3, proposal=prop, key=jax.random.key(6))
        sir.results_history[-1].print_summary()
        out = capsys.readouterr().out
        assert "Method: pmcmc" in out
        assert "Number of chains: 1" in out

    def test_theta_updated(self):
        sir = _get_sir()
        starting = deepcopy(sir.theta[0])
        prop = mvn_diag_rw({"beta1": 50.0})  # large jumps -> some acceptance
        sir.pmcmc(J=20, Nmcmc=8, proposal=prop, key=jax.random.key(7))
        # self.theta should be updated to last accepted sample of chain 0.
        final = sir.theta[0]
        # If at least one accepted, beta1 should differ; otherwise equal.
        res = sir.results_history[-1]
        if int(res.accepts[0]) > 0:
            assert final["beta1"] != starting["beta1"]

    def test_with_mvn_rw(self):
        sir = _get_sir()
        prop = mvn_rw(np.array([[1.0]]), ["beta1"])
        sir.pmcmc(J=20, Nmcmc=3, proposal=prop, key=jax.random.key(8))
        res = sir.results_history[-1]
        assert isinstance(res, PompPMCMCResult)

    def test_with_adaptive_proposal(self):
        sir = _get_sir()
        prop = mvn_rw_adaptive(
            rw_sd={"beta1": 1.0, "gamma": 0.1},
            scale_start=2, shape_start=2,
        )
        sir.pmcmc(J=20, Nmcmc=8, proposal=prop, key=jax.random.key(9))
        res = sir.results_history[-1]
        assert isinstance(res, PompPMCMCResult)


# ---------------------------------------------------------------
# Multi-chain tests
# ---------------------------------------------------------------

class TestPMCMCMultiChain:
    def test_three_chains(self):
        sir = _get_sir()
        t1 = dict(sir.theta[0])
        t2 = dict(sir.theta[0]); t2["beta1"] = 380.0
        t3 = dict(sir.theta[0]); t3["beta1"] = 420.0
        prop = mvn_diag_rw({"beta1": 1.0, "gamma": 0.1})
        sir.pmcmc(
            J=20, Nmcmc=5, proposal=prop,
            theta=[t1, t2, t3], key=jax.random.key(11),
        )
        res = sir.results_history[-1]
        assert res.n_chains == 3
        assert res.accepts.shape == (3,)
        assert res.traces_da.sizes["theta_idx"] == 3

    def test_chains_independent_keys(self):
        """Different chain start values + same key split should give different traces."""
        sir = _get_sir()
        t1 = dict(sir.theta[0])
        t2 = dict(sir.theta[0]); t2["beta1"] = 380.0
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.pmcmc(J=20, Nmcmc=3, proposal=prop,
                  theta=[t1, t2], key=jax.random.key(12))
        res = sir.results_history[-1]
        c0_lls = res.traces_da.isel(theta_idx=0).sel(variable="logLik").values
        c1_lls = res.traces_da.isel(theta_idx=1).sel(variable="logLik").values
        # Initial logLiks differ because starting theta differs.
        assert c0_lls[0] != c1_lls[0]


# ---------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------

class TestPMCMCValidation:
    def test_invalid_J(self):
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        with pytest.raises(ValueError, match="J must be"):
            sir.pmcmc(J=0, Nmcmc=3, proposal=prop, key=jax.random.key(0))

    def test_invalid_Nmcmc(self):
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        with pytest.raises(ValueError, match="Nmcmc must be"):
            sir.pmcmc(J=20, Nmcmc=0, proposal=prop, key=jax.random.key(0))


# ---------------------------------------------------------------
# Merge tests
# ---------------------------------------------------------------

class TestPMCMCMerge:
    def test_merge_two_chains(self):
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.pmcmc(J=20, Nmcmc=3, proposal=prop, key=jax.random.key(20))
        res1 = sir.results_history[-1]
        sir.pmcmc(J=20, Nmcmc=3, proposal=prop, key=jax.random.key(21))
        res2 = sir.results_history[-1]

        merged = PompPMCMCResult.merge(res1, res2)
        assert merged.n_chains == res1.n_chains + res2.n_chains
        assert merged.Nmcmc == 3  # unchanged
        assert merged.accepts.shape == (merged.n_chains,)
        # Variable coord preserved.
        assert list(merged.traces_da.coords["variable"].values) == \
            list(res1.traces_da.coords["variable"].values)

    def test_merge_different_Nmcmc_raises(self):
        sir = _get_sir()
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.pmcmc(J=20, Nmcmc=3, proposal=prop, key=jax.random.key(30))
        res1 = sir.results_history[-1]
        sir.pmcmc(J=20, Nmcmc=4, proposal=prop, key=jax.random.key(31))
        res2 = sir.results_history[-1]
        with pytest.raises(ValueError, match="same Nmcmc"):
            PompPMCMCResult.merge(res1, res2)

    def test_merge_empty_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            PompPMCMCResult.merge()
