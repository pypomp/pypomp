"""Tests for Pomp.pmcmc() -- JIT-compiled Particle MCMC (PMMH)."""

from copy import deepcopy
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pypomp as pp
from pypomp.core.results import Result
from pypomp import MVNDiagRW, MVNRWFull, MVNRWAdaptive


def _pmcmc_res(res) -> Result:
    assert isinstance(res, Result)
    return res


@dataclass(frozen=True)
class _FixedShiftProposal:
    """Deterministic proposal used for non-random PMCMC acceptance tests."""

    shift: jax.Array
    param_names: tuple[str, ...]

    def init_state(self, theta_arr: jax.Array) -> tuple:
        return ()

    def step(
        self,
        state: tuple,
        theta_arr: jax.Array,
        key: jax.Array,
        n: jax.Array,
        accepts: jax.Array,
    ) -> tuple[jax.Array, tuple]:
        return theta_arr + self.shift, state

    def canonicalize(self, canonical_names):
        # Fixed shift is already aligned to the model's parameter vector.
        return self


jax.tree_util.register_pytree_node(
    _FixedShiftProposal,
    lambda p: ((p.shift,), p.param_names),
    lambda aux, children: _FixedShiftProposal(shift=children[0], param_names=aux),
)


# ---------------------------------------------------------------
# Prior helpers
# ---------------------------------------------------------------


def _flat_dprior(theta_arr):
    """Flat (improper) prior -- always returns 0."""
    return jnp.zeros((), dtype=theta_arr.dtype)


def _informative_dprior(theta_arr):
    """Toy log-normal prior on the first parameter (gamma) using a JAX function."""
    v = jnp.maximum(theta_arr[0], 1e-10)
    return -0.5 * (jnp.log(v)) ** 2


# ---------------------------------------------------------------
# Proposal tests (no SIR model needed)
# ---------------------------------------------------------------


class TestProposals:
    def test_mvn_diag_rw_basic(self):
        prop = MVNDiagRW({"beta1": 0.1, "gamma": 0.1})
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
        prop = MVNDiagRW({"beta1": 0.1, "gamma": 0.0})
        assert prop.param_names == ("beta1",)

    def test_mvn_diag_rw_empty_raises(self):
        with pytest.raises(ValueError, match="at least one positive"):
            MVNDiagRW({"a": 0.0, "b": -1.0})

    def test_mvn_rw_basic(self):
        rw_var = np.array([[1.0, 0.0], [0.0, 0.1]])
        prop = MVNRWFull(rw_var, param_names=["beta1", "gamma"])
        assert isinstance(prop, MVNRWFull)
        theta = jnp.array([400.0, 26.0])
        state = prop.init_state(theta)
        new_theta, _ = prop.step(
            state, theta, jax.random.key(0), jnp.int32(1), jnp.int32(0)
        )
        assert not jnp.array_equal(new_theta, theta)

    def test_mvn_rw_nonsquare_raises(self):
        with pytest.raises(ValueError, match="square matrix"):
            MVNRWFull(np.ones((2, 3)), ["a", "b"])

    def test_mvn_rw_size_mismatch_raises(self):
        with pytest.raises(ValueError, match="dimensions must match"):
            MVNRWFull(np.eye(2), ["a", "b", "c"])

    def test_canonicalize_mvn_rw_freezes_missing_parameters(self):
        prop = MVNRWFull(np.array([[1.0]]), ["beta1"]).canonicalize(["beta1", "gamma"])
        theta = jnp.array([400.0, 26.0])
        state = prop.init_state(theta)
        new_theta, _ = prop.step(
            state, theta, jax.random.key(0), jnp.int32(1), jnp.int32(0)
        )
        assert new_theta[0] != theta[0]
        assert new_theta[1] == theta[1]

    def test_mvn_rw_adaptive_basic(self):
        prop = MVNRWAdaptive(rw_sd={"beta1": 1.0, "gamma": 0.1})
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
        prop = MVNRWAdaptive(rw_var=rw_var, param_names=["beta1"])
        theta = jnp.array([400.0])
        state = prop.init_state(theta)
        new_theta, _ = prop.step(
            state, theta, jax.random.key(0), jnp.int32(1), jnp.int32(0)
        )
        assert new_theta[0] != theta[0]

    def test_canonicalize_mvn_rw_adaptive_freezes_missing_parameters(self):
        prop = MVNRWAdaptive(rw_sd={"beta1": 1.0}).canonicalize(["beta1", "gamma"])
        theta = jnp.array([400.0, 26.0])
        state = prop.init_state(theta)
        new_theta, _ = prop.step(
            state, theta, jax.random.key(0), jnp.int32(1), jnp.int32(0)
        )
        assert new_theta[0] != theta[0]
        assert new_theta[1] == theta[1]

    def test_mvn_rw_adaptive_both_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            MVNRWAdaptive(rw_sd={"a": 1.0}, rw_var=np.eye(1), param_names=["a"])

    def test_proposal_jit_scan_compatible(self):
        """All proposals must work inside jax.lax.scan."""
        prop = MVNRWAdaptive(rw_sd={"beta1": 0.5, "gamma": 0.05})
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
    def test_basic_run(self, sir):
        """PMCMC should run and produce a Result."""
        prop = MVNDiagRW({"beta1": 1.0, "gamma": 0.1})
        sir.pmcmc(J=20, M=5, proposal=prop, key=jax.random.key(0))
        res = sir.results_history[-1]
        assert isinstance(res, Result)
        assert res.method == "pmcmc"
        assert res.M == 5
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

    def test_with_dprior(self, sir):
        prop = MVNDiagRW({"gamma": 0.1})
        sir.pmcmc(
            J=20,
            M=3,
            proposal=prop,
            dprior=_informative_dprior,
            key=jax.random.key(1),
        )
        res = _pmcmc_res(sir.results_history[-1])
        log_priors = res.traces_da.sel(variable="log_prior").values
        # Informative prior should not be identically zero.
        assert not np.allclose(log_priors, 0.0)

    def test_flat_prior_default(self, sir):
        prop = MVNDiagRW({"beta1": 1.0})
        sir.pmcmc(J=20, M=3, proposal=prop, key=jax.random.key(2))
        res = _pmcmc_res(sir.results_history[-1])
        log_priors = res.traces_da.sel(variable="log_prior").values
        np.testing.assert_array_equal(log_priors, 0.0)

    def test_acceptance_rate_in_range(self, sir):
        prop = MVNDiagRW({"beta1": 0.01, "gamma": 0.001})
        sir.pmcmc(J=20, M=10, proposal=prop, key=jax.random.key(3))
        res = _pmcmc_res(sir.results_history[-1])
        for r in res.acceptance_rate:
            assert 0.0 <= r <= 1.0

    def test_reproducibility(self, sir_module):
        """Needs two independent runs; uses sir_module directly."""
        model_orig, theta = sir_module
        prop = MVNDiagRW({"beta1": 1.0})

        sir1 = deepcopy(model_orig)
        sir1.results_history.clear()
        sir1.theta = theta
        sir1.pmcmc(J=20, M=5, proposal=prop, key=jax.random.key(99))
        res1 = _pmcmc_res(sir1.results_history[-1])

        sir2 = deepcopy(model_orig)
        sir2.results_history.clear()
        sir2.theta = theta
        sir2.pmcmc(J=20, M=5, proposal=prop, key=jax.random.key(99))
        res2 = _pmcmc_res(sir2.results_history[-1])

        np.testing.assert_array_equal(
            np.asarray(res1.traces_da.values),
            np.asarray(res2.traces_da.values),
        )
        np.testing.assert_array_equal(res1.accepts, res2.accepts)

    def test_to_dataframe(self, sir):
        prop = MVNDiagRW({"beta1": 1.0})
        sir.pmcmc(J=20, M=3, proposal=prop, key=jax.random.key(4))
        df = _pmcmc_res(sir.results_history[-1]).to_dataframe()
        assert "logLik" in df.columns
        assert "log_prior" in df.columns
        assert "theta_idx" in df.columns
        assert "iteration" in df.columns
        assert len(df) == 4  # M + 1

    def test_traces_method(self, sir):
        prop = MVNDiagRW({"beta1": 1.0})
        sir.pmcmc(J=20, M=3, proposal=prop, key=jax.random.key(5))
        df = _pmcmc_res(sir.results_history[-1]).traces()
        assert "method" in df.columns
        assert "theta_idx" in df.columns
        assert (df["method"] == "pmcmc").all()

    def test_print_summary(self, sir, capsys):
        prop = MVNDiagRW({"beta1": 1.0})
        sir.pmcmc(J=20, M=3, proposal=prop, key=jax.random.key(6))
        _pmcmc_res(sir.results_history[-1]).print_summary()
        out = capsys.readouterr().out
        assert "Method: pmcmc" in out
        assert "Number of chains: 1" in out

    def test_theta_updated(self, sir):
        prop = MVNDiagRW({"beta1": 50.0})  # large jumps -> some acceptance
        sir.pmcmc(J=20, M=8, proposal=prop, key=jax.random.key(7))
        res = _pmcmc_res(sir.results_history[-1])

        # self.theta should exactly match the final trace row, regardless of
        # whether the last chain happened to accept any proposals.
        final_row = res.traces_da.isel(theta_idx=0, iteration=-1)
        for name in sir.canonical_param_names:
            assert sir.theta[0][name] == float(final_row.sel(variable=name).values)
        np.testing.assert_allclose(
            sir.theta.logLik,
            np.asarray(res.traces_da.isel(iteration=-1).sel(variable="logLik")),
            rtol=0,
            atol=0,
        )

    def test_input_theta_is_deepcopied_and_unchanged(self, sir):
        theta_input = pp.PompParameters(sir.theta)
        theta_before = pp.PompParameters(theta_input)
        prop = MVNDiagRW({"beta1": 1.0})

        sir.pmcmc(
            J=20,
            M=3,
            proposal=prop,
            theta=theta_input,
            key=jax.random.key(71),
        )
        res = _pmcmc_res(sir.results_history[-1])

        assert theta_input == theta_before
        assert res.theta == theta_before
        assert res.theta is not theta_input

        mutated = theta_input.params()[0]
        mutated["beta1"] = 123.0
        theta_input.set_params(mutated)
        assert res.theta == theta_before

    def test_initial_loglik_matches_independent_pfilter(self, sir):
        run_key = jax.random.key(72)
        prop = MVNDiagRW({"beta1": 1.0})
        J = 20

        sir.pmcmc(J=J, M=2, proposal=prop, key=run_key)
        res = _pmcmc_res(sir.results_history[-1])
        recorded = np.asarray(
            res.traces_da.isel(theta_idx=0, iteration=0).sel(variable="logLik")
        )

        # Reconstruct the exact initial particle-filter key used internally by
        # Pomp.pmcmc -> F.pmcmc -> _pmcmc_internal.
        _, method_key = jax.random.split(run_key)
        chain_key = jax.random.split(method_key, 1)[0]
        _, init_pf_key = jax.random.split(chain_key)
        keys = jnp.asarray([[init_pf_key]])

        assert res.theta is not None
        recomputed = pp.functional.pfilter(
            sir.to_struct(),
            res.theta.to_jax_array(sir.canonical_param_names),
            J,
            keys,
            thresh=0.0,
        )["logLik"]
        np.testing.assert_allclose(
            recorded, np.asarray(recomputed)[0, 0], rtol=0, atol=0
        )

    def test_impossible_prior_rejects_all_proposals(self, sir):
        beta1_idx = sir.canonical_param_names.index("beta1")
        beta1_start = float(sir.theta[0]["beta1"])

        def point_mass_prior(theta_arr):
            return jnp.where(
                jnp.abs(theta_arr[beta1_idx] - beta1_start) < 1e-12,
                0.0,
                -jnp.inf,
            )

        prop = MVNDiagRW({"beta1": 1.0})
        sir.pmcmc(
            J=20,
            M=5,
            proposal=prop,
            dprior=point_mass_prior,
            key=jax.random.key(73),
        )
        res = _pmcmc_res(sir.results_history[-1])
        assert int(res.accepts[0]) == 0
        np.testing.assert_array_equal(
            np.asarray(res.traces_da.sel(variable="beta1")),
            np.full((1, 6), beta1_start),
        )

    def test_functional_pmcmc_accepts_higher_likelihood_proposal(
        self, static_normal_pomp
    ):
        theta_array = static_normal_pomp.theta.to_jax_array(
            static_normal_pomp.canonical_param_names
        )
        proposal = _FixedShiftProposal(
            shift=jnp.asarray([1.0, 0.0]),
            param_names=tuple(static_normal_pomp.canonical_param_names),
        )
        keys = jax.random.split(jax.random.key(74), 1)

        logliks, _, theta_trace, accepts = pp.functional.pmcmc(
            static_normal_pomp.to_struct(),
            theta_array,
            proposal,
            _flat_dprior,
            M=1,
            J=2,
            thresh=0.0,
            keys=keys,
        )

        mu_idx = static_normal_pomp.canonical_param_names.index("mu")
        assert int(np.asarray(accepts)[0]) == 1
        assert np.asarray(theta_trace)[0, 0, mu_idx] == 0.0
        assert np.asarray(theta_trace)[0, 1, mu_idx] == 1.0
        assert np.asarray(logliks)[0, 1] > np.asarray(logliks)[0, 0]

    def test_pmcmc_loglik_matches_analytic_static_normal(self, static_normal_pomp):
        prop = MVNDiagRW({"mu": 0.01})
        static_normal_pomp.pmcmc(J=3, M=1, proposal=prop, key=jax.random.key(75))
        res = _pmcmc_res(static_normal_pomp.results_history[-1])

        recorded = float(
            res.traces_da.isel(theta_idx=0, iteration=0).sel(variable="logLik").values
        )
        expected = float(jax.scipy.stats.norm.logpdf(1.0, loc=0.0, scale=0.1))
        np.testing.assert_allclose(recorded, expected, rtol=1e-6, atol=1e-6)

    def test_with_mvn_rw(self, sir):
        prop = MVNRWFull(np.array([[1.0]]), ["beta1"])
        sir.pmcmc(J=20, M=3, proposal=prop, key=jax.random.key(8))
        res = _pmcmc_res(sir.results_history[-1])
        assert isinstance(res, Result)

    def test_with_adaptive_proposal(self, sir):
        prop = MVNRWAdaptive(
            rw_sd={"beta1": 1.0, "gamma": 0.1},
            scale_start=2,
            shape_start=2,
        )
        sir.pmcmc(J=20, M=8, proposal=prop, key=jax.random.key(9))
        res = _pmcmc_res(sir.results_history[-1])
        assert isinstance(res, Result)


# ---------------------------------------------------------------
# Multi-chain tests
# ---------------------------------------------------------------


class TestPMCMCMultiChain:
    def test_three_chains(self, sir):
        t1 = dict(sir.theta[0])
        t2 = dict(sir.theta[0])
        t2["beta1"] = 380.0
        t3 = dict(sir.theta[0])
        t3["beta1"] = 420.0
        prop = MVNDiagRW({"beta1": 1.0, "gamma": 0.1})
        sir.pmcmc(
            J=20,
            M=5,
            proposal=prop,
            theta=pp.PompParameters([t1, t2, t3]),
            key=jax.random.key(11),
        )
        res = _pmcmc_res(sir.results_history[-1])
        assert res.n_chains == 3
        assert res.accepts.shape == (3,)
        assert res.traces_da.sizes["theta_idx"] == 3

    def test_chains_independent_keys(self, sir):
        """Different chain start values + same key split should give different traces."""
        t1 = dict(sir.theta[0])
        t2 = dict(sir.theta[0])
        t2["beta1"] = 380.0
        prop = MVNDiagRW({"beta1": 1.0})
        sir.pmcmc(
            J=20,
            M=3,
            proposal=prop,
            theta=pp.PompParameters([t1, t2]),
            key=jax.random.key(12),
        )
        res = _pmcmc_res(sir.results_history[-1])
        c0_lls = res.traces_da.isel(theta_idx=0).sel(variable="logLik").values
        c1_lls = res.traces_da.isel(theta_idx=1).sel(variable="logLik").values
        # Initial logLiks differ because starting theta differs.
        assert c0_lls[0] != c1_lls[0]


# ---------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------


class TestPMCMCValidation:
    def test_invalid_J(self, sir):
        prop = MVNDiagRW({"beta1": 1.0})
        with pytest.raises(ValueError, match="J must be"):
            sir.pmcmc(J=0, M=3, proposal=prop, key=jax.random.key(0))

    def test_invalid_M(self, sir):
        prop = MVNDiagRW({"beta1": 1.0})
        with pytest.raises(ValueError, match="M must be"):
            sir.pmcmc(J=20, M=0, proposal=prop, key=jax.random.key(0))


# ---------------------------------------------------------------
# Merge tests
# ---------------------------------------------------------------


class TestPMCMCMerge:
    def test_merge_two_chains(self, sir_module):
        """Needs two independent runs; uses sir_module directly."""
        model_orig, theta = sir_module
        prop = MVNDiagRW({"beta1": 1.0})

        sir1 = deepcopy(model_orig)
        sir1.results_history.clear()
        sir1.theta = theta
        sir1.pmcmc(J=20, M=3, proposal=prop, key=jax.random.key(20))
        res1 = _pmcmc_res(sir1.results_history[-1])

        sir2 = deepcopy(model_orig)
        sir2.results_history.clear()
        sir2.theta = theta
        sir2.pmcmc(J=20, M=3, proposal=prop, key=jax.random.key(21))
        res2 = _pmcmc_res(sir2.results_history[-1])

        merged = Result.merge(res1, res2)
        assert merged.n_chains == res1.n_chains + res2.n_chains
        assert merged.M == 3  # unchanged
        assert merged.accepts.shape == (merged.n_chains,)
        # Variable coord preserved.
        assert list(merged.traces_da.coords["variable"].values) == list(
            res1.traces_da.coords["variable"].values
        )

    def test_merge_different_M_raises(self, sir_module):
        """Needs two runs with different M."""
        model_orig, theta = sir_module
        prop = MVNDiagRW({"beta1": 1.0})

        sir1 = deepcopy(model_orig)
        sir1.results_history.clear()
        sir1.theta = theta
        sir1.pmcmc(J=20, M=3, proposal=prop, key=jax.random.key(30))
        res1 = _pmcmc_res(sir1.results_history[-1])

        sir2 = deepcopy(model_orig)
        sir2.results_history.clear()
        sir2.theta = theta
        sir2.pmcmc(J=20, M=4, proposal=prop, key=jax.random.key(31))
        res2 = _pmcmc_res(sir2.results_history[-1])

        with pytest.raises(ValueError, match="same M"):
            Result.merge(res1, res2)

    def test_merge_empty_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            Result.merge()
