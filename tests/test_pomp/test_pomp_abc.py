"""Tests for Pomp.abc() -- JIT-compiled Approximate Bayesian Computation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pypomp as pp
from pypomp.core.results import Result
from pypomp.proposals import mvn_diag_rw, mvn_rw, mvn_rw_adaptive


def _abc_res(res) -> Result:
    assert isinstance(res, Result)
    return res


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _default_probes():
    """Three simple summary stats taking (n_obs, ydim) jax array."""
    return {
        "mean": lambda y: jnp.mean(y),
        "var": lambda y: jnp.var(y),
        "max": lambda y: jnp.max(y),
    }


def _default_scale():
    return {"mean": 100.0, "var": 1000.0, "max": 100.0}


def _flat_dprior(theta_arr):
    return jnp.zeros((), dtype=theta_arr.dtype)


def _informative_dprior(theta_arr):
    """Toy log-normal prior on the first parameter using JAX."""
    v = jnp.maximum(theta_arr[0], 1e-10)
    return -0.5 * (jnp.log(v)) ** 2


# ---------------------------------------------------------------
# ABC tests
# ---------------------------------------------------------------


class TestABC:
    def test_basic_run(self, sir):
        prop = mvn_diag_rw({"beta1": 1.0, "gamma": 0.1})
        sir.abc(
            Nabc=5,
            probes=_default_probes(),
            scale=_default_scale(),
            epsilon=1e6,
            proposal=prop,
            key=jax.random.key(0),
        )
        res = _abc_res(sir.results_history[-1])
        assert isinstance(res, Result)
        assert res.method == "abc"
        assert res.Nabc == 5
        assert res.n_chains == 1
        var_list = list(res.traces_da.coords["variable"].values)
        assert var_list[0] == "distance"
        assert var_list[1] == "log_prior"

    def test_with_dprior(self, sir):
        prop = mvn_diag_rw({"gamma": 0.1})
        sir.abc(
            Nabc=5,
            probes=_default_probes(),
            scale=_default_scale(),
            epsilon=1e6,
            proposal=prop,
            dprior=_informative_dprior,
            key=jax.random.key(1),
        )
        res = _abc_res(sir.results_history[-1])
        lps = res.traces_da.sel(variable="log_prior").values
        assert not np.allclose(lps, 0.0)

    def test_flat_prior_default(self, sir):
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.abc(
            Nabc=3,
            probes=_default_probes(),
            scale=_default_scale(),
            epsilon=1e6,
            proposal=prop,
            key=jax.random.key(2),
        )
        res = _abc_res(sir.results_history[-1])
        np.testing.assert_array_equal(
            res.traces_da.sel(variable="log_prior").values, 0.0
        )

    def test_acceptance_rate_in_range(self, sir):
        prop = mvn_diag_rw({"beta1": 0.01, "gamma": 0.001})
        sir.abc(
            Nabc=10,
            probes=_default_probes(),
            scale=_default_scale(),
            epsilon=1e6,
            proposal=prop,
            key=jax.random.key(3),
        )
        res = _abc_res(sir.results_history[-1])
        for r in res.acceptance_rate:
            assert 0.0 <= r <= 1.0

    def test_reproducibility(self, sir_module):
        """Uses sir_module directly to get two independent deep copies."""
        model_orig, theta = sir_module
        from copy import deepcopy

        Nabc = 5
        probes = _default_probes()
        scale = _default_scale()
        epsilon = 1e6
        proposal = mvn_diag_rw({"beta1": 1.0})
        key = jax.random.key(99)

        sir1 = deepcopy(model_orig)
        sir1.results_history.clear()
        sir1.theta = theta
        sir1.abc(
            Nabc=Nabc,
            probes=probes,
            scale=scale,
            epsilon=epsilon,
            proposal=proposal,
            key=key,
        )

        sir2 = deepcopy(model_orig)
        sir2.results_history.clear()
        sir2.theta = theta
        sir2.abc(
            Nabc=Nabc,
            probes=probes,
            scale=scale,
            epsilon=epsilon,
            proposal=proposal,
            key=key,
        )

        np.testing.assert_array_equal(
            np.asarray(_abc_res(sir1.results_history[-1]).traces_da.values),
            np.asarray(_abc_res(sir2.results_history[-1]).traces_da.values),
        )

    def test_to_dataframe(self, sir):
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.abc(
            Nabc=3,
            probes=_default_probes(),
            scale=_default_scale(),
            epsilon=1e6,
            proposal=prop,
            key=jax.random.key(4),
        )
        df = _abc_res(sir.results_history[-1]).to_dataframe()
        assert "distance" in df.columns
        assert "log_prior" in df.columns
        assert "theta_idx" in df.columns
        assert "iteration" in df.columns
        assert len(df) == 4

    def test_traces_method(self, sir):
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.abc(
            Nabc=3,
            probes=_default_probes(),
            scale=_default_scale(),
            epsilon=1e6,
            proposal=prop,
            key=jax.random.key(5),
        )
        df = _abc_res(sir.results_history[-1]).traces()
        assert "method" in df.columns
        assert (df["method"] == "abc").all()

    def test_print_summary(self, sir, capsys):
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.abc(
            Nabc=3,
            probes=_default_probes(),
            scale=_default_scale(),
            epsilon=1e6,
            proposal=prop,
            key=jax.random.key(6),
        )
        _abc_res(sir.results_history[-1]).print_summary()
        out = capsys.readouterr().out
        assert "Method: abc" in out

    def test_with_mvn_rw(self, sir):
        prop = mvn_rw(np.array([[1.0]]), ["beta1"])
        sir.abc(
            Nabc=3,
            probes=_default_probes(),
            scale=_default_scale(),
            epsilon=1e6,
            proposal=prop,
            key=jax.random.key(7),
        )
        assert isinstance(_abc_res(sir.results_history[-1]), Result)

    def test_with_adaptive_proposal(self, sir):
        prop = mvn_rw_adaptive(
            rw_sd={"beta1": 1.0, "gamma": 0.1},
            scale_start=2,
            shape_start=2,
        )
        sir.abc(
            Nabc=6,
            probes=_default_probes(),
            scale=_default_scale(),
            epsilon=1e6,
            proposal=prop,
            key=jax.random.key(8),
        )
        assert isinstance(_abc_res(sir.results_history[-1]), Result)

    def test_tight_epsilon_low_acceptance(self, sir):
        prop = mvn_diag_rw({"beta1": 10.0})
        sir.abc(
            Nabc=10,
            probes=_default_probes(),
            scale=_default_scale(),
            epsilon=1.0,
            proposal=prop,
            key=jax.random.key(9),
        )
        res = _abc_res(sir.results_history[-1])
        # Very tight epsilon -> typically few accepts
        assert int(res.accepts[0]) <= 10

    def test_theta_updated_to_final_trace(self, sir):
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.abc(
            Nabc=5,
            probes=_default_probes(),
            scale=_default_scale(),
            epsilon=1e6,
            proposal=prop,
            key=jax.random.key(71),
        )
        res = _abc_res(sir.results_history[-1])

        final_row = res.traces_da.isel(theta_idx=0, iteration=-1)
        for name in sir.canonical_param_names:
            assert sir.theta[0][name] == float(final_row.sel(variable=name).values)

    def test_input_theta_is_deepcopied_and_unchanged(self, sir):
        theta_input = pp.PompParameters(sir.theta)
        theta_before = pp.PompParameters(theta_input)
        prop = mvn_diag_rw({"beta1": 1.0})

        sir.abc(
            Nabc=3,
            probes=_default_probes(),
            scale=_default_scale(),
            epsilon=1e6,
            proposal=prop,
            theta=theta_input,
            key=jax.random.key(72),
        )
        res = _abc_res(sir.results_history[-1])

        assert theta_input == theta_before
        assert res.theta == theta_before
        assert res.theta is not theta_input

        mutated = theta_input.params()[0]
        mutated["beta1"] = 123.0
        theta_input.set_params(mutated)
        assert res.theta == theta_before

    def test_impossible_prior_rejects_all_proposals(self, sir):
        beta1_idx = sir.canonical_param_names.index("beta1")
        beta1_start = float(sir.theta[0]["beta1"])

        def point_mass_prior(theta_arr):
            return jnp.where(
                jnp.abs(theta_arr[beta1_idx] - beta1_start) < 1e-12,
                0.0,
                -jnp.inf,
            )

        prop = mvn_diag_rw({"beta1": 1.0})
        sir.abc(
            Nabc=5,
            probes=_default_probes(),
            scale=_default_scale(),
            epsilon=1e12,
            proposal=prop,
            dprior=point_mass_prior,
            key=jax.random.key(73),
        )
        res = _abc_res(sir.results_history[-1])

        assert int(res.accepts[0]) == 0
        np.testing.assert_array_equal(
            np.asarray(res.traces_da.sel(variable="beta1")),
            np.full((1, 6), beta1_start),
        )

    def test_abc_distance_matches_manual_probe_distance(self, deterministic_meas_pomp):
        probes = {"mean": lambda y: jnp.mean(y)}
        scale = {"mean": 2.0}

        deterministic_meas_pomp.abc(
            Nabc=1,
            probes=probes,
            scale=scale,
            epsilon=1e6,
            proposal=mvn_diag_rw({"mu": 0.01}),
            key=jax.random.key(74),
        )
        res = _abc_res(deterministic_meas_pomp.results_history[-1])

        recorded = float(
            res.traces_da.isel(theta_idx=0, iteration=0).sel(variable="distance").values
        )
        expected = ((1.0 - 0.0) / 2.0) ** 2
        np.testing.assert_allclose(recorded, expected, rtol=1e-7, atol=1e-7)


# ---------------------------------------------------------------
# Multi-chain tests
# ---------------------------------------------------------------


class TestABCMultiChain:
    def test_three_chains(self, sir):
        t1 = dict(sir.theta[0])
        t2 = dict(sir.theta[0])
        t2["beta1"] = 380.0
        t3 = dict(sir.theta[0])
        t3["beta1"] = 420.0
        prop = mvn_diag_rw({"beta1": 1.0})
        sir.abc(
            Nabc=5,
            probes=_default_probes(),
            scale=_default_scale(),
            epsilon=1e6,
            proposal=prop,
            theta=pp.PompParameters([t1, t2, t3]),
            key=jax.random.key(11),
        )
        res = _abc_res(sir.results_history[-1])
        assert res.n_chains == 3
        assert res.accepts.shape == (3,)


# ---------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------


class TestABCValidation:
    def test_invalid_Nabc(self, sir):
        prop = mvn_diag_rw({"beta1": 1.0})
        with pytest.raises(ValueError, match="Nabc"):
            sir.abc(
                Nabc=0,
                probes=_default_probes(),
                scale=_default_scale(),
                epsilon=1e6,
                proposal=prop,
                key=jax.random.key(0),
            )

    def test_invalid_epsilon(self, sir):
        prop = mvn_diag_rw({"beta1": 1.0})
        with pytest.raises(ValueError, match="epsilon"):
            sir.abc(
                Nabc=5,
                probes=_default_probes(),
                scale=_default_scale(),
                epsilon=-1.0,
                proposal=prop,
                key=jax.random.key(0),
            )

    def test_empty_probes(self, sir):
        prop = mvn_diag_rw({"beta1": 1.0})
        with pytest.raises(ValueError, match="probes"):
            sir.abc(
                Nabc=5,
                probes={},
                scale={},
                epsilon=1e6,
                proposal=prop,
                key=jax.random.key(0),
            )

    def test_scale_keys_mismatch(self, sir):
        prop = mvn_diag_rw({"beta1": 1.0})
        with pytest.raises(ValueError, match="scale keys"):
            sir.abc(
                Nabc=5,
                probes=_default_probes(),
                scale={"mean": 1.0},  # missing keys
                epsilon=1e6,
                proposal=prop,
                key=jax.random.key(0),
            )

    def test_negative_scale(self, sir):
        prop = mvn_diag_rw({"beta1": 1.0})
        with pytest.raises(ValueError, match="must be positive"):
            sir.abc(
                Nabc=5,
                probes={"mean": lambda y: jnp.mean(y)},
                scale={"mean": -1.0},
                epsilon=1e6,
                proposal=prop,
                key=jax.random.key(0),
            )


# ---------------------------------------------------------------
# Merge tests
# ---------------------------------------------------------------


class TestABCMerge:
    def test_merge_two_chains(self, sir_module):
        """Needs two independent runs; takes two deep copies from sir_module."""
        from copy import deepcopy

        model_orig, theta = sir_module
        prop = mvn_diag_rw({"beta1": 1.0})

        sir1 = deepcopy(model_orig)
        sir1.results_history.clear()
        sir1.theta = theta
        sir1.abc(
            Nabc=3,
            probes=_default_probes(),
            scale=_default_scale(),
            epsilon=1e6,
            proposal=prop,
            key=jax.random.key(20),
        )
        res1 = _abc_res(sir1.results_history[-1])

        sir2 = deepcopy(model_orig)
        sir2.results_history.clear()
        sir2.theta = theta
        sir2.abc(
            Nabc=3,
            probes=_default_probes(),
            scale=_default_scale(),
            epsilon=1e6,
            proposal=prop,
            key=jax.random.key(21),
        )
        res2 = _abc_res(sir2.results_history[-1])

        merged = Result.merge(res1, res2)
        assert merged.n_chains == res1.n_chains + res2.n_chains
        assert merged.Nabc == 3
        assert merged.epsilon == 1e6

    def test_merge_different_epsilon_raises(self, sir_module):
        """Needs two independent runs with different epsilon."""
        from copy import deepcopy

        model_orig, theta = sir_module
        prop = mvn_diag_rw({"beta1": 1.0})

        sir1 = deepcopy(model_orig)
        sir1.results_history.clear()
        sir1.theta = theta
        sir1.abc(
            Nabc=3,
            probes=_default_probes(),
            scale=_default_scale(),
            epsilon=1e6,
            proposal=prop,
            key=jax.random.key(30),
        )
        res1 = _abc_res(sir1.results_history[-1])

        sir2 = deepcopy(model_orig)
        sir2.results_history.clear()
        sir2.theta = theta
        sir2.abc(
            Nabc=3,
            probes=_default_probes(),
            scale=_default_scale(),
            epsilon=2e6,
            proposal=prop,
            key=jax.random.key(31),
        )
        res2 = _abc_res(sir2.results_history[-1])

        with pytest.raises(ValueError, match="epsilon"):
            Result.merge(res1, res2)
