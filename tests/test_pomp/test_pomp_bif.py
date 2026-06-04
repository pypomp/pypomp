"""Tests for Pomp.bif() -- Bayesian iterated filtering."""

import jax
import numpy as np
import pytest

import pypomp as pp
from pypomp.core.results import PompBIFResult


def _get_sir():
    return pp.models.sir(seed=42)


def _perturb_sd(pomp, value=0.01):
    sigmas = {p: 0.0 for p in pomp.canonical_param_names}
    sigmas["beta1"] = value
    sigmas["gamma"] = value
    return pp.RWSigma(sigmas)


class TestBIF:
    def test_basic_run(self):
        sir = _get_sir()
        perturb_sd = _perturb_sd(sir)

        sir.bif(J=4, M=2, perturb_sd=perturb_sd, key=jax.random.key(0))
        res = sir.results_history[-1]

        assert isinstance(res, PompBIFResult)
        assert res.method == "bif"
        assert res.J == 4
        assert res.M == 2
        assert res.n_samples == 4
        assert res.active_params == ["gamma", "beta1"]
        assert res.traces_da.sizes == {
            "theta_idx": 1,
            "iteration": 3,
            "variable": 1 + len(sir.canonical_param_names),
        }
        assert res.cloud_da.sizes == {
            "theta_idx": 1,
            "particle": 4,
            "variable": len(sir.canonical_param_names),
        }
        assert np.isclose(np.asarray(res.weights_da.values).sum(), 1.0)
        assert 0.0 < res.ess <= 4.0

    def test_dataframe_and_summary(self):
        sir = _get_sir()
        perturb_sd = _perturb_sd(sir)

        sir.bif(J=4, M=2, perturb_sd=perturb_sd, key=jax.random.key(1))
        res = sir.results_history[-1]
        assert isinstance(res, PompBIFResult)

        df = res.to_dataframe()
        assert {"sample", "theta_idx", "particle", "weight", "log_Hf"}.issubset(
            df.columns
        )
        assert "beta1" in df.columns
        assert len(df) == 4

        summary = res.weighted_summary()
        assert {"parameter", "mean", "sd", "q0.025", "q0.5", "q0.975"}.issubset(
            summary.columns
        )
        assert "beta1" in set(summary["parameter"])

    def test_multiple_starts_are_pooled(self):
        sir = _get_sir()
        perturb_sd = _perturb_sd(sir)
        theta1 = dict(sir.theta[0])
        theta2 = dict(sir.theta[0])
        theta2["beta1"] = 420.0

        sir.bif(
            J=3,
            M=2,
            perturb_sd=perturb_sd,
            theta=[theta1, theta2],
            key=jax.random.key(2),
        )
        res = sir.results_history[-1]
        assert isinstance(res, PompBIFResult)

        assert res.n_samples == 6
        np.testing.assert_array_equal(
            np.asarray(res.weights_da.coords["theta_idx"]),
            np.array([0, 0, 0, 1, 1, 1]),
        )

    def test_zero_perturbation_raises(self):
        sir = _get_sir()
        perturb_sd = pp.RWSigma({p: 0.0 for p in sir.canonical_param_names})

        with pytest.raises(ValueError, match="At least one perturb_sd"):
            sir.bif(J=4, M=2, perturb_sd=perturb_sd, key=jax.random.key(3))
