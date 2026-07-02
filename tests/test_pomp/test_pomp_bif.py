"""Tests for Pomp.bif() -- Bayesian iterated filtering."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pypomp as pp
from pypomp.core.algorithms.bif import _bif_perfilter_internal
from pypomp.core.results import PompBIFResult


def _get_sir():
    return pp.models.sir(seed=42)


def _perturb_sd(pomp, value=0.01):
    sigmas = {p: 0.0 for p in pomp.canonical_param_names}
    sigmas["beta1"] = value
    sigmas["gamma"] = value
    return pp.RWSigma(sigmas)


def _zero_dprior(theta_arr):
    return jnp.zeros((), dtype=theta_arr.dtype)


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
            theta=pp.PompParameters([theta1, theta2]),
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

    def test_covariate_indexing_regression(self):
        """BIF must align obs_idx, nstep_array, times, ys, and covars_t."""

        def rinit(thetas, keys, covars0, t0, should_trans):
            return jnp.zeros((thetas.shape[0], 1), dtype=thetas.dtype)

        def rproc(
            particlesF,
            thetas,
            keys,
            covars_extended,
            dt_array_extended,
            t,
            t_idx,
            nstep,
            accumvars,
            should_trans,
        ):
            next_t_idx = t_idx + nstep
            return (
                jnp.full_like(particlesF, next_t_idx.astype(particlesF.dtype)),
                next_t_idx,
            )

        def dmeas(y, particlesP, thetas, covars_t, t, should_trans):
            return y[0] + 10.0 * t + 100.0 * covars_t[0] + 1000.0 * particlesP[:, 0]

        _, neg_loglik = _bif_perfilter_internal(
            m=0,
            thetas_Jd=jnp.zeros((1, 1)),
            key=jax.random.key(4),
            dt_array_extended=jnp.ones(6),
            nstep_array=jnp.asarray([1, 2, 3]),
            t0=0.0,
            times=jnp.asarray([1.0, 3.0, 6.0]),
            ys=jnp.asarray([[1.0], [2.0], [3.0]]),
            J=1,
            rw_sigmas=jnp.zeros(1),
            perturb_sigmas=jnp.zeros(1),
            rinitializers=rinit,
            rprocesses_interp=rproc,
            dmeasures=dmeas,
            dprior=_zero_dprior,
            accumvars=None,
            covars_extended=jnp.asarray(
                [[0.0], [10.0], [-999.0], [20.0], [-999.0], [-999.0], [30.0]]
            ),
            thresh=0.0,
            a=1.0,
        )

        # Correct alignment gives:
        # obs0: y=1, t=1, nstep->t_idx=1, covar=10 => 2011
        # obs1: y=2, t=3, nstep->t_idx=3, covar=20 => 5032
        # obs2: y=3, t=6, nstep->t_idx=6, covar=30 => 9063
        assert float(neg_loglik) == -16106.0
