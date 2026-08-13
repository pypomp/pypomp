"""Statistical accuracy of PMCMC and ABC against the exact MLE."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pypomp as pp
from pypomp.types import ParamDict
from tests.helpers.kalman import lg_mle
from tests.helpers.lg_accuracy import (
    A,
    ESTIMATED,
    FIXED,
    Q,
    R,
    lg_1d,
    lg_1d_loglik,
    lg_1d_ys,
)
from tests.helpers.plotting import save_traces_plotnine

pytestmark = pytest.mark.heavy


def _bounded_prior(lower: float, upper: float, sd_floor: float):
    """Flat prior on A11 in (lower, upper) with positive noise scales."""

    def dprior(params: ParamDict) -> float | jax.Array:
        in_bounds = (
            (params[A] > lower)
            & (params[A] < upper)
            & (params[Q] > sd_floor)
            & (params[R] > sd_floor)
        )
        return jnp.where(in_bounds, 0.0, -jnp.inf)

    return dprior


def _chains(start: dict[str, float], n: int) -> pp.PompParameters:
    return pp.PompParameters([{**start, **FIXED} for _ in range(n)])


def _posterior_mean(model, burn_in: int) -> dict[str, float]:
    traces = model.results_history[-1].traces_da.isel(iteration=slice(burn_in, None))
    mean = traces.mean(dim=["theta_idx", "iteration"])
    return {n: mean.sel(variable=n).item() for n in ESTIMATED}


def test_pomp_pmcmc_accuracy():
    """The PMCMC posterior mean sits near the exact MLE."""
    T = 200
    key = jax.random.key(1234)
    true = {A: 0.8, Q: 0.5, R: 0.8}
    start = {A: 0.5, Q: 0.8, R: 0.5}

    model = lg_1d(true[A], true[Q], true[R], T=T, key=key)
    ys = lg_1d_ys(model)
    model.theta = _chains(start, 14)

    model.pmcmc(
        J=3000,
        M=250,
        proposal=pp.MVNDiagRW({A: 0.25, Q: 0.2, R: 0.2}),
        dprior=_bounded_prior(0.0, 1.0, 0.0),
        key=key,
    )

    mle = lg_mle(ys, ESTIMATED, FIXED, true)
    est = _posterior_mean(model, burn_in=100)

    save_traces_plotnine(
        model,
        "tests/plots/pomp_pmcmc_traces.png",
        true_values={**true, "logLik": lg_1d_loglik(ys, true[A], true[Q], true[R])},
        mle_values=mle,
        expected_values={A: true[A] - (1 + 3 * true[A]) / T},
    )

    assert np.abs(est[A] - mle[A]) < 0.15
    assert np.abs(est[Q] - mle[Q]) < 0.15
    assert np.abs(est[R] - mle[R]) < 0.20


def test_pomp_abc_accuracy():
    """The ABC posterior mean sits near the exact MLE."""
    T = 100
    key = jax.random.key(1234)
    true = {A: 0.8, Q: 0.5, R: 0.3}
    start = {A: 0.5, Q: 0.8, R: 0.6}

    model = lg_1d(true[A], true[Q], true[R], T=T, key=key)
    ys = lg_1d_ys(model)
    model.theta = _chains(start, 10)

    obs = model.ys.columns[0]
    probes = {
        "var": lambda y: jnp.var(y[obs]),
        "autocov": lambda y: jnp.mean(y[obs][1:] * y[obs][:-1]),
        "autocov2": lambda y: jnp.mean(y[obs][2:] * y[obs][:-2]),
    }

    model.abc(
        M=30000,
        probes=probes,
        scale={name: 1.0 for name in probes},
        epsilon=0.2,
        proposal=pp.MVNDiagRW({A: 0.16, Q: 0.16, R: 0.16}),
        dprior=_bounded_prior(0.05, 0.95, 0.05),
        key=key,
    )

    mle = lg_mle(ys, ESTIMATED, FIXED, true)
    est = _posterior_mean(model, burn_in=15000)

    save_traces_plotnine(
        model,
        "tests/plots/pomp_abc_traces.png",
        true_values={**true, "logLik": lg_1d_loglik(ys, true[A], true[Q], true[R])},
        mle_values=mle,
    )

    assert np.abs(est[A] - mle[A]) < 0.095
    assert np.abs(est[Q] - mle[Q]) < 0.15
    assert np.abs(est[R] - mle[R]) < 0.09
