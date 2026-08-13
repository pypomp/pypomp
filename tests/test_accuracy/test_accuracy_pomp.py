"""Statistical accuracy of the single-unit algorithms against exact results."""

import jax
import numpy as np
import pytest

import pypomp as pp
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
    rw_sd,
    theta_bounds,
)
from tests.helpers.plotting import save_traces_plotnine

pytestmark = pytest.mark.heavy

T = 100
TRUE = {A: 0.8, Q: 0.5, R: 0.3}
START = {A: 0.5, Q: 0.8, R: 0.6}
BOUNDS = {A: (0.1, 1.0), Q: (0.1, 1.0), R: (0.1, 1.0)}

# Hurwicz bias of the autoregressive coefficient at this series length.
HURWICZ = TRUE[A] - (1 + 3 * TRUE[A]) / T


def _fit_model(key):
    """A model on data generated at TRUE, with 5 parameter sets from the box."""
    model = lg_1d(TRUE[A], TRUE[Q], TRUE[R], T=T, key=key)
    ys = lg_1d_ys(model)
    model.theta = pp.Pomp.sample_params(theta_bounds(BOUNDS), n=5, key=key)
    return model, ys


def _mle(ys):
    return lg_mle(ys, ESTIMATED, FIXED, TRUE)


def _mean_theta(model):
    final = model.theta.params(as_list=False).sel(unit="shared").mean(dim="theta_idx")
    return {n: final.sel(parameter=n).item() for n in ESTIMATED}


def _errors(est, mle):
    return {n: abs(est[n] - mle[n]) for n in ESTIMATED}


def _norm(values, mle):
    return float(np.linalg.norm([values[n] - mle[n] for n in ESTIMATED]))


def test_pomp_pfilter_accuracy():
    """The filter's log-likelihood matches the exact Kalman value."""
    key = jax.random.key(1234)
    model = lg_1d(TRUE[A], TRUE[Q], TRUE[R], T=T, key=key)
    exact_ll = lg_1d_loglik(lg_1d_ys(model), TRUE[A], TRUE[Q], TRUE[R])

    model.pfilter(J=5000, key=key, reps=30)
    est_ll = model.theta.logLik.item()

    assert np.abs(est_ll - exact_ll) < 0.225, (
        f"pfilter error: est={est_ll}, exact={exact_ll}"
    )


def test_pomp_mif_accuracy():
    """mif moves the sampled parameter sets toward the exact MLE."""
    key = jax.random.key(1234)
    model, ys = _fit_model(key)

    model.mif(J=3000, M=100, rw_sd=rw_sd(), key=key)

    mle = _mle(ys)
    est = _mean_theta(model)
    err = _errors(est, mle)

    assert err[A] < 0.12
    assert err[Q] < 0.15
    assert err[R] < 0.225
    assert _norm(est, mle) < 0.60 * _norm(START, mle)

    save_traces_plotnine(
        model,
        "tests/plots/pomp_mif_traces.png",
        true_values={**TRUE, "logLik": lg_1d_loglik(ys, TRUE[A], TRUE[Q], TRUE[R])},
        mle_values=mle,
        expected_values={A: HURWICZ},
    )


def test_pomp_train_accuracy():
    """train (MOP gradient ascent) moves the sampled sets toward the MLE."""
    key = jax.random.key(1234)
    model, ys = _fit_model(key)

    eta = pp.LearningRate({n: 0.05 for n in ESTIMATED} | {n: 0.0 for n in FIXED})
    model.train(
        J=1000,
        M=150,
        eta=eta.cosine_decay(0.1, M=150),
        optimizer=pp.Adam(scale=True, beta1=0.8),
        alpha=1.0,
        key=key,
    )

    mle = _mle(ys)
    est = _mean_theta(model)
    err = _errors(est, mle)

    assert err[A] < 0.15
    assert err[Q] < 0.225
    assert err[R] < 0.18
    assert _norm(est, mle) < 0.525 * _norm(START, mle)

    save_traces_plotnine(
        model,
        "tests/plots/pomp_train_traces.png",
        true_values={**TRUE, "logLik": lg_1d_loglik(ys, TRUE[A], TRUE[Q], TRUE[R])},
        mle_values=mle,
        expected_values={A: HURWICZ},
    )
