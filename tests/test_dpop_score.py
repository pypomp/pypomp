"""DPOP gradient against the exact score of a two-state hidden Markov model.

The chain's sample paths are not differentiable in its transition
probabilities, so MOP alone estimates their score as zero; DPOP should
recover it.  The exact log-likelihood comes from the forward algorithm.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import pypomp as pp
import pypomp.functional as F
from pypomp.random import binomial_logw

T = 10
THETA = {"p01": 0.3, "p10": 0.2, "mu": 1.5}


def _simulate_ys() -> np.ndarray:
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2)
    ys = []
    for _ in range(T):
        p_move = THETA["p10"] if x == 1 else THETA["p01"]
        x = 1 - x if rng.random() < p_move else x
        ys.append(THETA["mu"] * x + rng.normal())
    return np.array(ys)


def _rinit(theta_, key, covars, t0):
    return {"X": jax.random.bernoulli(key).astype(float), "_logw": 0.0}


def _rproc(X_, theta_, key, covars, t, dt):
    x = X_["X"]
    p_move = jnp.where(x == 1.0, theta_["p10"], theta_["p01"])
    move = jax.random.bernoulli(key, jax.lax.stop_gradient(p_move)).astype(float)
    return {
        "X": jnp.where(move == 1.0, 1.0 - x, x),
        "_logw": X_["_logw"] + binomial_logw(move, 1.0, p_move),
    }


def _dmeas(Y_, X_, theta_, covars, t):
    return jax.scipy.stats.norm.logpdf(Y_["y"], theta_["mu"] * X_["X"], 1.0)


def _exact_loglik(theta: jax.Array, ys: np.ndarray) -> jax.Array:
    p01, p10, mu = theta
    P = jnp.array([[1.0 - p01, p01], [p10, 1.0 - p10]])
    a = jnp.array([0.5, 0.5])
    loglik = jnp.array(0.0)
    for y in ys:
        a = (a @ P) * jax.scipy.stats.norm.pdf(y, mu * jnp.array([0.0, 1.0]), 1.0)
        loglik = loglik + jnp.log(a.sum())
        a = a / a.sum()
    return loglik


@pytest.fixture(scope="module")
def two_state_hmm():
    ys = _simulate_ys()
    model = pp.Pomp(
        ys=pd.DataFrame({"y": ys}, index=pd.Index(np.arange(1.0, T + 1))),
        theta=pp.PompParameters(THETA),
        statenames=["X", "_logw"],
        t0=0.0,
        rinit=_rinit,
        rproc=_rproc,
        dmeas=_dmeas,
        nstep=1,
    )
    theta = jnp.array([THETA[name] for name in model.canonical_param_names])
    return model.to_struct(), theta, jax.grad(_exact_loglik)(theta, ys)


def _score_estimates(objective, struct, theta, J=1000, n_reps=200):
    """Per-replicate score estimates at alpha=1, where they are consistent."""
    thetas = jnp.tile(theta, (n_reps, 1))
    keys = jax.random.split(jax.random.key(0), n_reps)
    return -jax.grad(lambda th: objective(struct, th, J, 1.0, keys).sum())(thetas)


def _within_4_se(estimates, exact) -> np.ndarray:
    mean = estimates.mean(axis=0)
    se = estimates.std(axis=0) / np.sqrt(estimates.shape[0])
    return np.asarray(jnp.abs(mean - exact) < 4.0 * se)


def test_dpop_gradient_matches_exact_score(two_state_hmm):
    struct, theta, exact = two_state_hmm

    assert np.all(_within_4_se(_score_estimates(F.pop, struct, theta), exact))


def test_mop_misses_the_transition_score(two_state_hmm):
    """Without the process score, the transition gradients are zero and the
    comparison above fails, so it can detect a missing log-weight term."""
    struct, theta, exact = two_state_hmm
    estimates = _score_estimates(F.mop, struct, theta)

    np.testing.assert_array_equal(estimates[:, :2], 0.0)
    assert not np.any(_within_4_se(estimates, exact)[:2])
