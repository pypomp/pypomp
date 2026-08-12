"""Equivalence between the object-oriented and functional entry points.

Comparisons are exact: both paths run the same compiled kernel, so any
difference is a real divergence and not float noise.
"""

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pypomp as pp
import pypomp.functional as F

SEED = 20260812
J = 6
M = 2


def _derive_new_key(key: jax.Array) -> jax.Array:
    """Reproduce the key a Pomp method uses internally.

    ``Pomp._update_fresh_key`` splits the supplied key and hands the second half
    to the algorithm, keeping the first as the next ``fresh_key``.
    """
    _, new_key = jax.random.split(key)
    return new_key


def _trace(result, variable: str) -> np.ndarray:
    """Pull one variable out of a packed (theta_idx, iteration, variable) trace."""
    return np.asarray(result.payload["traces"].sel(variable=variable))


@pytest.fixture(scope="module")
def lg_module():
    return pp.models.LG(A=np.array([[0.9]]), T=5, key=jax.random.key(0))


@pytest.fixture
def lg(lg_module):
    """Per-test copy: Pomp methods mutate theta and results_history in place."""
    model = deepcopy(lg_module)
    model.results_history.clear()
    return model, model.canonical_param_names


def test_pfilter_parity(lg):
    """Pomp.pfilter matches a hand-built F.pfilter call, including diagnostics."""
    model, param_names = lg
    theta_array = model.theta.to_jax_array(param_names)
    key = jax.random.key(SEED)
    reps = 2

    model.pfilter(J=J, key=key, reps=reps, CLL=True, ESS=True)
    payload = model.results_history[-1].payload

    n_theta = theta_array.shape[0]
    rep_keys = jax.random.split(_derive_new_key(key), n_theta * reps).reshape(
        n_theta, reps
    )
    expected = F.pfilter(
        model.to_struct(), theta_array, J, keys=rep_keys, thresh=0.0, CLL=True, ESS=True
    )

    for name, functional_name in [
        ("logLiks", "logLik"),
        ("CLL", "CLL"),
        ("ESS", "ESS"),
    ]:
        np.testing.assert_array_equal(
            np.asarray(payload[name]),
            np.asarray(expected[functional_name]),
            err_msg=f"{name} diverged between Pomp.pfilter and F.pfilter",
        )


def test_mif_parity(lg):
    """Pomp.mif matches F.mif on both the loglik and the theta traces."""
    model, param_names = lg
    theta_array = model.theta.to_jax_array(param_names)
    key = jax.random.key(SEED)
    rw_sd = pp.RWSigma({name: 0.02 for name in param_names}).geometric_cooling(0.5)

    model.mif(J=J, M=M, rw_sd=rw_sd, key=key)
    result = model.results_history[-1]

    keys = jax.random.split(_derive_new_key(key), theta_array.shape[0])
    theta_3d = jnp.repeat(theta_array[:, jnp.newaxis, :], J, axis=1)
    logliks, theta_traces, _ = F.mif(
        model.to_struct(), theta_3d, rw_sd, M=M, J=J, thresh=0.0, keys=keys
    )

    # Iteration 0 holds the starting theta, so its loglik is NaN by construction.
    packed_loglik = _trace(result, "logLik")
    assert np.all(np.isnan(packed_loglik[:, 0])), "iteration 0 loglik should be NaN"
    np.testing.assert_array_equal(packed_loglik[:, 1:], np.asarray(logliks))

    packed_theta = np.stack([_trace(result, name) for name in param_names], axis=-1)
    np.testing.assert_array_equal(packed_theta, np.asarray(theta_traces))


def test_simulate_parity(lg):
    """Pomp.simulate matches F.simulate, in the long-format row ordering."""
    model, param_names = lg
    theta_array = model.theta.to_jax_array(param_names)
    key = jax.random.key(SEED)
    nsim = 3

    states, obs = model.simulate(nsim=nsim, key=key)

    keys = jax.random.split(_derive_new_key(key), theta_array.shape[0])
    X_sims, Y_sims = F.simulate(
        model.to_struct(),
        theta_array,
        nsim,
        keys,
        times=jnp.array(model.ys.index),
    )

    # _to_long flattens (n_theta, n_sim, n_time, n_feat) row-major, so the
    # observation columns come back in exactly that order.
    obs_values = obs.drop(columns=["theta_idx", "sim", "time"]).to_numpy()
    np.testing.assert_array_equal(
        obs_values, np.asarray(Y_sims).reshape(obs_values.shape)
    )

    state_values = states.drop(columns=["theta_idx", "sim", "time"]).to_numpy()
    np.testing.assert_array_equal(
        state_values, np.asarray(X_sims).reshape(state_values.shape)
    )


def test_train_parity(lg):
    """Pomp.train matches F.train, including the to_est transform it applies."""
    model, param_names = lg
    key = jax.random.key(SEED)
    eta = pp.LearningRate({name: 0.01 for name in param_names})
    optimizer = pp.Adam(scale=False, ls=False, c=0.0, max_ls_itn=1)

    theta_est = deepcopy(model.theta).transformed(model.par_trans, direction="to_est")
    theta_array_est = theta_est.to_jax_array(param_names)

    model.train(J=J, M=M, eta=eta, key=key, optimizer=optimizer, alpha=0.0)
    result = model.results_history[-1]

    keys = jnp.array(jax.random.split(_derive_new_key(key), theta_array_est.shape[0]))
    nLLs, theta_ests = F.train(
        model.to_struct(),
        theta_array_est,
        J,
        optimizer=optimizer,
        M=M,
        eta=eta,
        thresh=0.0,
        alpha=0.0,
        keys=keys,
        alpha_cooling=1.0,
        n_monitors=1,
    )

    np.testing.assert_array_equal(_trace(result, "logLik"), -np.asarray(nLLs))
