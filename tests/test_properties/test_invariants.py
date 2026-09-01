"""Properties that must hold across algorithms and parameter containers.

Each test here targets a class of error rather than one instance: a result that
depends on dict ordering, a simulated state that leaves its valid range, or a
mutation contract that silently changes.

Per-model parameter transform round-trips live with their models in
test_models/, and logmeanexp's identities live in test_maths.py.
"""

from copy import deepcopy

import jax
import numpy as np
import pytest

import pypomp as pp
from tests.helpers.params import uniform_rw_sd

SEED = 20260812
J = 6


@pytest.fixture(scope="module")
def lg_module():
    return pp.models.lg(A=np.array([[0.9]]), T=5, key=jax.random.key(0))


@pytest.fixture
def lg(lg_module):
    model = deepcopy(lg_module)
    model.results_history.clear()
    return model


def _reversed_theta(theta):
    """Same parameters, reversed dict insertion order."""
    rows = theta.params(as_list=True)
    rev = list(reversed(list(rows[0].keys())))
    return pp.PompParameters([{k: row[k] for k in rev} for row in rows])


# ---------------------------------------------------------------------------
# Algorithm output invariants
#
# Properties of the results themselves, independent of any locked baseline.
# ---------------------------------------------------------------------------


def test_pfilter_cll_sums_to_loglik(lg):
    """Conditional log-likelihoods must sum to the reported log-likelihood."""
    lg.pfilter(J=J, key=jax.random.key(SEED), CLL=True)
    payload = lg.results_history[-1].payload

    np.testing.assert_allclose(
        np.asarray(payload["CLL"]).sum(axis=-1),
        np.asarray(payload["logLiks"]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_mif_trace_starts_at_initial_theta(lg):
    """Iteration 0 of the mif trace is the parameter vector mif started from."""
    param_names = lg.canonical_param_names
    theta_start = np.asarray(lg.theta.to_jax_array(param_names))
    rw_sd = uniform_rw_sd(param_names, cooling=0.5)

    lg.mif(J=J, M=2, rw_sd=rw_sd, key=jax.random.key(SEED), n_monitors=1)
    traces = lg.results_history[-1].payload["traces"]

    start = np.stack(
        [np.asarray(traces.sel(variable=n))[:, 0] for n in param_names], axis=-1
    )
    np.testing.assert_allclose(start, theta_start, rtol=1e-6, atol=1e-6)


def test_sir_simulate_state_invariants():
    """Counts stay non-negative and finite; accumvars index into the state.

    SIR rather than LG because LG has no accumulator variables and no
    non-negativity constraint to violate.
    """
    model = pp.models.sir(times=np.arange(1, 6) / 52.0, key=jax.random.key(11))
    states, obs = model.simulate(nsim=3, key=jax.random.key(SEED))

    obs_values = obs.drop(columns=["theta_idx", "sim", "time"]).to_numpy()
    assert np.all(np.isfinite(obs_values))
    assert np.all(obs_values >= 0), "simulated SIR case counts must be non-negative"

    # S, I, R and the cases accumulator are counts; W and logw are unconstrained.
    state_values = states.drop(columns=["theta_idx", "sim", "time"])
    counts = state_values[["S", "I", "R", "cases"]].to_numpy()
    assert np.all(np.isfinite(state_values.to_numpy()))
    assert np.all(counts >= 0), "SIR compartments and case accumulator must be >= 0"

    struct = model.to_struct()
    assert struct.accumvars is not None, "SIR model should declare accumvars"
    assert all(0 <= i < len(model.statenames) for i in struct.accumvars)


# ---------------------------------------------------------------------------
# Parameter-order invariance
#
# Parameters are held in dicts and aligned to canonical_param_names internally,
# so results must not depend on the order the user supplied them in.
#
# The dpop_train and panel_mif equivalents stay in their own files: they run a
# different model and a different code path, so they are not instances of this
# same check.
# ---------------------------------------------------------------------------


def _run_pfilter(model, theta):
    model.pfilter(J=J, key=jax.random.key(SEED), theta=theta)
    return np.asarray(model.results_history[-1].payload["logLiks"])


def _run_mif(model, theta):
    rw_sd = uniform_rw_sd(model.canonical_param_names, cooling=0.5)
    model.mif(
        J=J, M=2, rw_sd=rw_sd, key=jax.random.key(SEED), theta=theta, n_monitors=1
    )
    return np.asarray(model.results_history[-1].traces_da.values)


def _run_train(model, theta):
    eta = pp.LearningRate({n: 0.2 for n in model.canonical_param_names})
    model.train(
        J=J,
        M=2,
        eta=eta,
        optimizer=pp.Newton(scale=True),
        key=jax.random.key(SEED),
        theta=theta,
    )
    return np.asarray(model.results_history[-1].traces_da.values)


def _run_simulate(model, theta):
    X_sims, Y_sims = model.simulate(nsim=1, key=jax.random.key(SEED), theta=theta)
    return np.concatenate([X_sims.to_numpy().ravel(), Y_sims.to_numpy().ravel()])


@pytest.mark.parametrize(
    "run",
    [_run_pfilter, _run_mif, _run_train, _run_simulate],
    ids=["pfilter", "mif", "train", "simulate"],
)
def test_param_order_invariance(lg, run):
    theta = lg.theta

    baseline = run(lg, theta)
    lg.results_history.clear()
    permuted = run(lg, _reversed_theta(theta))

    np.testing.assert_allclose(
        baseline,
        permuted,
        atol=1e-7,
        err_msg="result depends on the order theta's keys were supplied in",
    )


# ---------------------------------------------------------------------------
# Mutation contracts
#
# transformed() and pruned() return new objects and leave the receiver alone.
# Pinning that here means a change to copy-vs-mutate semantics fails loudly
# rather than silently altering every caller that reuses a theta afterwards.
# ---------------------------------------------------------------------------


def test_transformed_leaves_receiver_untouched(lg):
    theta = deepcopy(lg.theta)
    before = deepcopy(theta).params(as_list=True)

    copy = theta.transformed(lg.par_trans, direction="to_est")

    assert theta.params(as_list=True) == before, (
        "transformed() must not mutate the receiver"
    )
    assert copy is not theta
    assert copy.params(as_list=True) != before


def test_pruned_leaves_receiver_untouched(lg):
    lg.pfilter(J=J, key=jax.random.key(SEED), reps=3)
    theta = lg.theta
    before = deepcopy(theta).params(as_list=True)

    copy = theta.pruned(n=1)

    assert theta.params(as_list=True) == before, "pruned() must not mutate the receiver"
    assert copy is not theta
    assert copy.num_replicates() <= theta.num_replicates()
