"""Properties that must hold across algorithms and parameter containers.

Each test here targets a class of error rather than one instance: a result that
depends on dict ordering, a transform that does not round-trip, a mutation
contract that silently changes, or an identity that stops holding.
"""

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pypomp as pp

SEED = 20260812
J = 6


@pytest.fixture(scope="module")
def lg_module():
    return pp.models.LG(A=np.array([[0.9]]), T=5, key=jax.random.key(0))


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
    rw_sd = pp.RWSigma({name: 0.02 for name in param_names}).geometric_cooling(0.5)

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
    model = pp.models.sir(times=np.arange(1, 6) / 52.0, seed=11)
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
# so results must not depend on the order the user supplied them in. This
# pattern already exists for simulate/train/dpop_train; pfilter and mif were
# uncovered.
# ---------------------------------------------------------------------------


def test_pfilter_param_order_invariance(lg):
    key = jax.random.key(SEED)
    theta = lg.theta

    lg.pfilter(J=J, key=key, theta=theta)
    baseline = np.asarray(lg.results_history[-1].payload["logLiks"])

    lg.pfilter(J=J, key=key, theta=_reversed_theta(theta))
    permuted = np.asarray(lg.results_history[-1].payload["logLiks"])

    np.testing.assert_array_equal(baseline, permuted)


def test_mif_param_order_invariance(lg):
    key = jax.random.key(SEED)
    theta = lg.theta
    param_names = lg.canonical_param_names
    rw_sd = pp.RWSigma({name: 0.02 for name in param_names}).geometric_cooling(0.5)

    lg.mif(J=J, M=2, rw_sd=rw_sd, key=key, theta=theta, n_monitors=1)
    baseline = lg.results_history[-1].payload["traces"]

    lg.mif(J=J, M=2, rw_sd=rw_sd, key=key, theta=_reversed_theta(theta), n_monitors=1)
    permuted = lg.results_history[-1].payload["traces"]

    for name in ["logLik", *param_names]:
        np.testing.assert_array_equal(
            np.asarray(baseline.sel(variable=name)),
            np.asarray(permuted.sel(variable=name)),
            err_msg=f"mif result depends on parameter dict order for {name}",
        )


# ---------------------------------------------------------------------------
# Parameter transforms
# ---------------------------------------------------------------------------


def test_par_trans_round_trip(lg):
    """from_est(to_est(theta)) recovers theta for the bundled LG model."""
    original = deepcopy(lg.theta).params(as_list=True)

    theta = deepcopy(lg.theta)
    round_tripped = theta.transformed(lg.par_trans, direction="to_est").transformed(
        lg.par_trans, direction="from_est"
    )

    for before, after in zip(original, round_tripped.params(as_list=True), strict=True):
        for name, value in before.items():
            np.testing.assert_allclose(
                float(after[name]), float(value), rtol=1e-5, atol=1e-6
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


# ---------------------------------------------------------------------------
# logmeanexp identities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shift", [-50.0, 0.0, 37.5])
def test_logmeanexp_shift_invariance(shift):
    """logmeanexp(x + c) == logmeanexp(x) + c."""
    x = jnp.array([-3.0, -1.5, 0.25, 2.0, 4.5])

    np.testing.assert_allclose(
        float(pp.maths.logmeanexp(x + shift)),
        float(pp.maths.logmeanexp(x)) + shift,
        rtol=1e-5,
        atol=1e-5,
    )


def test_logmeanexp_matches_direct_computation():
    """On values that do not overflow, it equals the naive formula."""
    x = jnp.array([-1.0, 0.0, 0.5, 1.25])

    np.testing.assert_allclose(
        float(pp.maths.logmeanexp(x)),
        float(jnp.log(jnp.mean(jnp.exp(x)))),
        rtol=1e-6,
        atol=1e-6,
    )


def test_logmeanexp_constant_input():
    """The log-mean-exp of a constant vector is that constant."""
    np.testing.assert_allclose(
        float(pp.maths.logmeanexp(jnp.full((7,), 3.25))), 3.25, rtol=1e-6, atol=1e-6
    )


def test_logmeanexp_survives_large_values():
    """Shift-invariance must hold at magnitudes where naive exp overflows."""
    x = jnp.array([800.0, 801.0, 802.0])

    result = float(pp.maths.logmeanexp(x))
    assert np.isfinite(result), "logmeanexp must not overflow"
    np.testing.assert_allclose(
        result, float(pp.maths.logmeanexp(x - 800.0)) + 800.0, rtol=1e-5, atol=1e-4
    )
