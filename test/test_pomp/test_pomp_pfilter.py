import jax
import jax.numpy as jnp
import pytest
import numpy as np

import pypomp as pp


@pytest.fixture(scope="function")
def simple():
    LG = pp.LG()
    key = jax.random.key(111)
    J = 3
    return LG, key, J


def test_class_basic_default(simple):
    LG, key, J = simple
    LG.pfilter(J=J, key=key)
    val1 = LG.results_history[-1].logLiks
    assert val1.shape == (1, 1)
    assert jnp.isfinite(val1.item())
    assert val1.dtype == jnp.float32
    assert LG.results_history[-1].CLL is None


def test_reps_default(simple):
    LG, key, J = simple
    theta = LG.theta
    theta_list = [theta[0], {k: v * 2 for k, v in theta[0].items()}]
    LG.pfilter(J=J, key=key, theta=theta_list, reps=2)
    val1 = LG.results_history[-1].logLiks
    assert val1.shape == (2, 2)
    assert LG.results_history[-1].CLL is None


def test_order_of_parameters_consistency(simple):
    # check that the order of parameters in the theta dict does not affect the results
    LG, key, J = simple
    theta_orig = LG.theta[0]

    keys = list(theta_orig.keys())
    reversed_keys = list(reversed(keys))
    theta_reordered = {k: theta_orig[k] for k in reversed_keys}

    LG.pfilter(J=J, key=key, theta=theta_orig)
    loglik_orig = LG.results_history[-1].logLiks

    LG.results_history.clear()
    LG.pfilter(J=J, key=key, theta=theta_reordered)
    loglik_reordered = LG.results_history[-1].logLiks

    assert np.allclose(loglik_orig.data, loglik_reordered.data), (
        "Log-likelihoods differed after reordering theta dict keys:\n"
        f"original: {loglik_orig}\nreordered: {loglik_reordered}"
    )


def test_diagnostics(simple):
    # (theta, reps, expected shape)
    LG, key, J = simple
    theta = LG.theta
    ys = LG.ys
    theta_cases = [
        (theta[0], 1, (1, 1)),
        ([theta[0], {k: v * 2 for k, v in theta[0].items()}], 2, (2, 2)),
    ]

    bool_cases = [
        (False, False, False, False),
        (True, True, False, False),
        (False, False, True, True),
    ]

    for theta, reps, expected_shape in theta_cases:
        for CLL, ESS, filter_mean, prediction_mean in bool_cases:
            LG.results_history.clear()
            LG.pfilter(
                J=J,
                key=key,
                theta=theta,
                reps=reps,
                CLL=CLL,
                ESS=ESS,
                filter_mean=filter_mean,
                prediction_mean=prediction_mean,
            )
            method = LG.results_history[-1].method
            assert method == "pfilter"
            negLogLiks = LG.results_history[-1].logLiks
            negLogLiks_arr = negLogLiks.data
            assert negLogLiks_arr.shape == expected_shape
            assert jnp.all(jnp.isfinite(negLogLiks_arr))
            assert jnp.issubdtype(negLogLiks_arr.dtype, jnp.floating)

            # CLL:
            if CLL:
                condLogLiks = LG.results_history[-1].CLL
                assert condLogLiks is not None
                condLogLiks_arr = condLogLiks.data
                assert condLogLiks_arr.shape == expected_shape + (len(ys),)
                assert jnp.all(jnp.isfinite(condLogLiks_arr))
                assert jnp.issubdtype(condLogLiks_arr.dtype, jnp.floating)
            else:
                assert LG.results_history[-1].CLL is None

            # ESS:
            if ESS:
                ess = LG.results_history[-1].ESS
                assert ess is not None
                ess_arr = ess.data
                assert ess_arr.shape == expected_shape + (len(ys),)
                assert jnp.all(jnp.isfinite(ess_arr))
                assert jnp.issubdtype(ess_arr.dtype, jnp.floating)
                # all elements should be  between 0 and J inclusive
                assert jnp.all((ess_arr >= 0) & (ess_arr <= J))
            else:
                assert LG.results_history[-1].ESS is None

            # filter_mean:
            if filter_mean:
                filt_mean = LG.results_history[-1].filter_mean
                assert filt_mean is not None
                filter_mean_arr = filt_mean.data
                assert filter_mean_arr.shape == expected_shape + (len(ys), 2)
                assert jnp.all(jnp.isfinite(filter_mean_arr))
                assert jnp.issubdtype(filter_mean_arr.dtype, jnp.floating)
            else:
                assert LG.results_history[-1].filter_mean is None

            # prediction_mean:
            if prediction_mean:
                pred_mean = LG.results_history[-1].prediction_mean
                assert pred_mean is not None
                prediction_mean_arr = pred_mean.data
                assert prediction_mean_arr.shape == expected_shape + (len(ys), 2)
                assert jnp.all(jnp.isfinite(prediction_mean_arr))
                assert jnp.issubdtype(prediction_mean_arr.dtype, jnp.floating)
            else:
                assert LG.results_history[-1].prediction_mean is None
