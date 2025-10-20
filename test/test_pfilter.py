import jax
import jax.numpy as jnp
import pytest

import pypomp as pp


@pytest.fixture(scope="function")
def simple():
    LG = pp.LG()
    key = jax.random.key(111)
    J = 3
    ys = LG.ys
    theta = LG.theta
    covars = LG.covars
    rinit = LG.rinit
    rproc = LG.rproc
    dmeas = LG.dmeas
    return LG, key, J, ys, theta, covars, rinit, rproc, dmeas


def test_class_basic_default(simple):
    LG, key, J, ys, theta, covars, rinit, rproc, dmeas = simple
    LG.pfilter(J=J, key=key)
    val1 = LG.results_history[-1]["logLiks"]
    assert val1.shape == (1, 1)
    assert jnp.isfinite(val1.item())
    assert val1.dtype == jnp.float32
    with pytest.raises(KeyError):
        _ = LG.results_history[-1]["CLL"]


def test_reps_default(simple):
    LG, key, J, ys, theta, covars, rinit, rproc, dmeas = simple
    theta_list = [
        theta[0],
        {k: v * 2 for k, v in theta[0].items()},
    ]
    LG.pfilter(J=J, key=key, theta=theta_list, reps=2)
    val1 = LG.results_history[-1]["logLiks"]
    assert val1.shape == (2, 2)
    with pytest.raises(KeyError):
        _ = LG.results_history[-1]["CLL"]


def test_diagnostics(simple):
    # (theta, reps, expected shape)
    LG, key, J, ys, theta, covars, rinit, rproc, dmeas = simple
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
            method = LG.results_history[-1]["method"]
            assert method == "pfilter"
            negLogLiks = LG.results_history[-1]["logLiks"]
            negLogLiks_arr = negLogLiks.data
            assert negLogLiks_arr.shape == expected_shape
            assert jnp.all(jnp.isfinite(negLogLiks_arr))
            assert jnp.issubdtype(negLogLiks_arr.dtype, jnp.floating)

            # CLL:
            if CLL:
                condLogLiks = LG.results_history[-1]["CLL"]
                condLogLiks_arr = condLogLiks.data
                assert condLogLiks_arr.shape == expected_shape + (len(ys),)
                assert jnp.all(jnp.isfinite(condLogLiks_arr))
                assert jnp.issubdtype(condLogLiks_arr.dtype, jnp.floating)
            else:
                with pytest.raises(KeyError):
                    _ = LG.results_history[-1]["CLL"]

                # ESS:
                if ESS:
                    ess = LG.results_history[-1]["ESS"]
                    ess_arr = ess.data
                    assert ess_arr.shape == expected_shape + (len(ys),)
                    assert jnp.all(jnp.isfinite(ess_arr))
                    assert jnp.issubdtype(ess_arr.dtype, jnp.floating)
                    # all elements should be  between 0 and J inclusive
                    assert jnp.all((ess_arr >= 0) & (ess_arr <= J))
                else:
                    with pytest.raises(KeyError):
                        _ = LG.results_history[-1]["ESS"]

                # filter_mean:
                if filter_mean:
                    filt_mean = LG.results_history[-1]["filter_mean"]
                    filter_mean_arr = filt_mean.data
                    assert filter_mean_arr.shape == expected_shape + (len(ys), 2)
                    assert jnp.all(jnp.isfinite(filter_mean_arr))
                    assert jnp.issubdtype(filter_mean_arr.dtype, jnp.floating)
                else:
                    with pytest.raises(KeyError):
                        _ = LG.results_history[-1]["filter_mean"]

                # prediction_mean:
                if prediction_mean:
                    pred_mean = LG.results_history[-1]["prediction_mean"]
                    prediction_mean_arr = pred_mean.data
                    assert prediction_mean_arr.shape == expected_shape + (len(ys), 2)
                    assert jnp.all(jnp.isfinite(prediction_mean_arr))
                    assert jnp.issubdtype(prediction_mean_arr.dtype, jnp.floating)
                else:
                    with pytest.raises(KeyError):
                        _ = LG.results_history[-1]["prediction_mean"]
