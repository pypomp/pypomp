import jax
import jax.numpy as jnp
import pytest
import pypomp as pp


@pytest.fixture(scope="module")
def simple_setup():
    # Set default values for tests
    LG = pp.LG()
    sigmas = 0.02
    sigmas_init = 0.1
    sigmas_long = jnp.array([0.02] * (len(LG.theta[0]) - 1) + [0])
    J = 5
    key = jax.random.key(111)
    a = 0.987
    M = 2
    theta = LG.theta
    return LG, sigmas, sigmas_init, sigmas_long, J, key, a, M, theta


@pytest.fixture(scope="function")
def simple(simple_setup):
    LG, sigmas, sigmas_init, sigmas_long, J, key, a, M, theta = simple_setup
    LG.results_history.clear()
    LG.theta = theta
    return LG, sigmas, sigmas_init, sigmas_long, J, key, a, M


def test_class_mif_basic(simple):
    LG, sigmas, sigmas_init, sigmas_long, J, key, a, M = simple
    for J, M in [(J, 2), (100, 10)]:
        LG.mif(
            J=J,
            M=M,
            sigmas=sigmas,
            sigmas_init=1e-20,
            a=a,
            key=key,
        )
        mif_out1 = LG.results_history[-1]
        traces = mif_out1["traces"]
        # traces is an xarray.DataArray with dims: (replicate, iteration, variable)
        # Check shape for first replicate
        assert traces.sel(replicate=0).shape == (M + 1, len(LG.theta[0]) + 1)
        # +1 for logLik column
        # Check that "logLik" is in variable coordinate
        assert "logLik" in list(traces.coords["variable"].values)
        # Check that all parameter names are in variable coordinate
        for param in LG.theta[0].keys():
            assert param in list(traces.coords["variable"].values)

    # check that sigmas isn't modified by mif
    assert sigmas == 0.02


def test_class_mif_sigmas_array(simple):
    LG, sigmas, sigmas_init, sigmas_long, J, key, a, M = simple
    # check that sigmas array input works
    LG.mif(
        sigmas=sigmas_long,
        sigmas_init=1e-20,
        J=J,
        M=2,
        a=0.9,
        key=key,
    )
    mif_out2 = LG.results_history[-1]
    traces2 = mif_out2["traces"]
    # check that sigmas isn't modified by mif when passed as an array
    assert (sigmas_long == jnp.array([0.02] * (len(LG.theta[0]) - 1) + [0])).all()
    # check that the last parameter is never perturbed (assuming it's the 16th parameter)
    param_names = list(LG.theta[0].keys())
    last_param = param_names[15] if len(param_names) > 15 else param_names[-1]
    last_param_trace = traces2.sel(replicate=0, variable=last_param).values
    assert (last_param_trace == last_param_trace[0]).all()
    # check that some other parameter is perturbed
    first_param = param_names[0]
    first_param_trace = traces2.sel(replicate=0, variable=first_param).values
    assert (first_param_trace != first_param_trace[0]).any()


def test_invalid_mif_input(simple):
    LG, sigmas, sigmas_init, sigmas_long, J, key, a, M = simple
    with pytest.raises(ValueError):
        LG.mif(
            sigmas=sigmas,
            sigmas_init=sigmas_init,
            M=M,
            a=a,
            J=-1,
            key=key,
        )
