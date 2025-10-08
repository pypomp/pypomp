import jax
import pytest

import pypomp as pp


@pytest.fixture(scope="function")
def simple():
    # Set default values for tests
    LG = pp.LG()
    ys = LG.ys
    covars = None
    theta = LG.theta
    J = 2
    key = jax.random.key(111)
    M = 2
    return LG, ys, covars, theta, J, key, M


def test_class_GD_basic(simple):
    LG, ys, covars, theta, J, key, M = simple
    optimizers = ["SGD", "Newton", "WeightedNewton", "BFGS"]
    for optimizer in optimizers:
        LG.train(
            J=J, M=M, optimizer=optimizer, scale=True, ls=False, n_monitors=0, key=key
        )
        GD_out = LG.results_history[-1]
        traces = GD_out["traces"]
        # Check shape for first replicate
        assert traces.sel(replicate=0).shape == (M + 1, len(LG.theta[0]) + 1)
        # +1 for logLik column
        # Check that "logLik" is in variable coordinate
        assert "logLik" in list(traces.coords["variable"].values)
        # Check that all parameter names are in variable coordinate
        for param in LG.theta[0].keys():
            assert param in list(traces.coords["variable"].values)


def test_class_GD_ls(simple):
    LG, ys, covars, theta, J, key, M = simple
    LG.train(J=J, M=M, optimizer="SGD", scale=True, ls=True, n_monitors=1, key=key)
    GD_out = LG.results_history[-1]
    traces = GD_out["traces"]
    assert traces.sel(replicate=0).shape == (M + 1, len(LG.theta[0]) + 1)
    assert "logLik" in list(traces.coords["variable"].values)


def test_invalid_GD_input(simple):
    LG, ys, covars, theta, J, key, M = simple
    with pytest.raises(ValueError):
        # Check that an error is thrown when J is not positive
        LG.train(J=0, M=M, scale=True, ls=True, key=key)
