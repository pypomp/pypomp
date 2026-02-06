import jax
import pytest
import numpy as np
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


@pytest.mark.parametrize("optimizer", ["SGD", "Newton", "WeightedNewton", "BFGS"])
def test_train_basic(optimizer, simple):
    """Test basic train functionality with per-parameter learning rates."""
    LG, ys, covars, theta, J, key, M = simple
    eta_dict = {param: 0.2 for param in LG.canonical_param_names}

    LG.train(J=J, M=M, eta=eta_dict, optimizer=optimizer, scale=True, key=key)

    GD_out = LG.results_history[-1]
    traces = GD_out.traces_da
    # Check shape for first replicate
    assert traces.sel(replicate=0).shape == (M + 1, len(LG.theta[0]) + 1)
    # Check that "logLik" and all parameter names are in variable coordinate
    assert "logLik" in list(traces.coords["variable"].values)
    for param in LG.theta.to_list()[0].keys():
        assert param in list(traces.coords["variable"].values)
    assert all(isinstance(v, float) for v in LG.theta[0].values())


def test_train_with_line_search(simple):
    """Test train with line search enabled."""
    LG, ys, covars, theta, J, key, M = simple
    eta_dict = {param: 0.2 for param in LG.canonical_param_names}

    LG.train(
        J=J,
        M=M,
        eta=eta_dict,
        optimizer="SGD",
        scale=True,
        ls=True,
        n_monitors=1,
        key=key,
    )

    GD_out = LG.results_history[-1]
    traces = GD_out.traces_da
    assert traces.sel(replicate=0).shape == (M + 1, len(LG.theta[0]) + 1)
    assert "logLik" in list(traces.coords["variable"].values)


def test_train_validation(simple):
    """Test train input validation."""
    LG, ys, covars, theta, J, key, M = simple
    eta_dict = {param: 0.2 for param in LG.canonical_param_names}

    # Invalid J should raise ValueError
    with pytest.raises(ValueError):
        LG.train(J=0, M=M, eta=eta_dict, scale=True, key=key)

    # Wrong eta keys should raise ValueError
    wrong_eta = {"wrong_param": 0.1, "another_wrong": 0.2}
    with pytest.raises(ValueError, match="eta keys.*must match parameter names"):
        LG.train(J=J, M=M, eta=wrong_eta, key=key)

    # Missing eta keys should raise ValueError
    partial_eta = {LG.canonical_param_names[0]: 0.1}
    with pytest.raises(ValueError, match="eta keys.*must match parameter names"):
        LG.train(J=J, M=M, eta=partial_eta, key=key)


def test_train_param_order_invariance(simple):
    """Test that parameter order doesn't affect results."""
    LG, ys, covars, theta, J, key, M = simple
    eta_dict = {param: 0.2 for param in LG.canonical_param_names}

    LG.train(
        J=J, M=M, eta=eta_dict, optimizer="Newton", scale=True, key=key, theta=theta
    )
    out1 = LG.results_history[-1].traces_da.values

    # Permute theta parameter order
    param_keys = list(theta.to_list()[0].keys())
    rev_keys = list(reversed(param_keys))
    permuted_theta = [{k: th[k] for k in rev_keys} for th in theta]

    LG.train(
        J=J,
        M=M,
        eta=eta_dict,
        optimizer="Newton",
        scale=True,
        key=key,
        theta=permuted_theta,
    )
    out2 = LG.results_history[-1].traces_da.values

    np.testing.assert_allclose(out1, out2, atol=1e-7)


def test_different_learning_rates(simple):
    """Test that different per-parameter learning rates produce different results."""
    LG, ys, covars, theta, J, key, M = simple
    params = LG.canonical_param_names

    # Run with uniform learning rates
    eta_uniform = {param: 0.1 for param in params}
    LG.train(J=J, M=M, eta=eta_uniform, optimizer="SGD", key=key)
    out_uniform = LG.results_history[-1].traces_da.values

    # Run with varied learning rates
    eta_varied = {params[0]: 0.05, params[1]: 0.2}
    for p in params[2:]:
        eta_varied[p] = 0.1

    LG.train(J=J, M=M, eta=eta_varied, optimizer="SGD", key=key)
    out_varied = LG.results_history[-1].traces_da.values

    # Results should differ
    assert not np.allclose(out_uniform, out_varied, atol=1e-10)
