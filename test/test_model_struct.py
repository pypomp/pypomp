import pypomp as pp
import jax.numpy as jnp
import pytest


def test_RInit_value_error():
    # Test that an error is thrown with incorrect arguments
    bad_lambdas = [
        lambda foo, key, covars, t0: {"state_0": 0},
        lambda theta_, foo, covars, t0: {"state_0": 0},
        lambda theta_, key, foo, t0: {"state_0": 0},
        lambda theta_, key, covars, foo: {"state_0": 0},
    ]
    for fn in bad_lambdas:
        with pytest.raises(ValueError):
            pp.RInit(
                fn, statenames=["state_0"], param_names=["param_0"], covar_names=[]
            )
    # Test that correct arguments run without error
    pp.RInit(
        lambda theta_, key, covars, t0: {"state_0": 0},
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
    )


def test_RProc_value_error():
    # Test that an error is thrown with incorrect arguments
    bad_lambdas = [
        lambda foo, theta_, key, covars, t, dt: {"state_0": 0},
        lambda X_, foo, key, covars, t, dt: {"state_0": 0},
        lambda X_, theta_, foo, covars, t, dt: {"state_0": 0},
        lambda X_, theta_, key, foo, t, dt: {"state_0": 0},
        lambda X_, theta_, key, covars, foo, dt: {"state_0": 0},
        lambda X_, theta_, key, covars, t, foo: {"state_0": 0},
    ]
    for fn in bad_lambdas:
        with pytest.raises(ValueError):
            pp.RProc(
                fn,
                statenames=["state_0"],
                param_names=["param_0"],
                covar_names=[],
                nstep=1,
            )
    # Test that correct arguments run without error
    pp.RProc(
        lambda X_, theta_, key, covars, t, dt: {"state_0": 0},
        statenames=["state_0"],
        param_names=["param_0"],
        nstep=1,
        covar_names=[],
    )


def test_DMeas_value_error():
    # Test that an error is thrown with incorrect arguments
    bad_lambdas = [
        lambda foo, X_, theta_, covars, t: 0.0,
        lambda Y_, foo, theta_, covars, t: 0.0,
        lambda Y_, X_, foo, covars, t: 0.0,
        lambda Y_, X_, theta_, foo, t: 0.0,
        lambda Y_, X_, theta_, covars, foo: 0.0,
    ]
    for fn in bad_lambdas:
        with pytest.raises(ValueError):
            pp.DMeas(
                fn, statenames=["state_0"], param_names=["param_0"], covar_names=[]
            )
    # Test that correct arguments run without error
    pp.DMeas(
        lambda Y_, X_, theta_, covars, t: 0.0,
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
    )


def test_RMeas_value_error():
    # Test that an error is thrown with incorrect arguments
    bad_lambdas = [
        lambda foo, theta_, key, covars, t: jnp.array([0]),
        lambda X_, foo, key, covars, t: jnp.array([0]),
        lambda X_, theta_, foo, covars, t: jnp.array([0]),
        lambda X_, theta_, key, foo, t: jnp.array([0]),
        lambda X_, theta_, key, covars, foo: jnp.array([0]),
    ]
    for fn in bad_lambdas:
        with pytest.raises(ValueError):
            pp.RMeas(
                fn,
                ydim=1,
                statenames=["state_0"],
                param_names=["param_0"],
                covar_names=[],
            )
    # Test that correct arguments run without error
    pp.RMeas(
        lambda X_, theta_, key, covars, t: jnp.array([0]),
        ydim=1,
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
    )
