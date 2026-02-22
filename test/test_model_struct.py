import pypomp as pp
import jax.numpy as jnp
import jax
import pytest
from pypomp.types import (
    StateDict,
    ParamDict,
    CovarDict,
    TimeFloat,
    StepSizeFloat,
    RNGKey,
    ObservationDict,
    InitialTimeFloat,
)
from pypomp.model_struct import RInit, RProc, DMeas, RMeas


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
            RInit(
                fn,
                statenames=["state_0"],
                param_names=["param_0"],
                covar_names=[],
                par_trans=pp.ParTrans(),
            )
    # Test that correct arguments run without error
    RInit(
        lambda theta_, key, covars, t0: {"state_0": 0},
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        par_trans=pp.ParTrans(),
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
            RProc(
                fn,
                statenames=["state_0"],
                param_names=["param_0"],
                covar_names=[],
                nstep=1,
                par_trans=pp.ParTrans(),
            )
    # Test that correct arguments run without error
    RProc(
        lambda X_, theta_, key, covars, t, dt: {"state_0": 0},
        statenames=["state_0"],
        param_names=["param_0"],
        nstep=1,
        covar_names=[],
        par_trans=pp.ParTrans(),
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
            DMeas(
                fn,
                statenames=["state_0"],
                param_names=["param_0"],
                covar_names=[],
                par_trans=pp.ParTrans(),
            )
    # Test that correct arguments run without error
    DMeas(
        lambda Y_, X_, theta_, covars, t: 0.0,
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        par_trans=pp.ParTrans(),
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
            RMeas(
                fn,
                ydim=1,
                statenames=["state_0"],
                param_names=["param_0"],
                covar_names=[],
                par_trans=pp.ParTrans(),
            )
    # Test that correct arguments run without error
    RMeas(
        lambda X_, theta_, key, covars, t: jnp.array([0]),
        ydim=1,
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        par_trans=pp.ParTrans(),
    )


def test_RInit_type_annotations():
    """Test that RInit works with type annotations and custom argument names."""

    def custom_init(
        p: ParamDict, rng: RNGKey, env: CovarDict, t_start: InitialTimeFloat
    ):  # type: ignore
        return {"state_0": p["param_0"] * 2.0}  # type: ignore

    rinit = RInit(
        custom_init,  # type: ignore
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        par_trans=pp.ParTrans(),
    )

    # Test that it actually works
    key = jax.random.key(42)
    theta_array = jnp.array([1.5])
    result = rinit.struct(theta_array, key, jnp.array([]), 0.0, False)
    assert result.shape == (1,)
    assert result[0] == 3.0


def test_RProc_type_annotations():
    """Test that RProc works with type annotations and custom argument names."""

    def custom_step(
        population: StateDict,
        params: ParamDict,
        rng_key: RNGKey,
        environment: CovarDict,
        current_time: TimeFloat,
        delta_t: StepSizeFloat,
    ):  # type: ignore
        return {"state_0": population["state_0"] + params["param_0"] * delta_t}  # type: ignore

    rproc = RProc(
        custom_step,  # type: ignore
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        nstep=1,
        par_trans=pp.ParTrans(),
    )

    # Test that it actually works
    key = jax.random.key(42)
    X_array = jnp.array([1.0])
    theta_array = jnp.array([2.0])
    result = rproc.struct(X_array, theta_array, key, jnp.array([]), 0.0, 0.5, False)
    assert result.shape == (1,)
    assert result[0] == 2.0  # 1.0 + 2.0 * 0.5


def test_DMeas_type_annotations():
    """Test that DMeas works with type annotations and custom argument names."""

    def custom_dmeas(
        obs: ObservationDict,
        x: StateDict,
        p: ParamDict,
        env: CovarDict,
        now: TimeFloat,
    ):  # type: ignore
        return -0.5 * (obs["y_0"] - x["state_0"]) ** 2  # type: ignore

    dmeas = DMeas(
        custom_dmeas,  # type: ignore
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        y_names=["y_0"],
        par_trans=pp.ParTrans(),
    )

    # Test that it actually works
    Y_array = jnp.array([1.0])
    X_array = jnp.array([1.0])
    theta_array = jnp.array([2.0])
    result = dmeas.struct(Y_array, X_array, theta_array, jnp.array([]), 0.0, False)
    assert result == 0.0


def test_RMeas_type_annotations():
    """Test that RMeas works with type annotations and custom argument names."""

    def custom_rmeas(
        pop: StateDict,
        p: ParamDict,
        k: RNGKey,
        env: CovarDict,
        t: TimeFloat,
    ):  # type: ignore
        return jnp.array([pop["state_0"] * p["param_0"]])  # type: ignore

    rmeas = RMeas(
        custom_rmeas,  # type: ignore
        ydim=1,
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        par_trans=pp.ParTrans(),
    )

    # Test that it actually works
    key = jax.random.key(42)
    X_array = jnp.array([3.0])
    theta_array = jnp.array([2.0])
    result = rmeas.struct(X_array, theta_array, key, jnp.array([]), 0.0, False)
    assert result.shape == (1,)
    assert result[0] == 6.0


def test_type_annotations_backward_compatibility():
    """Test that exact names still work (backward compatibility)."""
    # These should all work without type annotations
    rinit = RInit(
        lambda theta_, key, covars, t0: {"state_0": 0},
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        par_trans=pp.ParTrans(),
    )

    rproc = RProc(
        lambda X_, theta_, key, covars, t, dt: {"state_0": 0},
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        nstep=1,
        par_trans=pp.ParTrans(),
    )

    dmeas = DMeas(
        lambda Y_, X_, theta_, covars, t: 0.0,
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        par_trans=pp.ParTrans(),
    )

    rmeas = RMeas(
        lambda X_, theta_, key, covars, t: jnp.array([0]),
        ydim=1,
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        par_trans=pp.ParTrans(),
    )

    # Just verify they were created successfully
    assert rinit is not None
    assert rproc is not None
    assert dmeas is not None
    assert rmeas is not None


def test_type_annotations_missing_error():
    """Test that missing type annotations with wrong names raises error."""
    # Function with wrong names and no annotations should fail
    with pytest.raises(
        ValueError,
        match=r"Could not map arguments for:.*Use pypomp\.types or exact names\.",
    ):
        RInit(
            lambda wrong_name, key, covars, t0: {"state_0": 0},
            statenames=["state_0"],
            param_names=["param_0"],
            covar_names=[],
            par_trans=pp.ParTrans(),
        )


def test_type_annotations_incomplete_error():
    """Test that incomplete type annotations raise error."""

    # Function with some but not all annotations should fail
    # p: ParamDict maps to theta_, but wrong_key, wrong_covars, wrong_t0 don't match
    def incomplete(p: ParamDict, wrong_key, wrong_covars, wrong_t0):  # type: ignore
        return {"state_0": 0}

    with pytest.raises(
        ValueError,
        match=r"Could not map arguments for:.*Use pypomp\.types or exact names\.",
    ):
        RInit(
            incomplete,  # type: ignore
            statenames=["state_0"],
            param_names=["param_0"],
            covar_names=[],
            par_trans=pp.ParTrans(),
        )


def test_type_annotations_ambiguous_error():
    """Test that multiple parameters with same type annotation raises error."""

    # Function with two Params should fail
    def ambiguous(
        p1: ParamDict,
        p2: ParamDict,
        key: RNGKey,
        covars: CovarDict,
        t0: InitialTimeFloat,
    ):  # type: ignore
        return {"state_0": 0}

    with pytest.raises(ValueError, match="Multiple parameters annotated"):
        RInit(
            ambiguous,  # type: ignore
            statenames=["state_0"],
            param_names=["param_0"],
            covar_names=[],
            par_trans=pp.ParTrans(),
        )


def test_type_annotations_argument_order():
    """Test that argument order doesn't matter with type annotations."""

    # Arguments in different order should still work
    def reordered(
        t_start: InitialTimeFloat,
        env: CovarDict,
        rng: RNGKey,
        p: ParamDict,
    ):  # type: ignore
        return {"state_0": p["param_0"]}  # type: ignore

    rinit = RInit(
        reordered,  # type: ignore
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        par_trans=pp.ParTrans(),
    )

    # Test that it works
    key = jax.random.key(42)
    theta_array = jnp.array([5.0])
    result = rinit.struct(theta_array, key, jnp.array([]), 0.0, False)
    assert result[0] == 5.0


def test_type_annotations_with_covars():
    """Test type annotations work with covariates."""

    def with_covars(
        p: ParamDict,
        k: RNGKey,
        weather: CovarDict,
        t0: InitialTimeFloat,
    ):  # type: ignore
        return {"state_0": p["param_0"] + weather["temp"]}  # type: ignore[attr-defined]

    rinit = RInit(
        with_covars,  # type: ignore
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=["temp"],
        par_trans=pp.ParTrans(),
    )

    # Test that it works
    key = jax.random.key(42)
    theta_array = jnp.array([2.0])
    covars_array = jnp.array([10.0])
    result = rinit.struct(theta_array, key, covars_array, 0.0, False)
    assert result[0] == 12.0
