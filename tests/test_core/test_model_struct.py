import pypomp as pp
import jax.numpy as jnp
import jax
import pytest
import numpy as np
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
from pypomp.core.model_struct import _RInit, _RProc, _DMeas, _RMeas, _ModelComponent


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
            _RInit(
                fn,
                statenames=["state_0"],
                param_names=["param_0"],
                covar_names=[],
                par_trans=pp.ParTrans(),
            )
    # Test that correct arguments run without error
    _RInit(
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
            _RProc(
                fn,
                statenames=["state_0"],
                param_names=["param_0"],
                covar_names=[],
                nstep=1,
                par_trans=pp.ParTrans(),
            )
    # Test that correct arguments run without error
    _RProc(
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
            _DMeas(
                fn,
                statenames=["state_0"],
                param_names=["param_0"],
                covar_names=[],
                par_trans=pp.ParTrans(),
            )
    # Test that correct arguments run without error
    _DMeas(
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
            _RMeas(
                fn,
                y_names=["y_0"],
                statenames=["state_0"],
                param_names=["param_0"],
                covar_names=[],
                par_trans=pp.ParTrans(),
            )
    # Test that correct arguments run without error
    _RMeas(
        lambda X_, theta_, key, covars, t: jnp.array([0]),
        y_names=["y_0"],
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        par_trans=pp.ParTrans(),
    )


def test_RInit_type_annotations():
    """Test that _RInit works with type annotations and custom argument names."""

    def custom_init(
        p: ParamDict, rng: RNGKey, env: CovarDict, t_start: InitialTimeFloat
    ):  # type: ignore
        return {"state_0": p["param_0"] * 2.0}  # type: ignore

    rinit = _RInit(
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
    """Test that _RProc works with type annotations and custom argument names."""

    def custom_step(
        population: StateDict,
        params: ParamDict,
        rng_key: RNGKey,
        environment: CovarDict,
        current_time: TimeFloat,
        delta_t: StepSizeFloat,
    ):  # type: ignore
        return {"state_0": population["state_0"] + params["param_0"] * delta_t}  # type: ignore

    rproc = _RProc(
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
    """Test that _DMeas works with type annotations and custom argument names."""

    def custom_dmeas(
        obs: ObservationDict,
        x: StateDict,
        p: ParamDict,
        env: CovarDict,
        now: TimeFloat,
    ):  # type: ignore
        return -0.5 * (obs["y_0"] - x["state_0"]) ** 2  # type: ignore

    dmeas = _DMeas(
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
    """Test that _RMeas works with type annotations and custom argument names."""

    def custom_rmeas(
        pop: StateDict,
        p: ParamDict,
        k: RNGKey,
        env: CovarDict,
        t: TimeFloat,
    ):  # type: ignore
        return jnp.array([pop["state_0"] * p["param_0"]])  # type: ignore

    rmeas = _RMeas(
        custom_rmeas,  # type: ignore
        y_names=["y_0"],
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


# Define lambda patterns for the 4 component types
COMPONENT_SPECS = [
    (_RInit, lambda theta_, key, covars, t0: {"state_0": theta_["param_0"]}),
    (_RProc, lambda X_, theta_, key, covars, t, dt: {"state_0": X_["state_0"]}),
    (_DMeas, lambda Y_, X_, theta_, covars, t: 0.0),
    (_RMeas, lambda X_, theta_, key, covars, t: jnp.array([0])),
]


def make_component(comp_class, func):
    kwargs = {
        "statenames": ["state_0"],
        "param_names": ["param_0"],
        "covar_names": [],
        "par_trans": pp.ParTrans(),
    }
    if comp_class == _RProc:
        kwargs["nstep"] = 1
    elif comp_class == _RMeas:
        kwargs["y_names"] = ["y_0"]
    elif comp_class == _DMeas:
        pass

    return comp_class(func, **kwargs)


@pytest.mark.parametrize("comp_class, backward_func", COMPONENT_SPECS)
def test_type_annotations_backward_compatibility(comp_class, backward_func):
    """Test that exact names still work (backward compatibility)."""
    comp = make_component(comp_class, backward_func)
    assert comp is not None


@pytest.mark.parametrize("comp_class, _", COMPONENT_SPECS)
def test_type_annotations_missing_error(comp_class, _):
    """Test that missing type annotations with wrong names raises error."""
    with pytest.raises(
        ValueError,
        match=r"Could not map arguments for:.*Use pypomp\.types or exact names\.",
    ):
        make_component(comp_class, lambda wrong_name, k, c, t, *args: {"state_0": 0})


@pytest.mark.parametrize("comp_class, _", COMPONENT_SPECS)
def test_type_annotations_incomplete_error(comp_class, _):
    """Test that incomplete type annotations raise error."""

    def incomplete(p: ParamDict, wrong_key, wrong_covars, wrong_t0, *args):  # type: ignore
        return {"state_0": 0}

    with pytest.raises(
        ValueError,
        match=r"Could not map arguments for:.*Use pypomp\.types or exact names\.",
    ):
        make_component(comp_class, incomplete)


@pytest.mark.parametrize("comp_class, _", COMPONENT_SPECS)
def test_type_annotations_ambiguous_error(comp_class, _):
    """Test that multiple parameters with same type annotation raises error."""

    def ambiguous(
        p1: ParamDict,
        p2: ParamDict,
        key: RNGKey,
        covars: CovarDict,
        t0: InitialTimeFloat,
        *args,
    ):  # type: ignore
        return {"state_0": 0}

    with pytest.raises(ValueError, match="Multiple parameters annotated"):
        make_component(comp_class, ambiguous)


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

    rinit = _RInit(
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

    rinit = _RInit(
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


def test_align_by_type_except_exception():
    def custom_func(theta_: "SomeUndefinedType", key, covars, t0):  # type: ignore # noqa: F821
        return {"state_0": 0}

    # Since get_type_hints raises NameError, it should fall back to inspect.signature
    # and since the parameter names are exactly theta_, key, covars, t0, it should succeed.
    rinit = _RInit(
        custom_func,
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        par_trans=pp.ParTrans(),
    )
    assert rinit is not None


def test_align_by_type_rough_type_match():
    class MockType:
        def __eq__(self, other):
            return other is StateDict

    def custom_func(x_custom: MockType(), theta_, key, covars, t, dt):  # type: ignore
        return {"state_0": 0}

    # x_custom has MockType, which doesn't have Annotated origin, but compares equal to StateDict.
    # It should be matched to X_ in Step 2 via rough type match.
    from pypomp.core.model_struct import _align_by_type

    mapping = _align_by_type(custom_func, ["X_", "theta_", "key", "covars", "t", "dt"])
    assert mapping["X_"] == "x_custom"


def test_model_component_list_validation():
    # statenames not a list
    with pytest.raises(ValueError, match="statenames must be a list of strings"):
        _RInit(
            lambda theta_, key, covars, t0: {"state_0": 0},
            statenames="state_0",  # type: ignore
            param_names=["param_0"],
            covar_names=[],
            par_trans=pp.ParTrans(),
        )
    # param_names containing non-strings
    with pytest.raises(ValueError, match="param_names must be a list of strings"):
        _RInit(
            lambda theta_, key, covars, t0: {"state_0": 0},
            statenames=["state_0"],
            param_names=[123],  # type: ignore
            covar_names=[],
            par_trans=pp.ParTrans(),
        )


def test_model_component_equality():
    def func1(theta_, key, covars, t0):
        return {"state_0": theta_["param_0"]}

    def func2(theta_, key, covars, t0):
        return {"state_0": theta_["param_0"] * 2}

    rinit1 = _RInit(
        func1,
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        par_trans=pp.ParTrans(),
    )
    rinit1_dup = _RInit(
        func1,
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        par_trans=pp.ParTrans(),
    )
    rinit2 = _RInit(
        func2,
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        par_trans=pp.ParTrans(),
    )

    assert rinit1 == rinit1_dup
    assert rinit1 != rinit2
    assert rinit1 != "not a component"


def test_rinit_validate_output_non_dict():
    # returning a list instead of dict should raise TypeError
    with pytest.raises(TypeError, match="rinit function must return a dict"):
        _RInit(
            lambda theta_, key, covars, t0: [0.0],  # type: ignore
            statenames=["state_0"],
            param_names=["param_0"],
            covar_names=[],
            par_trans=pp.ParTrans(),
        )


def test_rinit_validate_output_missing_keys():
    with pytest.raises(ValueError, match="rinit function output missing state keys"):
        _RInit(
            lambda theta_, key, covars, t0: {"wrong_state": 0.0},
            statenames=["state_0"],
            param_names=["param_0"],
            covar_names=[],
            par_trans=pp.ParTrans(),
        )


def test_rinit_with_parameter_transform():
    # Create a simple parameter transformation where we transform 'param_0'
    def to_est(theta):
        return {k: (jnp.log(v) if k == "param_0" else v) for k, v in theta.items()}

    def from_est(theta):
        return {k: (jnp.exp(v) if k == "param_0" else v) for k, v in theta.items()}

    trans = pp.ParTrans(to_est=to_est, from_est=from_est)

    rinit = _RInit(
        lambda theta_, key, covars, t0: {"state_0": theta_["param_0"]},
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        par_trans=trans,
    )

    key = jax.random.key(1)
    theta_array = jnp.array([jnp.log(5.0)])

    # should_trans=True will apply from_est which does exp(log(5.0)) -> 5.0
    res = rinit.struct(theta_array, key, jnp.array([]), 0.0, True)
    assert jnp.allclose(res, 5.0)

    # should_trans=False will not transform, so we get log(5.0)
    res_raw = rinit.struct(theta_array, key, jnp.array([]), 0.0, False)
    assert jnp.allclose(res_raw, jnp.log(5.0))


def test_rproc_nstep_dt_exclusive():
    with pytest.raises(ValueError, match="Only nstep or dt can be provided, not both"):
        _RProc(
            lambda X_, theta_, key, covars, t, dt: {"state_0": 0.0},
            statenames=["state_0"],
            param_names=["param_0"],
            covar_names=[],
            nstep=1,
            dt=0.1,
            par_trans=pp.ParTrans(),
        )


def test_rproc_validate_output_non_dict():
    with pytest.raises(TypeError, match="rproc function must return a dict"):
        _RProc(
            lambda X_, theta_, key, covars, t, dt: [0.0],  # type: ignore
            statenames=["state_0"],
            param_names=["param_0"],
            covar_names=[],
            nstep=1,
            par_trans=pp.ParTrans(),
        )


def test_rproc_validate_output_missing_keys():
    with pytest.raises(ValueError, match="rproc function output missing state keys"):
        _RProc(
            lambda X_, theta_, key, covars, t, dt: {"wrong_state": 0.0},
            statenames=["state_0"],
            param_names=["param_0"],
            covar_names=[],
            nstep=1,
            par_trans=pp.ParTrans(),
        )


def test_rproc_interp_and_accumvars():
    # Two states: state_0 (normal), accum_state (accumulated, should be reset to 0 in wrapper)
    # The user function adds 1.0 to state_0 and 2.0 to accum_state at each step
    def step_func(X_, theta_, key, covars, t, dt):
        return {
            "state_0": X_["state_0"] + 1.0,
            "accum_state": X_["accum_state"] + 2.0,
        }

    rproc = _RProc(
        step_func,
        statenames=["state_0", "accum_state"],
        param_names=["param_0"],
        covar_names=["covar_0"],
        nstep=3,
        accumvars=(1,),
        par_trans=pp.ParTrans(),
    )

    X_ = jnp.array([[10.0, 50.0]])  # shape (n_particles, n_states)
    theta_ = jnp.array([1.0])
    keys = jax.random.split(jax.random.key(1), 1)
    covars_extended = jnp.array([[0.1], [0.1], [0.1]])
    dt_array_extended = jnp.array([0.1, 0.1, 0.1])
    t = 0.0
    t_idx = 0

    # Test interpolated run
    new_X, new_t_idx = rproc.struct_pf_interp(
        X_,
        theta_,
        keys,
        covars_extended,
        dt_array_extended,
        t,
        t_idx,
        nstep_dynamic=3,
        accumvars=rproc.accumvars,
        should_trans=False,
    )

    # Check that accum_state was set to 0 initially, and then increased by 2.0 each step for 3 steps -> 6.0
    # state_0 was 10.0, increased by 1.0 each step for 3 steps -> 13.0
    assert jnp.allclose(new_X[0, 0], 13.0)
    assert jnp.allclose(new_X[0, 1], 6.0)
    assert new_t_idx == 3


def test_rproc_nstep_array():
    # If all same
    rproc_same = _RProc(
        lambda X_, theta_, key, covars, t, dt: {"state_0": X_["state_0"]},
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        par_trans=pp.ParTrans(),
        nstep_array=np.array([2, 2, 2]),
    )
    assert rproc_same.nstep == 2

    # If not all same
    rproc_diff = _RProc(
        lambda X_, theta_, key, covars, t, dt: {"state_0": X_["state_0"]},
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        par_trans=pp.ParTrans(),
        nstep_array=np.array([2, 3, 2]),
    )
    assert rproc_diff.nstep is None


def test_rproc_equality():
    def func(X_, theta_, key, covars, t, dt):
        return {"state_0": X_["state_0"]}

    rproc1 = _RProc(
        func,
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        nstep=1,
        accumvars=(0,),
        par_trans=pp.ParTrans(),
    )
    rproc2 = _RProc(
        func,
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        nstep=1,
        accumvars=(0,),
        par_trans=pp.ParTrans(),
    )
    rproc3 = _RProc(
        func,
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        nstep=2,
        accumvars=(0,),
        par_trans=pp.ParTrans(),
    )
    rproc4 = _RProc(
        func,
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        nstep=1,
        accumvars=None,
        par_trans=pp.ParTrans(),
    )

    assert rproc1 == rproc2
    assert rproc1 != rproc3
    assert rproc1 != rproc4


def test_dmeas_validate_output_valid():
    # Python int/float/np.number
    d1 = _DMeas(
        lambda Y_, X_, theta_, covars, t: 1,
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        par_trans=pp.ParTrans(),
    )
    d2 = _DMeas(
        lambda Y_, X_, theta_, covars, t: np.float64(1.5),
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        par_trans=pp.ParTrans(),
    )
    # 0-d JAX array
    d3 = _DMeas(
        lambda Y_, X_, theta_, covars, t: jnp.array(1.5),
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        par_trans=pp.ParTrans(),
    )
    assert d1 is not None
    assert d2 is not None
    assert d3 is not None


def test_dmeas_validate_output_invalid():
    # returning a list raises TypeError
    with pytest.raises(TypeError, match="dmeas function must return a scalar"):
        _DMeas(
            lambda Y_, X_, theta_, covars, t: [1.0],  # type: ignore
            statenames=["state_0"],
            param_names=["param_0"],
            covar_names=[],
            par_trans=pp.ParTrans(),
        )
    # returning a 1-d array raises TypeError
    with pytest.raises(TypeError, match="dmeas function must return a scalar"):
        _DMeas(
            lambda Y_, X_, theta_, covars, t: jnp.array([1.0]),
            statenames=["state_0"],
            param_names=["param_0"],
            covar_names=[],
            par_trans=pp.ParTrans(),
        )


def test_validate_call_exception_wrapping():
    def bad_run(Y_, X_, theta_, covars, t):
        # Accessing an attribute on a float that doesn't exist to raise AttributeError
        x = t.non_existent_method()
        return x

    with pytest.raises(TypeError, match="Error running 'bad_run'"):
        _DMeas(
            bad_run,
            statenames=["state_0"],
            param_names=["param_0"],
            covar_names=[],
            par_trans=pp.ParTrans(),
        )


def test_rmeas_validate_output_non_array():
    with pytest.raises(TypeError, match="rmeas function must return a JAX array"):
        _RMeas(
            lambda X_, theta_, key, covars, t: [1.0],  # type: ignore
            y_names=["y_0"],
            statenames=["state_0"],
            param_names=["param_0"],
            covar_names=[],
            par_trans=pp.ParTrans(),
        )


def test_rmeas_validate_output_scalar_jax_array():
    # When ydim is 1, a 0-d array is acceptable
    rmeas = _RMeas(
        lambda X_, theta_, key, covars, t: jnp.array(1.0),
        y_names=["y_0"],
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        par_trans=pp.ParTrans(),
    )
    assert rmeas is not None


def test_rmeas_validate_output_mismatched_shape():
    with pytest.raises(
        ValueError, match="rmeas function output shape.*does not match ydim"
    ):
        _RMeas(
            lambda X_, theta_, key, covars, t: jnp.array([1.0, 2.0]),
            y_names=["y_0"],
            statenames=["state_0"],
            param_names=["param_0"],
            covar_names=[],
            par_trans=pp.ParTrans(),
        )


def test_base_model_component_validate_output_not_implemented():
    class DummyComponent(_ModelComponent):
        internal_names = ["theta_", "key", "covars", "t0"]
        vmap_axes_pf = (None, 0, None, None, None)
        vmap_axes_per = (0, 0, None, None, None)

        def _make_wrapper(self, user_func):
            return lambda *args: None

    with pytest.raises(NotImplementedError):
        DummyComponent(
            lambda theta_, key, covars, t0: {"state_0": 0.0},
            statenames=["state_0"],
            param_names=["param_0"],
            covar_names=[],
            par_trans=pp.ParTrans(),
            validate_logic=False,
        )._validate_output(None)


def test_base_model_component_make_wrapper_not_implemented():
    class DummyComponent(_ModelComponent):
        internal_names = ["theta_", "key", "covars", "t0"]
        vmap_axes_pf = (None, 0, None, None, None)
        vmap_axes_per = (0, 0, None, None, None)

        def _validate_output(self, result):
            pass

    with pytest.raises(NotImplementedError):
        DummyComponent(
            lambda theta_, key, covars, t0: {"state_0": 0.0},
            statenames=["state_0"],
            param_names=["param_0"],
            covar_names=[],
            par_trans=pp.ParTrans(),
            validate_logic=False,
        )
