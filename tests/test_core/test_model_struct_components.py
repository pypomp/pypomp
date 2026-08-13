"""Per-component behavior: RInit, RProc, DMeas, RMeas, DPrior."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pypomp as pp
from pypomp.core.model_mechanics import (
    _DMeas,
    _DPrior,
    _ModelComponent,
    _RInit,
    _RMeas,
    _RProc,
    _time_interp,
)
from pypomp.types import (
    ParamDict,
)

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
    res = rinit.mechanics(theta_array, key, jnp.array([]), 0.0, True)
    assert jnp.allclose(res, 5.0)

    # should_trans=False will not transform, so we get log(5.0)
    res_raw = rinit.mechanics(theta_array, key, jnp.array([]), 0.0, False)
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
    new_X, new_t_idx = rproc.mechanics_pf_interp(
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


def test_rmeas_validate_output_non_dict():
    with pytest.raises(TypeError, match="rmeas function must return a dict"):
        _RMeas(
            lambda X_, theta_, key, covars, t: [1.0],  # type: ignore
            y_names=["y_0"],
            statenames=["state_0"],
            param_names=["param_0"],
            covar_names=[],
            par_trans=pp.ParTrans(),
        )


def test_rmeas_validate_output_missing_keys():
    with pytest.raises(
        ValueError, match="rmeas function output missing observation keys"
    ):
        _RMeas(
            lambda X_, theta_, key, covars, t: {"y_1": 1.0},
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

        def _make_wrapper(self) -> Callable:
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


def test_DPrior_basic():
    def dprior(params: ParamDict) -> float:
        return float(-0.5 * (params["beta"] - 1.0) ** 2)

    dprior_obj = _DPrior(
        dprior,
        statenames=["S"],
        param_names=["beta"],
        covar_names=[],
        par_trans=pp.ParTrans(),
    )
    val = dprior_obj.mechanics(jnp.array([1.0]), should_trans=False)
    assert float(val) == 0.0

    val_off = dprior_obj.mechanics(jnp.array([2.0]), should_trans=False)
    assert float(val_off) == -0.5


def test_DPrior_value_error():
    def bad_fn(foo):
        return 0.0

    with pytest.raises(ValueError, match="Could not map arguments for"):
        _DPrior(
            bad_fn,
            statenames=["S"],
            param_names=["beta"],
            covar_names=[],
            par_trans=pp.ParTrans(),
        )


def test_DPrior_output_validation():
    def bad_out_fn(theta_):
        return "invalid"

    with pytest.raises(TypeError, match="dprior function must return a scalar"):
        _DPrior(
            bad_out_fn,
            statenames=["S"],
            param_names=["beta"],
            covar_names=[],
            par_trans=pp.ParTrans(),
        )


def test_DPrior_par_trans():
    def dprior(params: ParamDict) -> float:
        return float(params["beta"])

    par_trans = pp.ParTrans(
        to_est=lambda p: {"beta": jnp.log(p["beta"])},
        from_est=lambda p: {"beta": jnp.exp(p["beta"])},
    )
    dprior_obj = _DPrior(
        dprior,
        statenames=["S"],
        param_names=["beta"],
        covar_names=[],
        par_trans=par_trans,
    )
    # in estimation space log(beta)=0.0 -> beta=1.0
    val_trans = dprior_obj.mechanics(jnp.array([0.0]), should_trans=True)
    assert np.isclose(float(val_trans), 1.0)


def test_rproc_struct_with_should_trans():
    """_RProc.mechanics_pf_interp applies the from_est transform when should_trans=True."""

    def step(X_, theta_, key, covars, t, dt):
        return {"state_0": X_["state_0"] + theta_["param_0"] * dt}

    par_trans = pp.ParTrans(
        to_est=lambda p: {"param_0": jnp.log(p["param_0"])},
        from_est=lambda p: {"param_0": jnp.exp(p["param_0"])},
    )
    rproc = _RProc(
        step,
        statenames=["state_0"],
        param_names=["param_0"],
        covar_names=[],
        nstep=1,
        par_trans=par_trans,
    )
    key = jax.random.key(100)
    # theta_arr carries log(2.0) on the estimation scale; from_est maps it back to 2.0.
    result, _ = rproc.mechanics_pf_interp(
        jnp.array([[1.0]]),
        jnp.array([jnp.log(2.0)]),
        jax.random.split(key, 1),
        None,
        jnp.array([0.5]),
        0.0,
        0,
        nstep_dynamic=1,
        accumvars=None,
        should_trans=True,
    )
    assert np.isclose(float(result[0, 0]), 2.0)  # 1.0 + 2.0 * 0.5


def test_time_interp_requires_statenames():
    """_time_interp must be given statenames to build the sub-step loop."""

    def step(X_, theta_, key, covars, t, dt):
        return X_

    with pytest.raises(ValueError, match="statenames are required"):
        _time_interp(step, nstep_fixed=1, statenames=None)  # type: ignore
