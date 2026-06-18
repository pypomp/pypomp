import pandas as pd
import pypomp as pp
import numpy as np
import jax.numpy as jnp
from typing import cast
from pypomp.types import ParamDict


def dummy_to_est(theta: ParamDict) -> ParamDict:
    return {k: jnp.array(v + 1.0) for k, v in theta.items()}


def dummy_from_est(theta: ParamDict) -> ParamDict:
    return {k: jnp.array(v - 1.0) for k, v in theta.items()}


def test_ParTrans_to_est_panel():
    """Test that the panel transform correctly transforms the parameters for shared and unit-specific parameters."""

    def to_est(theta: ParamDict) -> ParamDict:
        return {
            "pos_param_shared": jnp.log(theta["pos_param_shared"]),
            "standard_param_shared": theta["standard_param_shared"],
            "pos_param_unit": jnp.log(theta["pos_param_unit"]),
            "standard_param_unit": theta["standard_param_unit"],
        }

    def from_est(theta: ParamDict) -> ParamDict:
        return {
            "pos_param_shared": jnp.exp(theta["pos_param_shared"]),
            "standard_param_shared": theta["standard_param_shared"],
            "pos_param_unit": jnp.exp(theta["pos_param_unit"]),
            "standard_param_unit": theta["standard_param_unit"],
        }

    par_trans = pp.ParTrans(to_est, from_est)

    shared = pd.DataFrame(
        index=pd.Index(["pos_param_shared", "standard_param_shared"])
    ).assign(shared=[5.0, 6.0])
    unit_specific = pd.DataFrame(
        index=pd.Index(["pos_param_unit", "standard_param_unit"])
    ).assign(unit1=[1.0, 2.0], unit2=[3.0, 4.0])
    theta = {
        "shared": shared,
        "unit_specific": unit_specific,
    }
    theta_out = par_trans._panel_transform(
        cast(dict[str, pd.DataFrame | None], theta), direction="to_est"
    )

    # Test that shared parameters are transformed correctly
    # Shared
    shared = theta_out["shared"]
    assert shared is not None
    assert shared.index.tolist() == ["pos_param_shared", "standard_param_shared"]
    assert shared.shape == (2, 1)
    s_col = shared.columns[0]
    assert abs(cast(float, shared.loc["pos_param_shared", s_col]) - np.log(5.0)) < 1e-10
    assert abs(cast(float, shared.loc["standard_param_shared", s_col]) - 6.0) < 1e-10

    # Unit specific
    unit = theta_out["unit_specific"]
    assert unit is not None
    assert unit.index.tolist() == ["pos_param_unit", "standard_param_unit"]
    assert unit.columns.tolist() == ["unit1", "unit2"]
    assert abs(cast(float, unit.loc["pos_param_unit", "unit1"]) - np.log(1.0)) < 1e-10
    assert abs(cast(float, unit.loc["standard_param_unit", "unit1"]) - 2.0) < 1e-10
    assert abs(cast(float, unit.loc["pos_param_unit", "unit2"]) - np.log(3.0)) < 1e-10
    assert abs(cast(float, unit.loc["standard_param_unit", "unit2"]) - 4.0) < 1e-10


def test_ParTrans_to_est_panel_none_cases():
    """Test that the panel transform correctly handles None cases for shared and unit-specific parameters by creating a None dictionary entry."""

    def to_est(theta: ParamDict) -> ParamDict:
        return {k: jnp.array(v * 2) for k, v in theta.items()}

    def from_est(theta: ParamDict) -> ParamDict:
        return {k: jnp.array(v / 2) for k, v in theta.items()}

    par_trans = pp.ParTrans(to_est, from_est)

    # Test both None
    theta_out = par_trans._panel_transform(
        cast(dict[str, pd.DataFrame | None], {}), direction="to_est"
    )
    assert theta_out == {"shared": None, "unit_specific": None}

    # Test shared only
    shared = pd.DataFrame(index=pd.Index(["param1"])).assign(shared=[5.0])
    theta_out = par_trans._panel_transform(
        cast(dict[str, pd.DataFrame | None], {"shared": shared, "unit_specific": None}),
        direction="to_est",
    )
    assert theta_out["shared"] is not None
    assert theta_out["unit_specific"] is None
    assert (
        abs(
            cast(
                float, theta_out["shared"].loc["param1", theta_out["shared"].columns[0]]
            )
            - 10.0
        )
        < 1e-10
    )

    # Test unit-specific only
    unit_specific = pd.DataFrame(index=pd.Index(["param2"])).assign(unit1=[3.0])
    theta_out = par_trans._panel_transform(
        cast(
            dict[str, pd.DataFrame | None],
            {"shared": None, "unit_specific": unit_specific},
        ),
        direction="to_est",
    )
    assert theta_out["shared"] is None
    assert theta_out["unit_specific"] is not None
    assert (
        abs(cast(float, theta_out["unit_specific"].loc["param2", "unit1"]) - 6.0)
        < 1e-10
    )


def test_panel_transform_from_est():
    """Test panel transform in the from_est direction."""

    def to_est(theta: ParamDict) -> ParamDict:
        return {k: jnp.array(v * 2.0) for k, v in theta.items()}

    def from_est(theta: ParamDict) -> ParamDict:
        return {k: jnp.array(v / 2.0) for k, v in theta.items()}

    par_trans = pp.ParTrans(to_est, from_est)

    shared = pd.DataFrame(index=pd.Index(["param1"])).assign(shared=[10.0])
    unit_specific = pd.DataFrame(index=pd.Index(["param2"])).assign(unit1=[20.0])

    theta: dict[str, pd.DataFrame | None] = {
        "shared": shared,
        "unit_specific": unit_specific,
    }

    # from_est: 10 -> 5, 20 -> 10
    theta_out = par_trans._panel_transform(theta, direction="from_est")

    assert theta_out["shared"] is not None
    assert theta_out["unit_specific"] is not None
    assert abs(cast(float, theta_out["shared"].iloc[0, 0]) - 5.0) < 1e-10
    assert abs(cast(float, theta_out["unit_specific"].iloc[0, 0]) - 10.0) < 1e-10


def test_panel_transform_list():
    """Test panel transform list for both directions."""

    def to_est(theta: ParamDict) -> ParamDict:
        return {k: jnp.array(v + 1.0) for k, v in theta.items()}

    def from_est(theta: ParamDict) -> ParamDict:
        return {k: jnp.array(v - 1.0) for k, v in theta.items()}

    par_trans = pp.ParTrans(to_est, from_est)

    theta_list: list[dict[str, pd.DataFrame | None]] = [
        {
            "shared": pd.DataFrame(index=pd.Index(["param1"])).assign(shared=[1.0]),
            "unit_specific": pd.DataFrame(index=pd.Index(["param2"])).assign(
                unit1=[2.0]
            ),
        },
        {
            "shared": pd.DataFrame(index=pd.Index(["param1"])).assign(shared=[3.0]),
            "unit_specific": pd.DataFrame(index=pd.Index(["param2"])).assign(
                unit1=[4.0]
            ),
        },
    ]

    res_to = par_trans._panel_transform_list(theta_list, direction="to_est")
    assert len(res_to) == 2
    shared_to0 = res_to[0]["shared"]
    unit_to0 = res_to[0]["unit_specific"]
    shared_to1 = res_to[1]["shared"]
    unit_to1 = res_to[1]["unit_specific"]
    assert shared_to0 is not None and unit_to0 is not None
    assert shared_to1 is not None and unit_to1 is not None
    assert abs(cast(float, shared_to0.iloc[0, 0]) - 2.0) < 1e-10
    assert abs(cast(float, unit_to0.iloc[0, 0]) - 3.0) < 1e-10
    assert abs(cast(float, shared_to1.iloc[0, 0]) - 4.0) < 1e-10
    assert abs(cast(float, unit_to1.iloc[0, 0]) - 5.0) < 1e-10

    res_from = par_trans._panel_transform_list(theta_list, direction="from_est")
    assert len(res_from) == 2
    shared_from0 = res_from[0]["shared"]
    unit_from0 = res_from[0]["unit_specific"]
    shared_from1 = res_from[1]["shared"]
    unit_from1 = res_from[1]["unit_specific"]
    assert shared_from0 is not None and unit_from0 is not None
    assert shared_from1 is not None and unit_from1 is not None
    assert abs(cast(float, shared_from0.iloc[0, 0]) - 0.0) < 1e-10
    assert abs(cast(float, unit_from0.iloc[0, 0]) - 1.0) < 1e-10
    assert abs(cast(float, shared_from1.iloc[0, 0]) - 2.0) < 1e-10
    assert abs(cast(float, unit_from1.iloc[0, 0]) - 3.0) < 1e-10


def test_to_floats():
    """Test _to_floats conversion and error checking."""

    def to_est(theta: ParamDict) -> ParamDict:
        return {k: jnp.array(v + 1.0) for k, v in theta.items()}

    def from_est(theta: ParamDict) -> ParamDict:
        return {k: jnp.array(v - 1.0) for k, v in theta.items()}

    par_trans = pp.ParTrans(to_est, from_est)

    theta = {"p1": 2.0, "p2": jnp.array(3.0)}

    res_to = par_trans._to_floats(theta, direction="to_est")
    assert isinstance(res_to["p1"], float)
    assert isinstance(res_to["p2"], float)
    assert abs(res_to["p1"] - 3.0) < 1e-10
    assert abs(res_to["p2"] - 4.0) < 1e-10

    res_from = par_trans._to_floats(theta, direction="from_est")
    assert isinstance(res_from["p1"], float)
    assert isinstance(res_from["p2"], float)
    assert abs(res_from["p1"] - 1.0) < 1e-10
    assert abs(res_from["p2"] - 2.0) < 1e-10

    import pytest

    with pytest.raises(ValueError, match="Invalid direction"):
        par_trans._to_floats(theta, direction="invalid")  # type: ignore


def test_partrans_equality():
    """Test __eq__ operator of ParTrans."""

    def f1(t):
        return t

    def f2(t):
        return t

    p1 = pp.ParTrans(f1, f2)
    p2 = pp.ParTrans(f1, f2)
    p3 = pp.ParTrans(f2, f2)
    p4 = pp.ParTrans(f1, f1)

    assert p1 == p2
    assert p1 != p3
    assert p1 != p4
    assert p1 != "not a ParTrans object"


def test_partrans_serialization():
    """Test serialization and deserialization of ParTrans."""
    import pickle

    # 1. Test defaults
    p_default = pp.ParTrans()
    data_default = pickle.dumps(p_default)
    p_loaded_default = pickle.loads(data_default)
    assert p_loaded_default == p_default

    # Test they function correctly
    theta: ParamDict = {"p": 5.0}
    assert p_loaded_default.to_est(theta) == {"p": 5.0}
    assert p_loaded_default.from_est(theta) == {"p": 5.0}

    # 2. Test lambdas/closures (should fall back to defaults)
    p_lambda = pp.ParTrans(lambda x: {"p": x["p"] * 2.0}, lambda x: {"p": x["p"] / 2.0})
    data_lambda = pickle.dumps(p_lambda)
    p_loaded_lambda = pickle.loads(data_lambda)
    # Since they are lambdas, they restore to default functions
    assert p_loaded_lambda.to_est(theta) == {"p": 5.0}
    assert p_loaded_lambda.from_est(theta) == {"p": 5.0}

    # 3. Test module-level functions
    p_module = pp.ParTrans(dummy_to_est, dummy_from_est)
    data_module = pickle.dumps(p_module)
    p_loaded_module = pickle.loads(data_module)

    # Verify module-level functions are restored correctly
    assert p_loaded_module.to_est(cast(ParamDict, {"p": 1.0})) == {"p": 2.0}
    assert p_loaded_module.from_est(cast(ParamDict, {"p": 2.0})) == {"p": 1.0}

    # 4. Test unpickling error fallback
    state = p_module.__getstate__()
    # Corrupt the state to refer to a non-existent module/function
    state["_to_est_module"] = "non_existent_module_foo"
    state["_from_est_name"] = "non_existent_function_bar"

    p_corrupted = pp.ParTrans()
    p_corrupted.__setstate__(state)

    # Should fall back to defaults
    assert p_corrupted.to_est(theta) == {"p": 5.0}
    assert p_corrupted.from_est(theta) == {"p": 5.0}
