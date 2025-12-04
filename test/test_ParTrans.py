import pandas as pd
import pypomp as pp
import jax
import jax.numpy as jnp
from typing import cast


def test_ParTrans_to_est_panel():
    """Test that the panel transform correctly transforms the parameters for shared and unit-specific parameters."""

    def to_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return {
            "pos_param_shared": jnp.log(theta["pos_param_shared"]),
            "standard_param_shared": theta["standard_param_shared"],
            "pos_param_unit": jnp.log(theta["pos_param_unit"]),
            "standard_param_unit": theta["standard_param_unit"],
        }

    def from_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
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
    theta_out = par_trans.panel_transform(
        cast(dict[str, pd.DataFrame | None], theta), direction="to_est"
    )

    # Test that shared parameters are transformed correctly
    # Shared
    shared = theta_out["shared"]
    assert shared is not None
    assert shared.index.tolist() == ["pos_param_shared", "standard_param_shared"]
    assert shared.shape == (2, 1)
    s_col = shared.columns[0]
    assert abs(shared.loc["pos_param_shared", s_col] - jnp.log(5.0)) < 1e-10
    assert abs(shared.loc["standard_param_shared", s_col] - 6.0) < 1e-10

    # Unit specific
    unit = theta_out["unit_specific"]
    assert unit is not None
    assert unit.index.tolist() == ["pos_param_unit", "standard_param_unit"]
    assert unit.columns.tolist() == ["unit1", "unit2"]
    assert abs(unit.loc["pos_param_unit", "unit1"] - jnp.log(1.0)) < 1e-10
    assert abs(unit.loc["standard_param_unit", "unit1"] - 2.0) < 1e-10
    assert abs(unit.loc["pos_param_unit", "unit2"] - jnp.log(3.0)) < 1e-10
    assert abs(unit.loc["standard_param_unit", "unit2"] - 4.0) < 1e-10


def test_ParTrans_to_est_panel_none_cases():
    """Test that the panel transform correctly handles None cases for shared and unit-specific parameters by creating a None dictionary entry."""

    def to_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return {k: jnp.array(v * 2) for k, v in theta.items()}

    def from_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return {k: jnp.array(v / 2) for k, v in theta.items()}

    par_trans = pp.ParTrans(to_est, from_est)

    # Test both None
    theta_out = par_trans.panel_transform(
        cast(dict[str, pd.DataFrame | None], {}), direction="to_est"
    )
    assert theta_out == {"shared": None, "unit_specific": None}

    # Test shared only
    shared = pd.DataFrame(index=pd.Index(["param1"])).assign(shared=[5.0])
    theta_out = par_trans.panel_transform(
        cast(dict[str, pd.DataFrame | None], {"shared": shared, "unit_specific": None}),
        direction="to_est",
    )
    assert theta_out["shared"] is not None
    assert theta_out["unit_specific"] is None
    assert (
        abs(theta_out["shared"].loc["param1", theta_out["shared"].columns[0]] - 10.0)
        < 1e-10
    )

    # Test unit-specific only
    unit_specific = pd.DataFrame(index=pd.Index(["param2"])).assign(unit1=[3.0])
    theta_out = par_trans.panel_transform(
        cast(
            dict[str, pd.DataFrame | None],
            {"shared": None, "unit_specific": unit_specific},
        ),
        direction="to_est",
    )
    assert theta_out["shared"] is None
    assert theta_out["unit_specific"] is not None
    assert abs(theta_out["unit_specific"].loc["param2", "unit1"] - 6.0) < 1e-10
