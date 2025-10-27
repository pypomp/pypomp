import pandas as pd
import pypomp as pp
import jax
import jax.numpy as jnp


def test_ParTrans_to_est_panel():
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
    shared_out, unit_specific_out = par_trans.panel_transform(
        shared, unit_specific, direction="to_est"
    )

    # Test that shared parameters are transformed correctly
    assert shared_out is not None
    assert shared_out.index.tolist() == ["pos_param_shared", "standard_param_shared"]
    # The DataFrame should have a single column with the shared values
    assert len(shared_out.columns) == 1
    assert (
        abs(shared_out.loc["pos_param_shared", shared_out.columns[0]] - jnp.log(5.0))
        < 1e-10
    )
    assert (
        abs(shared_out.loc["standard_param_shared", shared_out.columns[0]] - 6.0)
        < 1e-10
    )

    # Test that unit-specific parameters are transformed correctly
    assert unit_specific_out is not None
    assert unit_specific_out.index.tolist() == ["pos_param_unit", "standard_param_unit"]
    assert unit_specific_out.columns.tolist() == ["unit1", "unit2"]
    assert abs(unit_specific_out.loc["pos_param_unit", "unit1"] - jnp.log(1.0)) < 1e-10
    assert abs(unit_specific_out.loc["standard_param_unit", "unit1"] - 2.0) < 1e-10
    assert abs(unit_specific_out.loc["pos_param_unit", "unit2"] - jnp.log(3.0)) < 1e-10
    assert abs(unit_specific_out.loc["standard_param_unit", "unit2"] - 4.0) < 1e-10


def test_ParTrans_to_est_panel_none_cases():
    def to_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return {k: jnp.array(v * 2) for k, v in theta.items()}

    def from_est(theta: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return {k: jnp.array(v / 2) for k, v in theta.items()}

    par_trans = pp.ParTrans(to_est, from_est)

    # Test both None
    shared_out, unit_specific_out = par_trans.panel_transform(
        None, None, direction="to_est"
    )
    assert shared_out is None
    assert unit_specific_out is None

    # Test shared only
    shared = pd.DataFrame(index=pd.Index(["param1"])).assign(shared=[5.0])
    shared_out, unit_specific_out = par_trans.panel_transform(
        shared, None, direction="to_est"
    )
    assert shared_out is not None
    assert unit_specific_out is None
    assert abs(shared_out.loc["param1", shared_out.columns[0]] - 10.0) < 1e-10

    # Test unit-specific only
    unit_specific = pd.DataFrame(index=pd.Index(["param2"])).assign(unit1=[3.0])
    shared_out, unit_specific_out = par_trans.panel_transform(
        None, unit_specific, direction="to_est"
    )
    assert shared_out is None
    assert unit_specific_out is not None
    assert abs(unit_specific_out.loc["param2", "unit1"] - 6.0) < 1e-10
