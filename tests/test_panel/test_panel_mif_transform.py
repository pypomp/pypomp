"""
Integration tests for parameter transformations in PanelPomp.mif method.
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import pypomp as pp
import pytest
from typing import cast
from pypomp.types import ParamDict


@pytest.fixture
def panel_pomp_with_transform():
    """Create a simple PanelPomp model with custom ParTrans."""
    LG1 = pp.models.LG()
    LG2 = pp.models.LG()

    def to_est(theta: ParamDict) -> ParamDict:
        result = {}
        for k, v in theta.items():
            if k.startswith("Q") or k.startswith("R"):
                result[k] = jnp.log(v)
            else:
                result[k] = v
        return result

    def from_est(theta: ParamDict) -> ParamDict:
        result = {}
        for k, v in theta.items():
            if k.startswith("Q") or k.startswith("R"):
                result[k] = jnp.exp(v)
            else:
                result[k] = v
        return result

    LG1.par_trans = pp.ParTrans(to_est, from_est)
    LG2.par_trans = pp.ParTrans(to_est, from_est)

    theta_base = LG1.theta.params()[0]

    shared_param_names = ["A11", "A12", "A21", "A22", "C11", "C12", "C21", "C22"]
    unit_param_names = ["Q11", "Q12", "Q22", "R11", "R12", "R22"]

    shared_params = pd.DataFrame(
        index=pd.Index(shared_param_names),
        data={"shared": [theta_base[name] for name in shared_param_names]},
    )

    unit_specific_params = pd.DataFrame(
        index=pd.Index(unit_param_names),
        data={
            "unit1": [theta_base[name] * 0.8 for name in unit_param_names],
            "unit2": [theta_base[name] * 1.2 for name in unit_param_names],
        },
    )

    theta = cast(
        list[dict[str, pd.DataFrame | None]],
        [{"shared": shared_params, "unit_specific": unit_specific_params}],
    )
    panel = pp.PanelPomp(
        Pomp_dict={"unit1": LG1, "unit2": LG2},
        theta=pp.PanelParameters(theta),
    )

    return panel


def test_panel_mif_traces_transformed(panel_pomp_with_transform):
    """
    Test that with rw_sd=0, parameters remain unchanged after transformation cycle.
    """
    panel = panel_pomp_with_transform

    panel_theta = panel.theta.params()
    panel_shared = [t.get("shared") for t in panel_theta if t.get("shared") is not None]
    panel_unit_specific = [
        t.get("unit_specific")
        for t in panel_theta
        if t.get("unit_specific") is not None
    ]
    initial_shared = [df.copy() for df in panel_shared] if panel_shared else None
    initial_unit_specific = (
        [df.copy() for df in panel_unit_specific] if panel_unit_specific else None
    )

    shared_names = panel.canonical_shared_param_names
    unit_names = panel.canonical_unit_param_names

    all_param_names = list(shared_names) + list(unit_names)
    rw_sd = pp.RWSigma(
        sigmas={k: 0.0 for k in all_param_names},
        init_names=[],
    ).geometric_cooling(0.5)

    panel.mif(J=2, M=1, rw_sd=rw_sd, key=jax.random.key(42))

    final_panel_theta = panel.theta.params()
    final_shared = [
        t.get("shared") for t in final_panel_theta if t.get("shared") is not None
    ]
    final_unit_specific = [
        t.get("unit_specific")
        for t in final_panel_theta
        if t.get("unit_specific") is not None
    ]
    final_shared = final_shared if final_shared else None
    final_unit_specific = final_unit_specific if final_unit_specific else None

    if initial_shared is not None and final_shared is not None:
        for rep_idx in range(len(final_shared)):
            initial_df = initial_shared[rep_idx]
            final_df = final_shared[rep_idx]

            for param in shared_names:
                initial_val = initial_df.loc[param, "shared"]
                final_val = final_df.loc[param, "shared"]
                assert np.allclose(
                    initial_val,
                    final_val,
                    rtol=1e-6,
                    atol=1e-6,
                ), (
                    f"Shared parameter {param} changed from {initial_val} to {final_val} "
                    "with rw_sd=0"
                )

    if initial_unit_specific is not None and final_unit_specific is not None:
        for rep_idx in range(len(final_unit_specific)):
            initial_df = initial_unit_specific[rep_idx]
            final_df = final_unit_specific[rep_idx]

            for param in unit_names:
                for unit in final_df.columns:
                    initial_val = initial_df.loc[param, unit]
                    final_val = final_df.loc[param, unit]
                    assert np.allclose(
                        initial_val,
                        final_val,
                        rtol=1e-6,
                        atol=1e-6,
                    ), (
                        f"Unit parameter {param} for {unit} changed from {initial_val} "
                        f"to {final_val} with rw_sd=0"
                    )
