"""
Integration tests for parameter transformations in PanelPomp.mif method.
"""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pypomp as pp
from pypomp.types import ParamDict
from tests.helpers.models import lg_panel


@pytest.fixture
def panel_pomp_with_transform():
    """Create a simple PanelPomp model with custom ParTrans."""

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

    shared_param_names = ["A11", "A12", "A21", "A22", "C11", "C12", "C21", "C22"]

    panel = lg_panel(
        sharing="some",
        shared_names=shared_param_names,
        unit_scales=[0.8, 1.2],
        par_trans=pp.ParTrans(to_est, from_est),
    )

    return panel


def test_panel_mif_traces_transformed(panel_pomp_with_transform):
    """
    Test that with rw_sd=0, parameters remain unchanged after transformation cycle.
    """
    panel = panel_pomp_with_transform

    panel_theta = panel.theta.params(as_list=True)
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

    final_panel_theta = panel.theta.params(as_list=True)
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


def test_panel_mif_transform_before_tiling(panel_pomp_with_transform):
    """Test that PanelPomp.mif transforms parameters before tiling across J particles.

    Verifies that _transform_panel_array with direction="to_est" receives untiled
    arrays (ndim == 2 for shared, ndim == 3 for unit-specific), rather than J-tiled
    swarms (ndim == 3 and ndim == 4 with particle dimension J).
    """
    panel = panel_pomp_with_transform
    shared_names = panel.canonical_shared_param_names
    unit_names = panel.canonical_unit_param_names
    all_param_names = list(shared_names) + list(unit_names)
    rw_sd = pp.RWSigma(
        sigmas={k: 0.0 for k in all_param_names},
        init_names=[],
    ).geometric_cooling(0.5)

    J = 5
    M = 1
    rep_unit = panel.unit_objects[panel.get_unit_names()[0]]

    with patch.object(
        rep_unit.par_trans,
        "_transform_panel_array",
        wraps=rep_unit.par_trans._transform_panel_array,
    ) as spy_transform:
        panel.mif(J=J, M=M, rw_sd=rw_sd, key=jax.random.key(123))

        to_est_calls = [
            call
            for call in spy_transform.call_args_list
            if call.kwargs.get("direction") == "to_est"
        ]
        assert len(to_est_calls) >= 1
        first_call = to_est_calls[0]
        shared_arg = first_call.kwargs.get("shared_array")
        if shared_arg is None and len(first_call.args) > 0:
            shared_arg = first_call.args[0]
        unit_arg = first_call.kwargs.get("unit_array")
        if unit_arg is None and len(first_call.args) > 1:
            unit_arg = first_call.args[1]

        assert shared_arg is not None
        assert unit_arg is not None
        # Untiled parameters have shape (n_reps, n_shared) and (n_reps, U, n_spec)
        assert shared_arg.ndim == 2, (
            f"shared_arg has shape {shared_arg.shape}, expected 2D without J={J}"
        )
        assert unit_arg.ndim == 3, (
            f"unit_arg has shape {unit_arg.shape}, expected 3D without J={J}"
        )
