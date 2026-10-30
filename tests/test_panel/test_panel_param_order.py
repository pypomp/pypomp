"""Panels whose units list their parameters in different orders."""

from copy import deepcopy

import jax
import numpy as np
import pytest

import pypomp as pp
import pypomp.functional as F
from tests.helpers.models import lg_panel
from tests.helpers.params import uniform_rw_sd

J = 5


def _with_unit_reversed(panel: pp.PanelPomp, unit: str) -> pp.PanelPomp:
    """Rebuild ``panel`` with ``unit``'s parameters in reverse order."""
    old = panel.unit_objects[unit]
    assert old.dmeas is not None and old.rmeas is not None
    theta = panel.get_unit_parameters(unit)[0]
    pomps = dict(panel.unit_objects)
    pomps[unit] = pp.Pomp(
        ys=old.ys,
        theta=pp.PompParameters(dict(reversed(theta.items()))),
        statenames=old.statenames,
        t0=old.t0,
        nstep=1,
        rinit=old.rinit.original_func,
        rproc=old.rproc.original_func,
        dmeas=old.dmeas.original_func,
        rmeas=old.rmeas.original_func,
        par_trans=old.par_trans,
    )
    return pp.PanelPomp(pomp_dict=pomps, theta=deepcopy(panel.theta))


@pytest.fixture
def panel():
    return lg_panel(sharing="some", unit_scales=(1.0, 0.9))


@pytest.mark.parametrize("unit", ["unit1", "unit2"])
def test_pfilter_ignores_unit_param_order(panel, unit):
    reordered = _with_unit_reversed(panel, unit)
    assert (
        panel.unit_objects[unit].canonical_param_names
        != reordered.unit_objects[unit].canonical_param_names
    )
    key = jax.random.key(0)

    panel.pfilter(J=J, key=key)
    reordered.pfilter(J=J, key=key)

    np.testing.assert_allclose(reordered.theta.logLik_unit, panel.theta.logLik_unit)


def test_mif_ignores_later_unit_param_order(panel):
    reordered = _with_unit_reversed(panel, "unit2")
    rw_sd = uniform_rw_sd(panel, cooling=0.5)
    key = jax.random.key(1)

    panel.mif(J=J, M=2, rw_sd=rw_sd, key=key)
    reordered.mif(J=J, M=2, rw_sd=rw_sd, key=key)

    np.testing.assert_allclose(
        reordered.theta.to_jax_array(), panel.theta.to_jax_array()
    )


def test_train_ignores_later_unit_param_order(panel):
    reordered = _with_unit_reversed(panel, "unit2")
    eta = pp.LearningRate({n: 0.01 for n in panel.canonical_param_names})
    key = jax.random.key(2)

    panel.train(J=J, M=2, eta=eta, key=key)
    reordered.train(J=J, M=2, eta=eta, key=key)

    np.testing.assert_allclose(
        reordered.theta.to_jax_array(), panel.theta.to_jax_array()
    )


def test_panel_pfilter_reads_struct_param_names_order(panel):
    """The functional filter takes ``struct.param_names`` order, as the method does."""
    key = jax.random.key(3)
    panel.pfilter(J=J, key=key)

    struct = panel.to_struct()
    thetas = panel.theta.to_jax_array(struct.param_names)
    _, sub_key = jax.random.split(key)
    keys = jax.random.split(sub_key, thetas.shape[0] * thetas.shape[1])
    keys = keys.reshape(thetas.shape[:2] + keys.shape[1:])

    results = F.panel_pfilter(struct, thetas, J=J, keys=keys)

    np.testing.assert_allclose(results["logLik"], panel.theta.logLik_unit)
