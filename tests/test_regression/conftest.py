"""Fixed, small models for algorithm regression tests.

Most algorithms are exercised on a 1-D linear-Gaussian model so that the locked
baselines stay small. ``sir_struct`` adds a non-Gaussian counterpart: the LG
model has no accumulator variables, no discrete measurement, and a trivial
parameter transform, so on its own it leaves those paths unlocked.

``lg_struct_multi`` carries two distinct parameter sets so that the
vmap-over-replicates axis is covered; a broadcasting bug that collapsed
replicates would pass against a single-replicate baseline.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import pypomp as pp


@pytest.fixture(scope="module")
def lg_struct():
    """(struct, theta0, key, J, n_reps, param_names) for a 1-D LG model."""
    model = pp.models.LG(A=np.array([[0.9]]), T=4, key=jax.random.key(0))
    struct = model.to_struct()
    param_names = model.canonical_param_names
    theta0 = model.theta.to_jax_array(param_names)
    key = jax.random.key(20260811)
    J = 12
    n_reps = 1
    return struct, theta0, key, J, n_reps, param_names


@pytest.fixture(scope="module")
def lg_struct_multi():
    """(struct, thetas, key, J, n_reps, param_names) with two distinct thetas."""
    model = pp.models.LG(A=np.array([[0.9]]), T=4, key=jax.random.key(0))
    struct = model.to_struct()
    param_names = model.canonical_param_names
    theta0 = model.theta.to_jax_array(param_names)
    # Second replicate is a deterministic perturbation of the first, so the two
    # rows stay distinguishable in the baseline.
    thetas = jnp.concatenate([theta0, theta0 * 1.05], axis=0)
    return struct, thetas, jax.random.key(20260812), 12, 2, param_names


@pytest.fixture(scope="module")
def sir_struct():
    """(struct, theta0, key, J, n_reps, param_names) for a small SIR model.

    Non-Gaussian and discrete-valued, with accumulator variables that reset at
    every observation time.
    """
    model = pp.models.sir(times=np.arange(1, 6) / 52.0, seed=11)
    struct = model.to_struct()
    param_names = model.canonical_param_names
    theta0 = model.theta.to_jax_array(param_names)
    return struct, theta0, jax.random.key(20260812), 8, 1, param_names


@pytest.fixture(scope="module")
def lg_panel_struct():
    """(struct, shared0, unit0, shared_names, unit_names, key, J, n_reps)."""
    lg1 = pp.models.LG(A=np.array([[0.9]]), T=4, key=jax.random.key(0))
    lg2 = pp.models.LG(A=np.array([[0.9]]), T=4, key=jax.random.key(1))
    lg1.par_trans = pp.ParTrans()
    lg2.par_trans = pp.ParTrans()

    shared_param_names = ["A11", "C11"]
    unit_param_names = ["Q11", "R11", "X0_1"]

    theta1 = lg1.theta.params(as_list=True)[0]
    theta2 = lg2.theta.params(as_list=True)[0]

    shared_params = pd.DataFrame(
        index=pd.Index(shared_param_names),
        data={"shared": [theta1[name] for name in shared_param_names]},
    )
    unit_specific_params = pd.DataFrame(
        index=pd.Index(unit_param_names),
        data={
            "unit1": [theta1[name] for name in unit_param_names],
            "unit2": [theta2[name] for name in unit_param_names],
        },
    )

    panel = pp.PanelPomp(
        Pomp_dict={"unit1": lg1, "unit2": lg2},
        theta=pp.PanelParameters(
            [{"shared": shared_params, "unit_specific": unit_specific_params}]
        ),
    )
    struct = panel.to_struct()

    shared0 = jnp.array([theta1[name] for name in shared_param_names])[None, :]
    unit0 = jnp.stack(
        [
            jnp.array([theta1[name] for name in unit_param_names]),
            jnp.array([theta2[name] for name in unit_param_names]),
        ]
    )[None, :, :]
    key = jax.random.key(20260811)
    J = 12
    n_reps = 1
    return (
        struct,
        shared0,
        unit0,
        shared_param_names,
        unit_param_names,
        key,
        J,
        n_reps,
    )


@pytest.fixture
def tol():
    """Tolerance for every regression comparison.

    Tight enough to catch a real algorithmic change, loose enough to absorb
    float32 non-associativity across platforms.
    """
    return dict(atol=1e-6, rtol=1e-4)
