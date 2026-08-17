"""Constructor and method argument validation for Pomp."""

import jax
import jax.numpy as jnp
import pandas as pd
import pytest

import pypomp as pp
from tests.helpers.assertions import pickle_roundtrip
from tests.helpers.dummy import (
    dummy_dmeas,
    dummy_rinit,
    dummy_rmeas,
    dummy_rproc,
)


def test_init_validation():
    """Test invalid arguments during Pomp initialization."""
    ys = pd.DataFrame({"y": [1.0, 2.0]}, index=[1.0, 2.0])
    theta = pp.PompParameters({"X0": 0.0, "sigma": 0.1})

    # ys must be a DataFrame
    with pytest.raises(TypeError, match="ys must be a pandas DataFrame"):
        pp.Pomp(
            ys="not_df",  # type: ignore
            theta=theta,
            statenames=["X"],
            t0=0.0,
            rinit=dummy_rinit,
            rproc=dummy_rproc,
        )

    # covars must be a DataFrame
    with pytest.raises(TypeError, match="covars must be a pandas DataFrame"):
        pp.Pomp(
            ys=ys,
            theta=theta,
            statenames=["X"],
            t0=0.0,
            rinit=dummy_rinit,
            rproc=dummy_rproc,
            covars="not_df",  # type: ignore
        )

    # theta must be a PompParameters instance
    with pytest.raises(TypeError, match="theta must be a PompParameters instance"):
        pp.Pomp(
            ys=ys,
            theta={"X0": 0.0},  # type: ignore
            statenames=["X"],
            t0=0.0,
            rinit=dummy_rinit,
            rproc=dummy_rproc,
        )

    # statenames must be provided
    with pytest.raises(ValueError, match="statenames must be provided"):
        pp.Pomp(
            ys=ys,
            theta=theta,
            statenames=None,  # type: ignore
            t0=0.0,
            rinit=dummy_rinit,
            rproc=dummy_rproc,
        )

    # statenames must be list of strings
    with pytest.raises(
        ValueError, match="statenames must be a tuple or list of strings"
    ):
        pp.Pomp(
            ys=ys,
            theta=theta,
            statenames=[1, 2],  # type: ignore
            t0=0.0,
            rinit=dummy_rinit,
            rproc=dummy_rproc,
        )

    # accumvars validation
    with pytest.raises(
        ValueError, match="accumvars must be a tuple or list of strings"
    ):
        pp.Pomp(
            ys=ys,
            theta=theta,
            statenames=["X"],
            t0=0.0,
            rinit=dummy_rinit,
            rproc=dummy_rproc,
            accumvars=[1],  # type: ignore
        )

    with pytest.raises(ValueError, match="all accumvars must be in statenames"):
        pp.Pomp(
            ys=ys,
            theta=theta,
            statenames=["X"],
            t0=0.0,
            rinit=dummy_rinit,
            rproc=dummy_rproc,
            accumvars=["Y"],
        )

    # both dmeas and rmeas cannot be None
    with pytest.raises(
        ValueError, match="You must supply at least one of dmeas or rmeas"
    ):
        pp.Pomp(
            ys=ys,
            theta=theta,
            statenames=["X"],
            t0=0.0,
            rinit=dummy_rinit,
            rproc=dummy_rproc,
        )


def test_theta_getter_setter(base_pomp):
    """Test getter/setter validation for theta property."""
    # Getter when _theta is None
    base_pomp._theta = None
    with pytest.raises(ValueError, match="Model parameters have not been set"):
        _ = base_pomp.theta

    # Setter type check
    with pytest.raises(TypeError, match="theta must be a PompParameters instance"):
        base_pomp.theta = {"X0": 0.0}


def test_prepare_theta_input(base_pomp):
    """Test _prepare_theta_input validation."""
    # TypeError for wrong class
    with pytest.raises(
        TypeError, match="theta must be a PompParameters object or None"
    ):
        base_pomp._prepare_theta_input("invalid")

    # ValueError for mismatched param names
    bad_theta = pp.PompParameters({"unknown": 1.0})
    with pytest.raises(
        ValueError, match="theta parameter names must match canonical_param_names"
    ):
        base_pomp._prepare_theta_input(bad_theta)


def test_update_fresh_key(base_pomp):
    """Test _update_fresh_key validation."""
    # ValueError when both keys are None
    base_pomp.fresh_key = None
    with pytest.raises(
        ValueError,
        match="Both the key argument and the fresh_key attribute are None",
    ):
        base_pomp._update_fresh_key(None)


def test_simulate_edge_cases(base_pomp):
    """Test simulate with custom times, warnings, and as_pomp=True."""
    # 1. rmeas is None raises ValueError
    pomp_no_rmeas = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        dmeas=dummy_dmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    with pytest.raises(ValueError, match="self.rmeas cannot be None"):
        pomp_no_rmeas.simulate(key=jax.random.key(1))

    # 2. nsim > 1 when as_pomp is True triggers UserWarning
    with pytest.warns(UserWarning, match="as_pomp is True, but nsim > 1"):
        pomp_copy = base_pomp.simulate(key=jax.random.key(1), nsim=5, as_pomp=True)
    assert isinstance(pomp_copy, pp.Pomp)

    # 3. simulate with custom times
    times = jnp.array([1.5, 2.5])
    states, obs = base_pomp.simulate(key=jax.random.key(1), times=times)
    assert set(states["time"].unique()) == {0.0, 1.5, 2.5}
    assert set(obs["time"].unique()) == {1.5, 2.5}


def test_filtering_methods_validation(base_pomp):
    # Test validations for pfilter, mif, and train when components are missing.
    pomp_no_dmeas = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        rmeas=dummy_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    pomp_no_dmeas.fresh_key = jax.random.key(1)

    with pytest.raises(ValueError, match="self.dmeas cannot be None"):
        pomp_no_dmeas.pfilter(J=10)

    with pytest.raises(ValueError, match="self.dmeas cannot be None"):
        pomp_no_dmeas.mif(J=10, M=1, rw_sd=pp.RWSigma({"X0": 0.01, "sigma": 0.01}))

    with pytest.raises(ValueError, match="self.dmeas cannot be None"):
        pomp_no_dmeas.train(J=10, M=1, eta=pp.LearningRate({"X0": 0.1, "sigma": 0.1}))


def test_dpop_train_validation(base_pomp):
    """Test input validations for dpop_train."""
    eta = pp.LearningRate({"X0": 0.1, "sigma": 0.1})

    # 1. missing dmeas
    pomp_no_dmeas = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        rmeas=dummy_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    pomp_no_dmeas.fresh_key = jax.random.key(1)
    with pytest.raises(
        ValueError, match="dpop_train requires self.dmeas to be not None"
    ):
        pomp_no_dmeas._dpop_train(J=5, M=1, eta=eta, process_weight_state="logw")

    # 2. invalid eta
    with pytest.raises(TypeError, match="eta must be a LearningRate object"):
        base_pomp._dpop_train(J=5, M=1, eta="not_lr", process_weight_state="logw")

    # 3. missing process_weight_state
    with pytest.raises(ValueError, match="dpop_train requires a process-weight state"):
        base_pomp._dpop_train(J=5, M=1, eta=eta, process_weight_state=None)

    # 4. process_weight_state not in statenames
    with pytest.raises(ValueError, match="not found in statenames"):
        base_pomp._dpop_train(J=5, M=1, eta=eta, process_weight_state="non_existent")

    # 5. Test valid call with optimizer='SGD'
    # Add logw to states
    pomp_dpop = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=lambda theta_, key, covars, t0: {"X": theta_["X0"], "logw": 0.0},
        rproc=lambda X_, theta_, key, covars, t, dt: {
            "X": X_["X"],
            "logw": X_["logw"] + 0.1,
        },
        dmeas=dummy_dmeas,
        statenames=["X", "logw"],
        t0=0.0,
        nstep=1,
    )
    pomp_dpop.fresh_key = jax.random.key(1)
    pomp_dpop.results_history.clear()
    ret = pomp_dpop._dpop_train(
        J=2, M=2, eta=eta, process_weight_state="logw", optimizer=pp.SGD()
    )
    assert ret is None
    res = pomp_dpop.results_history[-1]
    assert res.method == "dpop_train"
    assert res.kind == "trace"
    traces = res.traces()
    assert len(traces) > 0


def test_merge_validation(base_pomp):
    """Test merge validation exceptions."""
    # 1. No arguments
    with pytest.raises(ValueError, match="At least one Pomp object must be provided"):
        pp.Pomp.merge()

    # 2. Different type
    with pytest.raises(TypeError, match="All merged objects must be of type Pomp"):
        pp.Pomp.merge(base_pomp, "not_pomp")  # type: ignore

    # 3. Mismatched parameter names
    p2_diff_params = pp.Pomp(
        ys=base_pomp.ys,
        theta=pp.PompParameters({"diff_name": 0.0, "sigma": 0.1}),
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        dmeas=dummy_dmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    with pytest.raises(ValueError, match="same canonical_param_names"):
        pp.Pomp.merge(base_pomp, p2_diff_params)

    # 4. Mismatched statenames
    p2_diff_states = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=lambda theta_, key, covars, t0: {"Y": theta_["X0"]},
        rproc=lambda X_, theta_, key, covars, t, dt: {"Y": X_["Y"]},
        dmeas=lambda Y_, X_, theta_, covars, t: 0.0,
        statenames=["Y"],
        t0=0.0,
        nstep=1,
    )
    with pytest.raises(ValueError, match="same statenames"):
        pp.Pomp.merge(base_pomp, p2_diff_states)

    # 5. Mismatched ys
    p2_diff_ys = pickle_roundtrip(base_pomp)
    p2_diff_ys.ys = pd.DataFrame({"y": [3.0, 4.0]}, index=[1.0, 2.0])
    with pytest.raises(ValueError, match="same ys data"):
        pp.Pomp.merge(base_pomp, p2_diff_ys)

    # 6. Mismatched t0
    p2_diff_t0 = pickle_roundtrip(base_pomp)
    p2_diff_t0.t0 = 5.0
    with pytest.raises(ValueError, match="same t0"):
        pp.Pomp.merge(base_pomp, p2_diff_t0)

    # 7. Mismatched dmeas None vs not None
    pomp_no_dmeas = pp.Pomp(
        ys=base_pomp.ys,
        theta=base_pomp.theta,
        rinit=dummy_rinit,
        rproc=dummy_rproc,
        rmeas=dummy_rmeas,
        statenames=["X"],
        t0=0.0,
        nstep=1,
    )
    with pytest.raises(ValueError, match="same dmeas"):
        pp.Pomp.merge(base_pomp, pomp_no_dmeas)

    # 8. Mismatched theta is None
    p2_no_theta = pickle_roundtrip(base_pomp)
    p2_no_theta._theta = None
    with pytest.raises(
        ValueError, match="Cannot merge Pomp objects with no parameters"
    ):
        pp.Pomp.merge(base_pomp, p2_no_theta)
