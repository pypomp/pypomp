import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pypomp as pp
import pytest
from typing import Any, cast

pytestmark = pytest.mark.heavy


def save_traces_plotnine(
    model, filename="traces.png", true_values=None, expected_values=None
):
    """
    Saves a plot of the parameter and log-likelihood traces using plotnine.
    Only runs locally (skipped when GITHUB_ACTIONS environment variable is set).

    Args:
        model: A Pomp or PanelPomp object with a non-empty results history.
        filename (str): Path where the PNG image will be saved.
        true_values (dict, optional): Dictionary of true parameter values.
            Can map parameter name strings (e.g. 'a') or (parameter, unit) tuples
            (e.g. ('sigma_x', 'unit1')) to floats.
        expected_values (dict, optional): Dictionary of expected/biased parameter values.
            Can map parameter name strings (e.g. 'a') or (parameter, unit) tuples
            (e.g. ('sigma_x', 'unit1')) to floats.
    """
    import os

    if os.getenv("GITHUB_ACTIONS"):
        return

    try:
        from plotnine import (
            ggplot,
            aes,
            geom_line,
            geom_hline,
            facet_wrap,
            theme_minimal,
            labs,
        )
        from plotnine.exceptions import PlotnineWarning
        import warnings

        warnings.filterwarnings("ignore", category=PlotnineWarning)

        # 1. Retrieve the tidy trace dataframe
        traces = model.traces()
        if traces.empty:
            return

        # 2. Identify shared and unit-specific parameters
        if hasattr(model, "canonical_shared_param_names"):
            shared_params = list(model.canonical_shared_param_names)
            unit_params = list(model.canonical_unit_param_names)
        else:
            # Pointwise Pomp: all parameters are considered shared
            shared_params = [
                c
                for c in traces.columns
                if c not in {"theta_idx", "iteration", "method", "unit", "logLik", "se"}
            ]
            unit_params = []

        df_list = []

        # 3. Process shared parameters + joint logLik (from unit == 'shared')
        shared_cols = shared_params + ["logLik"]
        if "unit" in traces.columns:
            shared_df = traces[traces["unit"] == "shared"]
        else:
            shared_df = traces.copy()
            shared_df["unit"] = "shared"

        if not shared_df.empty:
            df_s = pd.melt(
                shared_df,
                id_vars=["theta_idx", "iteration", "method", "unit"],
                value_vars=[c for c in shared_cols if c in shared_df.columns],
                var_name="parameter",
                value_name="value",
            ).dropna(subset=["value"])
            df_list.append(df_s)

        # 4. Process unit-specific parameters (from unit != 'shared')
        if unit_params and "unit" in traces.columns:
            unit_df = traces[traces["unit"] != "shared"]
            if not unit_df.empty:
                df_u = pd.melt(
                    unit_df,
                    id_vars=["theta_idx", "iteration", "method", "unit"],
                    value_vars=[c for c in unit_params if c in unit_df.columns],
                    var_name="parameter",
                    value_name="value",
                ).dropna(subset=["value"])
                df_list.append(df_u)

        if not df_list:
            return
        df_long = pd.concat(df_list, ignore_index=True)

        # 5. Create clean facet labels
        df_long["facet_label"] = df_long.apply(
            lambda r: (
                f"{r['parameter']} ({r['unit']})"
                if r["unit"] != "shared"
                else r["parameter"]
            ),
            axis=1,
        )

        # Convert theta_idx to a string/category for discrete color mapping
        df_long["theta_idx"] = df_long["theta_idx"].astype(str)

        # 6. Extract true values for each facet if true_values is provided
        df_true = pd.DataFrame()
        if true_values is not None:
            true_rows = []
            for _, row in df_long.drop_duplicates(["facet_label"]).iterrows():
                param = row["parameter"]
                unit = row["unit"]
                true_val = None
                if (param, unit) in true_values:
                    true_val = true_values[(param, unit)]
                elif param in true_values:
                    true_val = true_values[param]
                if true_val is not None:
                    true_rows.append(
                        {
                            "facet_label": row["facet_label"],
                            "true_value": float(true_val),
                        }
                    )
            if true_rows:
                df_true = pd.DataFrame(true_rows)

        # Extract expected values for each facet if expected_values is provided
        df_expected = pd.DataFrame()
        if expected_values is not None:
            expected_rows = []
            for _, row in df_long.drop_duplicates(["facet_label"]).iterrows():
                param = row["parameter"]
                unit = row["unit"]
                exp_val = None
                if (param, unit) in expected_values:
                    exp_val = expected_values[(param, unit)]
                elif param in expected_values:
                    exp_val = expected_values[param]
                if exp_val is not None:
                    expected_rows.append(
                        {
                            "facet_label": row["facet_label"],
                            "expected_value": float(exp_val),
                        }
                    )
            if expected_rows:
                df_expected = pd.DataFrame(expected_rows)

        # 7. Construct the plotnine ggplot object
        p = (
            ggplot(
                df_long,
                aes(x="iteration", y="value", color="theta_idx", group="theta_idx"),
            )
            + geom_line(alpha=0.8, size=1)
            + facet_wrap("~facet_label", scales="free_y", ncol=1)
            + theme_minimal()
            + labs(
                title="Parameter & Log-Likelihood Traces",
                subtitle="Red line: True Value | Blue line: Expected Biased Estimate (Hurwicz Bias)",
                x="Iteration",
                y="Value",
                color="Replicate",
            )
        )

        # Add horizontal line for true values if available
        if not df_true.empty:
            p = p + geom_hline(
                aes(yintercept="true_value"),
                data=df_true,
                linetype="dashed",
                color="red",
                size=0.8,
                alpha=0.8,
            )

        # Add horizontal line for expected values if available
        if not df_expected.empty:
            p = p + geom_hline(
                aes(yintercept="expected_value"),
                data=df_expected,
                linetype="dashed",
                color="blue",
                size=0.8,
                alpha=0.8,
            )

        # Save plot (height scales dynamically based on the number of facets)
        n_facets = len(df_long["facet_label"].unique())
        try:
            # Ensure target directory exists
            dirname = os.path.dirname(filename)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            p.save(filename, width=8, height=2.5 * n_facets, dpi=300)
        except Exception as e:
            # Handle sandbox permission errors or Matplotlib font manager issues gracefully
            print(f"Could not save trace plot to {filename}: {e}")
    except ImportError:
        pass


# =====================================================================
# 1. 1D Linear Gaussian Model (LGM) Definition & Exact Likelihood
# =====================================================================


def make_lgm_pomp(ys, a, sigma_x, sigma_y):
    """
    Constructs a 1D Linear Gaussian Model as a pp.Pomp object.

    Equations:
      X_0 = 0.0
      X_t = a * X_{t-1} + eta_t,  eta_t ~ N(0, sigma_x^2)
      Y_t = X_t + epsilon_t,      epsilon_t ~ N(0, sigma_y^2)
    """
    theta_dict = {"a": float(a), "sigma_x": float(sigma_x), "sigma_y": float(sigma_y)}

    def rinit(theta_, key, covars, t0):
        return {"X": 0.0}

    def rproc(X_, theta_, key, covars, t, dt):
        noise = jax.random.normal(key) * theta_["sigma_x"]
        return {"X": theta_["a"] * X_["X"] + noise}

    def dmeas(Y_, X_, theta_, covars, t):
        return jax.scipy.stats.norm.logpdf(
            Y_["Y"], loc=X_["X"], scale=theta_["sigma_y"]
        )

    def rmeas(X_, theta_, key, covars, t):
        val = jax.random.normal(key) * theta_["sigma_y"] + X_["X"]
        return jnp.array([val])

    def to_est(theta: dict[str, float | jax.Array]) -> dict[str, float | jax.Array]:
        # Constrain a in (0, 1) using logit, and sigmas > 0 using log
        a_clip = jnp.clip(theta["a"], 1e-6, 1.0 - 1e-6)
        res: dict[str, float | jax.Array] = {
            "a": jnp.log(a_clip / (1.0 - a_clip)),
            "sigma_x": jnp.log(jnp.maximum(theta["sigma_x"], 1e-6)),
            "sigma_y": jnp.log(jnp.maximum(theta["sigma_y"], 1e-6)),
        }
        return res

    def from_est(theta: dict[str, float | jax.Array]) -> dict[str, float | jax.Array]:
        res: dict[str, float | jax.Array] = {
            "a": 1.0 / (1.0 + jnp.exp(-theta["a"])),
            "sigma_x": jnp.exp(theta["sigma_x"]),
            "sigma_y": jnp.exp(theta["sigma_y"]),
        }
        return res

    return pp.Pomp(
        ys=ys,
        theta=pp.PompParameters(theta_dict),
        statenames=["X"],
        t0=0.0,
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        rmeas=rmeas,
        par_trans=pp.ParTrans(to_est=to_est, from_est=from_est),
        nstep=1,
    )


def lgm_exact_loglik(ys, a, sigma_x, sigma_y, x0=0.0):
    """
    Computes the exact log-likelihood of a 1D LGM via the Kalman Filter.
    """
    y_vals = np.array(ys["Y"]) if isinstance(ys, pd.DataFrame) else np.array(ys)
    x = x0
    p = 0.0
    loglik = 0.0
    for y in y_vals:
        # Predict
        x_pred = a * x
        p_pred = (a**2) * p + sigma_x**2
        # Update
        v = y - x_pred
        S = p_pred + sigma_y**2
        loglik += -0.5 * (np.log(2.0 * np.pi * S) + (v**2) / S)
        K = p_pred / S
        x = x_pred + K * v
        p = (1.0 - K) * p_pred
    return loglik


# =====================================================================
# 3. Panel POMP Construction Helpers
# =====================================================================


def make_lgm_panel_pomp(ys_dict, shared_params, unit_specific_params):
    """
    Constructs a PanelPomp for the LGM model.
    """
    pomp_dict = {}
    for name, spec in unit_specific_params.items():
        all_params = {**shared_params, **spec}
        pomp_dict[name] = make_lgm_pomp(ys_dict[name], **all_params)

    shared_names = list(shared_params.keys())
    unit_names = list(unit_specific_params.keys())
    spec_names = list(unit_specific_params[unit_names[0]].keys())

    shared_df = pd.DataFrame(
        {"shared": [shared_params[n] for n in shared_names]},
        index=pd.Index(shared_names),
    )
    unit_specific_df = pd.DataFrame(
        {
            unit: [unit_specific_params[unit][n] for n in spec_names]
            for unit in unit_names
        },
        index=pd.Index(spec_names),
    )

    theta = pp.PanelParameters(
        theta=[{"shared": shared_df, "unit_specific": unit_specific_df}]
    )
    return pp.PanelPomp(Pomp_dict=pomp_dict, theta=theta)


# =====================================================================
# 4. POMP Accuracy Tests
# =====================================================================


def test_pomp_pfilter_accuracy():
    """
    Verify that the estimated log-likelihood matches the exact log-likelihood
    for LGM (Kalman Filter) using vectorized replicates.
    """
    T = 100
    key = jax.random.key(1234)
    ys_dummy = pd.DataFrame(0.0, index=np.arange(1, T + 1, dtype=float), columns=["Y"])

    true_a, true_sx, true_sy = 0.8, 0.5, 0.3
    model_gen = make_lgm_pomp(ys_dummy, true_a, true_sx, true_sy)
    sim_model = model_gen.simulate(key=key, nsim=1, as_pomp=True)

    exact_ll = lgm_exact_loglik(sim_model.ys, true_a, true_sx, true_sy)

    sim_model.pfilter(J=5000, key=key, reps=30)
    est_ll = sim_model.theta.logLik.item()

    err = np.abs(est_ll - exact_ll)
    assert err < 0.225, f"LGM pfilter error: est={est_ll}, exact={exact_ll}"


def test_pomp_mif_accuracy():
    """
    Verify that MIF parameter estimation converges toward the true parameter values
    starting from perturbed parameters, using vectorized replicates sampled from a box.
    """
    T = 100
    key = jax.random.key(1234)
    ys_dummy = pd.DataFrame(0.0, index=np.arange(1, T + 1, dtype=float), columns=["Y"])

    true_a, true_sx, true_sy = 0.8, 0.5, 0.3
    sim_model = make_lgm_pomp(ys_dummy, true_a, true_sx, true_sy).simulate(
        key=key, nsim=1, as_pomp=True
    )

    fit_model = make_lgm_pomp(sim_model.ys, a=0.5, sigma_x=0.8, sigma_y=0.6)

    # Sample 5 sets of parameters
    param_bounds = {"a": (0.1, 1.0), "sigma_x": (0.1, 1.0), "sigma_y": (0.1, 1.0)}
    fit_model.theta = pp.Pomp.sample_params(param_bounds, n=5, key=key)

    rw_sd = pp.RWSigma(
        sigmas={"a": 0.02, "sigma_x": 0.02, "sigma_y": 0.02}, init_names=[]
    ).geometric_cooling(0.5)

    fit_model.mif(J=3000, M=100, rw_sd=rw_sd, key=key)

    final_theta = fit_model.theta.params(as_list=False)
    mean_theta = final_theta.sel(unit="shared").mean(dim="theta_idx")

    est_a = mean_theta.sel(parameter="a").item()
    est_sx = mean_theta.sel(parameter="sigma_x").item()
    est_sy = mean_theta.sel(parameter="sigma_y").item()

    err_a = np.abs(est_a - true_a)
    err_sx = np.abs(est_sx - true_sx)
    err_sy = np.abs(est_sy - true_sy)

    initial_err = np.linalg.norm([0.5 - true_a, 0.8 - true_sx, 0.6 - true_sy])
    final_err = np.linalg.norm([est_a - true_a, est_sx - true_sx, est_sy - true_sy])

    assert err_a < 0.12
    assert err_sx < 0.15
    assert err_sy < 0.225
    assert final_err < 0.60 * initial_err

    save_traces_plotnine(
        fit_model,
        "tests/plots/pomp_mif_traces.png",
        true_values={"a": 0.8, "sigma_x": 0.5, "sigma_y": 0.3},
        expected_values={"a": 0.8 - (1 + 3 * 0.8) / 100},  # Hurwicz bias for T=100
    )


def test_pomp_train_accuracy():
    """
    Verify that train (MOP optimization) converges toward the true parameter values
    starting from perturbed parameters, using vectorized replicates sampled from a box.
    """
    T = 100
    key = jax.random.key(1234)
    ys_dummy = pd.DataFrame(0.0, index=np.arange(1, T + 1, dtype=float), columns=["Y"])

    true_a, true_sx, true_sy = 0.8, 0.5, 0.3
    sim_model = make_lgm_pomp(ys_dummy, true_a, true_sx, true_sy).simulate(
        key=key, nsim=1, as_pomp=True
    )

    fit_model = make_lgm_pomp(sim_model.ys, a=0.5, sigma_x=0.8, sigma_y=0.6)

    # Sample 5 sets of parameters
    param_bounds = {"a": (0.1, 1.0), "sigma_x": (0.1, 1.0), "sigma_y": (0.1, 1.0)}
    fit_model.theta = pp.Pomp.sample_params(param_bounds, n=5, key=key)

    eta = pp.LearningRate({"a": 0.05, "sigma_x": 0.05, "sigma_y": 0.05}).cosine_decay(
        0.1, M=150
    )

    fit_model.train(
        J=1000,
        M=150,
        eta=eta,
        optimizer=pp.Adam(scale=True, beta1=0.8),
        alpha=1.0,
        key=key,
    )

    final_theta = fit_model.theta.params(as_list=False)
    mean_theta = final_theta.sel(unit="shared").mean(dim="theta_idx")

    est_a = mean_theta.sel(parameter="a").item()
    est_sx = mean_theta.sel(parameter="sigma_x").item()
    est_sy = mean_theta.sel(parameter="sigma_y").item()

    err_a = np.abs(est_a - true_a)
    err_sx = np.abs(est_sx - true_sx)
    err_sy = np.abs(est_sy - true_sy)

    initial_err = np.linalg.norm([0.5 - true_a, 0.8 - true_sx, 0.6 - true_sy])
    final_err = np.linalg.norm([est_a - true_a, est_sx - true_sx, est_sy - true_sy])

    assert err_a < 0.15
    assert err_sx < 0.225
    assert err_sy < 0.18
    assert final_err < 0.525 * initial_err

    save_traces_plotnine(
        fit_model,
        "tests/plots/pomp_train_traces.png",
        true_values={"a": 0.8, "sigma_x": 0.5, "sigma_y": 0.3},
        expected_values={"a": 0.8 - (1 + 3 * 0.8) / 100},  # Hurwicz bias for T=100
    )


# =====================================================================
# 5. Panel POMP Accuracy Tests
# =====================================================================


def test_panel_pfilter_accuracy():
    """
    Verify Panel POMP pfilter accuracy: unit-specific and joint likelihood estimates.
    """
    T = 100
    key = jax.random.key(1234)
    ys_dummy = pd.DataFrame(0.0, index=np.arange(1, T + 1, dtype=float), columns=["Y"])

    # 2 units: shared autoregressive parameter 'a'
    shared_params = {"a": 0.8}
    unit_spec_params = {
        "unit1": {"sigma_x": 0.5, "sigma_y": 0.3},
        "unit2": {"sigma_x": 0.4, "sigma_y": 0.2},
    }

    # Generate data
    ys_dict = {}
    for unit, spec in unit_spec_params.items():
        all_params = {**shared_params, **spec}
        p_sim = make_lgm_pomp(ys_dummy, **all_params).simulate(
            key=key, nsim=1, as_pomp=True
        )
        ys_dict[unit] = p_sim.ys

    panel = make_lgm_panel_pomp(ys_dict, shared_params, unit_spec_params)

    # Evaluate exact unit likelihoods
    exact_unit1_ll = lgm_exact_loglik(ys_dict["unit1"], 0.8, 0.5, 0.3)
    exact_unit2_ll = lgm_exact_loglik(ys_dict["unit2"], 0.8, 0.4, 0.2)

    # Run panel pfilter
    panel.pfilter(J=5000, reps=30, key=key)
    res = cast(Any, panel.results_history[-1])

    # Check unit-specific log-likelihood coordinates
    # For PanelPomp results, logLiks is a DataArray of shape (1, U, reps)
    est_unit1_ll = res.logLiks.sel(unit="unit1").mean().item()
    est_unit2_ll = res.logLiks.sel(unit="unit2").mean().item()

    assert np.abs(est_unit1_ll - exact_unit1_ll) < 0.225
    assert np.abs(est_unit2_ll - exact_unit2_ll) < 0.225


@pytest.mark.parametrize(
    "shared_params, unit_spec_params, pert_shared, pert_spec, shared_names, expected_targets, plot_filename, plot_true_values",
    [
        # Case 1: Mixed
        (
            {"a": 0.8},
            {
                "unit1": {"sigma_x": 0.5, "sigma_y": 0.3},
                "unit2": {"sigma_x": 0.4, "sigma_y": 0.2},
            },
            {"a": 0.5},
            {
                "unit1": {"sigma_x": 0.8, "sigma_y": 0.6},
                "unit2": {"sigma_x": 0.7, "sigma_y": 0.5},
            },
            ["a"],
            {
                ("a", None): (0.8, 0.18),
                ("sigma_x", "unit1"): (0.5, 0.12),
                ("sigma_y", "unit1"): (0.3, 0.09),
                ("sigma_x", "unit2"): (0.4, 0.09),
                ("sigma_y", "unit2"): (0.2, 0.105),
            },
            "tests/plots/panel_mif_traces.png",
            {
                "a": 0.8,
                ("sigma_x", "unit1"): 0.5,
                ("sigma_y", "unit1"): 0.3,
                ("sigma_x", "unit2"): 0.4,
                ("sigma_y", "unit2"): 0.2,
            },
        ),
        # Case 2: Shared-only
        (
            {"a": 0.8, "sigma_x": 0.5, "sigma_y": 0.3},
            {
                "unit1": {},
                "unit2": {},
            },
            {"a": 0.5, "sigma_x": 0.8, "sigma_y": 0.6},
            {
                "unit1": {},
                "unit2": {},
            },
            ["a", "sigma_x", "sigma_y"],
            {
                ("a", None): (0.8, 0.18),
                ("sigma_x", None): (0.5, 0.15),
                ("sigma_y", None): (0.3, 0.225),
            },
            "tests/plots/panel_mif_shared_only_traces.png",
            {
                "a": 0.8,
                "sigma_x": 0.5,
                "sigma_y": 0.3,
            },
        ),
        # Case 3: Unit-specific-only
        (
            {},
            {
                "unit1": {"a": 0.8, "sigma_x": 0.5, "sigma_y": 0.3},
                "unit2": {"a": 0.8, "sigma_x": 0.4, "sigma_y": 0.2},
            },
            {},
            {
                "unit1": {"a": 0.5, "sigma_x": 0.8, "sigma_y": 0.6},
                "unit2": {"a": 0.5, "sigma_x": 0.7, "sigma_y": 0.5},
            },
            [],
            {
                ("a", "unit1"): (0.8, 0.18),
                ("sigma_x", "unit1"): (0.5, 0.12),
                ("sigma_y", "unit1"): (0.3, 0.09),
                ("a", "unit2"): (0.8, 0.18),
                ("sigma_x", "unit2"): (0.4, 0.09),
                ("sigma_y", "unit2"): (0.2, 0.105),
            },
            "tests/plots/panel_mif_unit_specific_only_traces.png",
            {
                ("a", "unit1"): 0.8,
                ("sigma_x", "unit1"): 0.5,
                ("sigma_y", "unit1"): 0.3,
                ("a", "unit2"): 0.8,
                ("sigma_x", "unit2"): 0.4,
                ("sigma_y", "unit2"): 0.2,
            },
        ),
    ],
    ids=["mixed", "shared_only", "unit_specific_only"],
)
def test_panel_mif_accuracy(
    shared_params,
    unit_spec_params,
    pert_shared,
    pert_spec,
    shared_names,
    expected_targets,
    plot_filename,
    plot_true_values,
):
    """
    Verify Panel POMP MIF convergence for different parameter configurations.
    """
    T = 100
    key = jax.random.key(1234)
    ys_dummy = pd.DataFrame(0.0, index=np.arange(1, T + 1, dtype=float), columns=["Y"])

    ys_dict = {}
    for unit, spec in unit_spec_params.items():
        all_params = {**shared_params, **spec}
        p_sim = make_lgm_pomp(ys_dummy, **all_params).simulate(
            key=key, nsim=1, as_pomp=True
        )
        ys_dict[unit] = p_sim.ys

    panel = make_lgm_panel_pomp(ys_dict, pert_shared, pert_spec)

    # Sample 5 sets of parameters
    param_bounds = {"a": (0.1, 1.0), "sigma_x": (0.1, 1.0), "sigma_y": (0.1, 1.0)}
    unit_names = list(panel.unit_objects.keys())
    panel.theta = pp.PanelPomp.sample_params(
        param_bounds=param_bounds,
        units=unit_names,
        n=5,
        key=key,
        shared_names=shared_names,
    )

    rw_sd = pp.RWSigma(
        sigmas={"a": 0.02, "sigma_x": 0.02, "sigma_y": 0.02}, init_names=[]
    ).geometric_cooling(0.5)

    panel.mif(J=3000, M=100, rw_sd=rw_sd, key=key)

    final_theta = panel.theta.params(as_list=False)
    mean_shared = final_theta["shared"].mean(dim="theta_idx")
    mean_spec = final_theta["unit_specific"].mean(dim="theta_idx")

    for (param, unit), (true_val, tolerance) in expected_targets.items():
        if unit is None:
            est_val = mean_shared.sel(parameter=param).item()
        else:
            est_val = mean_spec.sel(unit=unit, parameter=param).item()
        err = np.abs(est_val - true_val)
        assert err < tolerance, (
            f"MIF error for parameter={param}, unit={unit}: est={est_val}, true={true_val}"
        )

    save_traces_plotnine(
        panel,
        plot_filename,
        true_values=plot_true_values,
        expected_values={"a": 0.8 - (1 + 3 * 0.8) / 100}
        if "a" in panel.canonical_param_names
        else None,
    )


@pytest.mark.parametrize(
    "shared_params, unit_spec_params, pert_shared, pert_spec, shared_names, expected_targets, plot_filename, plot_true_values",
    [
        # Case 1: Mixed
        (
            {"a": 0.8},
            {
                "unit1": {"sigma_x": 0.5, "sigma_y": 0.3},
                "unit2": {"sigma_x": 0.4, "sigma_y": 0.2},
            },
            {"a": 0.5},
            {
                "unit1": {"sigma_x": 0.8, "sigma_y": 0.6},
                "unit2": {"sigma_x": 0.7, "sigma_y": 0.5},
            },
            ["a"],
            {
                ("a", None): (0.8, 0.18),
                ("sigma_x", "unit1"): (0.5, 0.105),
                ("sigma_y", "unit1"): (0.3, 0.18),
                ("sigma_x", "unit2"): (0.4, 0.075),
                ("sigma_y", "unit2"): (0.2, 0.09),
            },
            "tests/plots/panel_train_traces.png",
            {
                "a": 0.8,
                ("sigma_x", "unit1"): 0.5,
                ("sigma_y", "unit1"): 0.3,
                ("sigma_x", "unit2"): 0.4,
                ("sigma_y", "unit2"): 0.2,
            },
        ),
        # Case 2: Shared-only
        (
            {"a": 0.8, "sigma_x": 0.5, "sigma_y": 0.3},
            {
                "unit1": {},
                "unit2": {},
            },
            {"a": 0.5, "sigma_x": 0.8, "sigma_y": 0.6},
            {
                "unit1": {},
                "unit2": {},
            },
            ["a", "sigma_x", "sigma_y"],
            {
                ("a", None): (0.8, 0.18),
                ("sigma_x", None): (0.5, 0.105),
                ("sigma_y", None): (0.3, 0.18),
            },
            "tests/plots/panel_train_shared_only_traces.png",
            {
                "a": 0.8,
                "sigma_x": 0.5,
                "sigma_y": 0.3,
            },
        ),
        # Case 3: Unit-specific-only
        (
            {},
            {
                "unit1": {"a": 0.8, "sigma_x": 0.5, "sigma_y": 0.3},
                "unit2": {"a": 0.8, "sigma_x": 0.4, "sigma_y": 0.2},
            },
            {},
            {
                "unit1": {"a": 0.5, "sigma_x": 0.8, "sigma_y": 0.6},
                "unit2": {"a": 0.5, "sigma_x": 0.7, "sigma_y": 0.5},
            },
            [],
            {
                ("a", "unit1"): (0.8, 0.18),
                ("sigma_x", "unit1"): (0.5, 0.105),
                ("sigma_y", "unit1"): (0.3, 0.18),
                ("a", "unit2"): (0.8, 0.18),
                ("sigma_x", "unit2"): (0.4, 0.075),
                ("sigma_y", "unit2"): (0.2, 0.09),
            },
            "tests/plots/panel_train_unit_specific_only_traces.png",
            {
                ("a", "unit1"): 0.8,
                ("sigma_x", "unit1"): 0.5,
                ("sigma_y", "unit1"): 0.3,
                ("a", "unit2"): 0.8,
                ("sigma_x", "unit2"): 0.4,
                ("sigma_y", "unit2"): 0.2,
            },
        ),
    ],
    ids=["mixed", "shared_only", "unit_specific_only"],
)
def test_panel_train_accuracy(
    shared_params,
    unit_spec_params,
    pert_shared,
    pert_spec,
    shared_names,
    expected_targets,
    plot_filename,
    plot_true_values,
):
    """
    Verify Panel POMP train optimization convergence for different parameter configurations.
    """
    T = 100
    key = jax.random.key(1234)
    ys_dummy = pd.DataFrame(0.0, index=np.arange(1, T + 1, dtype=float), columns=["Y"])

    ys_dict = {}
    for unit, spec in unit_spec_params.items():
        all_params = {**shared_params, **spec}
        p_sim = make_lgm_pomp(ys_dummy, **all_params).simulate(
            key=key, nsim=1, as_pomp=True
        )
        ys_dict[unit] = p_sim.ys

    panel = make_lgm_panel_pomp(ys_dict, pert_shared, pert_spec)

    # Sample 5 sets of parameters
    param_bounds = {"a": (0.1, 1.0), "sigma_x": (0.1, 1.0), "sigma_y": (0.1, 1.0)}
    unit_names = list(panel.unit_objects.keys())
    panel.theta = pp.PanelPomp.sample_params(
        param_bounds=param_bounds,
        units=unit_names,
        n=5,
        key=key,
        shared_names=shared_names,
    )

    eta = pp.LearningRate({"a": 0.05, "sigma_x": 0.05, "sigma_y": 0.05}).cosine_decay(
        0.05, M=150
    )

    panel.train(
        J=1000,
        M=150,
        eta=eta,
        optimizer=pp.Adam(scale=True, beta1=0.8),
        alpha=1.0,
        key=key,
    )

    final_theta = panel.theta.params(as_list=False)
    mean_shared = final_theta["shared"].mean(dim="theta_idx")
    mean_spec = final_theta["unit_specific"].mean(dim="theta_idx")

    for (param, unit), (true_val, tolerance) in expected_targets.items():
        if unit is None:
            est_val = mean_shared.sel(parameter=param).item()
        else:
            est_val = mean_spec.sel(unit=unit, parameter=param).item()
        err = np.abs(est_val - true_val)
        assert err < tolerance, (
            f"Train error for parameter={param}, unit={unit}: est={est_val}, true={true_val}"
        )

    save_traces_plotnine(
        panel,
        plot_filename,
        true_values=plot_true_values,
        expected_values={"a": 0.8 - (1 + 3 * 0.8) / 100}
        if "a" in panel.canonical_param_names
        else None,
    )
