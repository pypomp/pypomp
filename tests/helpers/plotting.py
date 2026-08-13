"""Trace plotting for the accuracy tests.

Plots are a debugging aid, not an assertion: the helper no-ops when plotnine is
absent or when running in CI.
"""

import pandas as pd


def save_traces_plotnine(
    model,
    filename="traces.png",
    true_values=None,
    expected_values=None,
    mle_values=None,
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
        mle_values (dict, optional): Dictionary of true MLE values.
            Can map parameter name strings (e.g. 'a') or (parameter, unit) tuples
            (e.g. ('sigma_x', 'unit1')) to floats.
    """
    import os

    if os.getenv("GITHUB_ACTIONS"):
        return

    try:
        import warnings

        from plotnine import (
            aes,
            facet_wrap,
            geom_hline,
            geom_line,
            ggplot,
            labs,
            theme_minimal,
        )
        from plotnine.exceptions import PlotnineWarning

        warnings.filterwarnings("ignore", category=PlotnineWarning)
        warnings.filterwarnings(
            "ignore",
            message=".*'generic' unit for NumPy timedelta is deprecated.*",
            category=DeprecationWarning,
        )

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

        # Extract MLE values for each facet if mle_values is provided
        df_mle = pd.DataFrame()
        if mle_values is not None:
            mle_rows = []
            for _, row in df_long.drop_duplicates(["facet_label"]).iterrows():
                param = row["parameter"]
                unit = row["unit"]
                mle_val = None
                if (param, unit) in mle_values:
                    mle_val = mle_values[(param, unit)]
                elif param in mle_values:
                    mle_val = mle_values[param]
                if mle_val is not None:
                    mle_rows.append(
                        {
                            "facet_label": row["facet_label"],
                            "mle_value": float(mle_val),
                        }
                    )
            if mle_rows:
                df_mle = pd.DataFrame(mle_rows)

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
        subtitle_parts = []
        if not df_true.empty:
            subtitle_parts.append("Red: True Value")
        if not df_mle.empty:
            subtitle_parts.append("Green: True MLE")
        if not df_expected.empty:
            subtitle_parts.append("Blue: Expected Biased Estimate (Hurwicz Bias)")
        subtitle_text = " | ".join(subtitle_parts)

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
                subtitle=subtitle_text,
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

        # Add horizontal line for MLE values if available
        if not df_mle.empty:
            p = p + geom_hline(
                aes(yintercept="mle_value"),
                data=df_mle,
                linetype="dashed",
                color="green",
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
