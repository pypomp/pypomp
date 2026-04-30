from typing import Any
import pandas as pd
import numpy as np


def _check_plotly():
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots

        return go, px, make_subplots
    except ImportError:
        raise ImportError(
            "Plotly is required for plotting. "
            "Please install it with `pip install plotly` or `pip install 'pypomp[viz]'`."
        )


def plot_traces_internal(traces: pd.DataFrame, title: str = "Traces") -> Any:
    """
    Internal function to plot traces using Plotly.
    """
    if traces.empty:
        print("No trace data to plot.")
        return None

    go, px, make_subplots = _check_plotly()

    id_vars = ["theta_idx", "iteration", "method", "unit"]
    present_id_vars = [col for col in id_vars if col in traces.columns]
    value_vars = [col for col in traces.columns if col not in id_vars]
    df_long = traces.melt(
        id_vars=present_id_vars,
        value_vars=value_vars,
        var_name="variable",
        value_name="value",
    )
    df_long = df_long.dropna(subset=["value"])

    if df_long.empty:
        print("No valid trace data to plot.")
        return None

    variables = df_long["variable"].unique()
    units = df_long["unit"].unique() if "unit" in df_long.columns else [None]

    if len(variables) == 1 and len(units) > 1:
        facet_col = "unit"
        facet_values = units
        subplot_titles = [str(u) for u in units]
        plot_df = df_long
    else:
        facet_col = "variable"
        facet_values = variables
        subplot_titles = [str(v) for v in variables]
        plot_df = df_long

    n_facets = len(facet_values)
    cols = 3 if n_facets > 3 else n_facets
    rows = (n_facets + cols - 1) // cols if n_facets > 0 else 1

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        vertical_spacing=0.1 if rows > 1 else 0.2,
    )

    # Use Scattergl if many points are present
    total_points = len(df_long)
    Scatter = go.Scattergl if total_points > 50000 else go.Scatter

    colors = px.colors.qualitative.Plotly
    theta_indices = sorted(df_long["theta_idx"].unique())
    color_map = {idx: colors[i % len(colors)] for i, idx in enumerate(theta_indices)}

    for i, facet_val in enumerate(facet_values):
        row = (i // cols) + 1
        col = (i % cols) + 1

        facet_data = plot_df[plot_df[facet_col] == facet_val]

        for theta_idx in theta_indices:
            theta_data = facet_data[facet_data["theta_idx"] == theta_idx]
            if theta_data.empty:
                continue

            color = color_map[theta_idx]

            for method in ["mif", "train"]:
                sub = theta_data[theta_data["method"] == method]
                if len(sub) > 1:
                    fig.add_trace(
                        Scatter(
                            x=sub["iteration"],
                            y=sub["value"],
                            mode="lines",
                            line=dict(color=color, width=2),
                            legendgroup=f"rep_{theta_idx}",
                            showlegend=False,
                        ),
                        row=row,
                        col=col,
                    )

            sub = theta_data[theta_data["method"] == "pfilter"]
            if not sub.empty:
                fig.add_trace(
                    Scatter(
                        x=sub["iteration"],
                        y=sub["value"],
                        mode="markers",
                        marker=dict(
                            color=color, size=8, line=dict(color="black", width=1)
                        ),
                        legendgroup=f"rep_{theta_idx}",
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

    fig.update_layout(
        title=title,
        height=min(1200, 300 * rows + 100),
        width=1000,
        showlegend=False,
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Iteration")
    fig.update_yaxes(title_text="Value")

    return fig


def plot_simulations_internal(
    sims: pd.DataFrame,
    obs: pd.DataFrame,
    mode: str = "lines",
    title: str = "Simulations vs Data",
) -> Any:
    """
    Internal function to plot simulations vs data using Plotly.
    """
    go, px, make_subplots = _check_plotly()

    obs_cols = [c for c in obs.columns if c not in ["time", "theta_idx", "sim"]]
    n_vars = len(obs_cols)

    cols = 2 if n_vars > 2 else n_vars
    rows = (n_vars + cols - 1) // cols if n_vars > 0 else 1

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=obs_cols,
        shared_xaxes=True,
        vertical_spacing=0.1,
    )

    total_points = len(sims)
    Scatter = go.Scattergl if total_points > 100000 else go.Scatter

    for i, var in enumerate(obs_cols):
        row = (i // cols) + 1
        col = (i % cols) + 1

        val_col = var if var in sims.columns else f"obs_{i}"

        if mode == "lines":
            for sim_idx in sims["sim"].unique():
                sub = sims[sims["sim"] == sim_idx]
                fig.add_trace(
                    Scatter(
                        x=sub["time"],
                        y=sub[val_col],
                        mode="lines",
                        line=dict(color="rgba(31, 119, 180, 0.2)", width=1),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
        elif mode == "quantiles":
            grouped = sims.groupby("time")[val_col]
            q05 = grouped.quantile(0.05)
            q50 = grouped.quantile(0.50)
            q95 = grouped.quantile(0.95)
            times = q05.index

            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([times, times[::-1]]),
                    y=np.concatenate([q95, q05[::-1]]),
                    fill="toself",
                    fillcolor="rgba(31, 119, 180, 0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="90% Interval",
                    showlegend=(i == 0),
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                Scatter(
                    x=times,
                    y=q50,
                    mode="lines",
                    line=dict(color="rgba(31, 119, 180, 0.8)", width=2),
                    name="Median Sim",
                    showlegend=(i == 0),
                ),
                row=row,
                col=col,
            )

        fig.add_trace(
            Scatter(
                x=obs.index if "time" not in obs.columns else obs["time"],
                y=obs[var],
                mode="markers+lines",
                marker=dict(color="red", size=6),
                line=dict(color="red", width=1),
                name="Actual Data",
                showlegend=(i == 0),
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=title,
        height=min(1200, 400 * rows),
        width=1000,
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_panel_simulations_internal(
    sims: pd.DataFrame,
    obs: pd.DataFrame,
    mode: str = "lines",
    title: str = "Panel Simulations vs Data",
) -> Any:
    """
    Internal function to plot panel simulations vs data using Plotly.
    Facets by unit.
    """
    go, px, make_subplots = _check_plotly()

    units = sorted(obs["unit"].unique())
    n_units = len(units)

    obs_cols = [c for c in obs.columns if c not in ["time", "theta_idx", "sim", "unit"]]
    var = obs_cols[0]  # Take first variable for now

    # Map var to sims column (sims uses obs_0, obs_1 etc if generated via simulate)
    val_col = var if var in sims.columns else "obs_0"

    cols = 3 if n_units > 3 else n_units
    rows = (n_units + cols - 1) // cols if n_units > 0 else 1

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"Unit: {u}" for u in units],
        shared_xaxes=True,
        vertical_spacing=0.05,
    )

    total_points = len(sims)
    Scatter = go.Scattergl if total_points > 100000 else go.Scatter

    for i, unit in enumerate(units):
        row = (i // cols) + 1
        col = (i % cols) + 1

        u_sims = sims[sims["unit"] == unit]
        u_obs = obs[obs["unit"] == unit]

        if mode == "lines":
            for sim_idx in u_sims["sim"].unique():
                sub = u_sims[u_sims["sim"] == sim_idx]
                fig.add_trace(
                    Scatter(
                        x=sub["time"],
                        y=sub[val_col],
                        mode="lines",
                        line=dict(color="rgba(31, 119, 180, 0.2)", width=1),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
        elif mode == "quantiles":
            grouped = u_sims.groupby("time")[val_col]
            q05 = grouped.quantile(0.05)
            q50 = grouped.quantile(0.50)
            q95 = grouped.quantile(0.95)
            times = q05.index

            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([times, times[::-1]]),
                    y=np.concatenate([q95, q05[::-1]]),
                    fill="toself",
                    fillcolor="rgba(31, 119, 180, 0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="90% Interval",
                    showlegend=(i == 0),
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                Scatter(
                    x=times,
                    y=q50,
                    mode="lines",
                    line=dict(color="rgba(31, 119, 180, 0.8)", width=2),
                    name="Median Sim",
                    showlegend=(i == 0),
                ),
                row=row,
                col=col,
            )

        # Plot actual data
        fig.add_trace(
            Scatter(
                x=u_obs["time"],
                y=u_obs[var],
                mode="markers+lines",
                marker=dict(color="red", size=6),
                line=dict(color="red", width=1),
                name="Actual Data",
                showlegend=(i == 0),
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=f"{title} ({var})",
        height=min(2000, 300 * rows + 100),
        width=1000,
        template="plotly_white",
    )
    return fig
