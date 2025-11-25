import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import TYPE_CHECKING
from pypomp.util import logmeanexp

if TYPE_CHECKING:
    from .interfaces import PanelPompInterface as Base
else:
    Base = object  # At runtime, this is just a normal class


class PanelAnalysisMixin(Base):
    """
    Handles results processing, pruning, and visualization for PanelPomp.
    """

    def prune(self, n: int = 1, index: int = -1, refill: bool = True):
        df = self.results(index)
        if df.empty or "shared logLik" not in df.columns:
            raise ValueError("No log-likelihoods found in results(index).")

        replicate_logliks = df.groupby("replicate")["shared logLik"].first()
        top_indices = replicate_logliks.to_numpy().argsort()[-n:][::-1]
        index_list = list(replicate_logliks.index)
        top_replicates = [index_list[i] for i in top_indices]

        res = self.results_history[index]
        shared_list = res.get("shared")
        unit_specific_list = res.get("unit_specific")

        if shared_list is not None:
            top_shared = [shared_list[rep_idx] for rep_idx in top_replicates]
        else:
            top_shared = None

        if unit_specific_list is not None:
            top_unit_specific = [
                unit_specific_list[rep_idx] for rep_idx in top_replicates
            ]
        else:
            top_unit_specific = None

        if refill:
            prev_shared_len = len(self.shared) if self.shared is not None else 0
            prev_unit_specific_len = (
                len(self.unit_specific) if self.unit_specific is not None else 0
            )
            prev_len = max(prev_shared_len, prev_unit_specific_len)

            if prev_len > 0:
                if top_shared is not None:
                    repeats = (prev_len + n - 1) // n
                    new_shared = (top_shared * repeats)[:prev_len]
                else:
                    new_shared = None

                if top_unit_specific is not None:
                    repeats = (prev_len + n - 1) // n
                    new_unit_specific = (top_unit_specific * repeats)[:prev_len]
                else:
                    new_unit_specific = None
            else:
                new_shared = top_shared
                new_unit_specific = top_unit_specific
        else:
            new_shared = top_shared
            new_unit_specific = top_unit_specific

        self.shared = new_shared
        self.unit_specific = new_unit_specific

    def mix_and_match(self, index: int = -1):
        df = self.results(index)
        if df.empty or "shared logLik" not in df.columns:
            raise ValueError("No log-likelihoods found in results(index).")

        res = self.results_history[index]
        shared_list = res.get("shared")
        unit_specific_list = res.get("unit_specific")

        replicate_logliks = df.groupby("replicate")["shared logLik"].first()
        n = len(replicate_logliks)

        shared_loglik_values = replicate_logliks.to_numpy()
        shared_ranked_indices = shared_loglik_values.argsort()[::-1]
        replicate_index_list = list(replicate_logliks.index)
        shared_ranked_replicates = [
            replicate_index_list[i] for i in shared_ranked_indices
        ]

        unit_names = list(self.unit_objects.keys())

        if "unit logLik" not in df.columns:
            raise ValueError("No unit logLik found in results.")

        unit_loglik_pivot = df.pivot(
            index="replicate", columns="unit", values="unit logLik"
        )
        unit_ranked_replicates = {}
        for unit in unit_names:
            if unit not in unit_loglik_pivot.columns:
                raise ValueError(f"Unit '{unit}' not found in results.")
            unit_logliks = unit_loglik_pivot[unit]
            unit_ranked_indices = unit_logliks.to_numpy().argsort()[::-1]
            unit_index_list = list(unit_logliks.index)
            unit_ranked_replicates[unit] = [
                unit_index_list[i] for i in unit_ranked_indices
            ]

        new_shared_list: list[pd.DataFrame] | None = (
            [] if shared_list is not None else None
        )
        new_unit_specific_list: list[pd.DataFrame] | None = (
            [] if unit_specific_list is not None else None
        )

        for i in range(n):
            if shared_list is not None and new_shared_list is not None:
                shared_rep_idx = shared_ranked_replicates[i]
                new_shared_list.append(shared_list[shared_rep_idx].copy())

            if unit_specific_list is not None and new_unit_specific_list is not None:
                new_unit_df_data = {}
                for unit in unit_names:
                    unit_rep_idx = unit_ranked_replicates[unit][i]
                    unit_df = unit_specific_list[unit_rep_idx]
                    if unit in unit_df.columns:
                        new_unit_df_data[unit] = unit_df[unit].copy()
                    else:
                        raise ValueError(
                            f"Unit '{unit}' not found in unit_specific DataFrame for replicate {unit_rep_idx}."
                        )

                new_unit_df = pd.DataFrame(
                    new_unit_df_data, index=unit_specific_list[0].index
                )
                new_unit_specific_list.append(new_unit_df)

        self.shared = new_shared_list
        self.unit_specific = new_unit_specific_list

    def results(self, index: int = -1, ignore_nan: bool = False) -> pd.DataFrame:
        res = self.results_history[index]
        method = res.get("method", None)
        if method == "mif":
            return self._results_from_mif(res)
        elif method == "pfilter":
            return self._results_from_pfilter(res, ignore_nan=ignore_nan)
        else:
            raise ValueError(f"Unknown method '{method}' for results()")

    def _results_from_mif(self, res) -> pd.DataFrame:
        shared_da = res["shared_traces"]
        unit_da = res["unit_traces"]
        full_logliks = res["logLiks"]

        all_shared_vars = list(shared_da.coords["variable"].values)
        shared_names = all_shared_vars[1:] if len(all_shared_vars) > 1 else []

        all_unit_vars = list(unit_da.coords["variable"].values)
        unit_names = list(unit_da.coords["unit"].values)
        n_reps = shared_da.sizes["replicate"]

        shared_final_values = shared_da.isel(iteration=-1).values
        unit_final_values = unit_da.isel(iteration=-1).values

        shared_logliks = full_logliks[:, 0].values
        unit_logliks = full_logliks[:, 1:].values

        rep_indices = np.repeat(np.arange(n_reps), len(unit_names))
        unit_indices = np.tile(np.arange(len(unit_names)), n_reps)

        data = {
            "replicate": rep_indices,
            "unit": [unit_names[i] for i in unit_indices],
            "shared logLik": shared_logliks[rep_indices],
            "unit logLik": unit_logliks[rep_indices, unit_indices],
        }

        if shared_names:
            shared_param_values = shared_final_values[:, 1:]
            for i, param_name in enumerate(shared_names):
                data[param_name] = shared_param_values[rep_indices, i]

        unit_param_values = unit_final_values[:, 1:, :]
        for i, param_name in enumerate(all_unit_vars[1:]):
            data[param_name] = unit_param_values[rep_indices, i, unit_indices]

        return pd.DataFrame(data)

    def _results_from_pfilter(self, res, ignore_nan) -> pd.DataFrame:
        logLiks = res["logLiks"]
        shared_params = res.get("shared", [])
        unit_specific_params = res.get("unit_specific", [])

        unit_names = list(logLiks.coords["unit"].values)
        n_reps = logLiks.sizes["theta"]

        logliks_array = logLiks.values

        unit_logliks = np.array(
            [
                [
                    logmeanexp(logliks_array[rep, unit_idx, :], ignore_nan=ignore_nan)
                    for unit_idx in range(len(unit_names))
                ]
                for rep in range(n_reps)
            ]
        )

        shared_logliks = np.sum(unit_logliks, axis=1)

        rep_indices = np.repeat(np.arange(n_reps), len(unit_names))
        unit_indices = np.tile(np.arange(len(unit_names)), n_reps)

        data = {
            "replicate": rep_indices,
            "unit": [unit_names[i] for i in unit_indices],
            "shared logLik": shared_logliks[rep_indices],
            "unit logLik": unit_logliks[rep_indices, unit_indices],
        }

        if shared_params and len(shared_params) > 0:
            shared_param_data = {}
            for rep in range(min(n_reps, len(shared_params))):
                shared_df = shared_params[rep]
                if hasattr(shared_df, "values") and shared_df.shape[1] >= 1:
                    shared_vals = shared_df.iloc[:, 0].values
                    shared_names = (
                        list(shared_df.columns) if hasattr(shared_df, "columns") else []
                    )
                    for i, param_name in enumerate(shared_names):
                        if param_name not in shared_param_data:
                            shared_param_data[param_name] = np.full(n_reps, np.nan)
                        shared_param_data[param_name][rep] = (
                            shared_vals[i] if i < len(shared_vals) else np.nan
                        )

            for param_name, values in shared_param_data.items():
                data[param_name] = values[rep_indices]

        if unit_specific_params and len(unit_specific_params) > 0:
            unit_param_data = {}
            for rep in range(min(n_reps, len(unit_specific_params))):
                unit_df = unit_specific_params[rep]
                if hasattr(unit_df, "columns"):
                    unit_param_names = (
                        list(unit_df.index) if hasattr(unit_df, "index") else []
                    )
                    for param_name in unit_param_names:
                        if param_name not in unit_param_data:
                            unit_param_data[param_name] = np.full(
                                (n_reps, len(unit_names)), np.nan
                            )
                        for unit_idx, unit in enumerate(unit_names):
                            if unit in unit_df.columns:
                                unit_param_data[param_name][rep, unit_idx] = (
                                    unit_df.loc[param_name, unit]
                                )

            for param_name, values in unit_param_data.items():
                data[param_name] = values[rep_indices, unit_indices]

        return pd.DataFrame(data)

    def time(self):
        rows = []
        for idx, res in enumerate(self.results_history):
            method = res.get("method", None)
            exec_time = res.get("execution_time", None)
            rows.append({"method": method, "time": exec_time})
        df = pd.DataFrame(rows)
        df.index.name = "history_index"
        return df

    def traces(self) -> pd.DataFrame:
        if not self.results_history:
            return pd.DataFrame()

        shared_param_names, unit_param_names = self._get_param_names()
        all_param_names = shared_param_names + unit_param_names

        all_data = []
        global_iters: dict[int, int] = {}

        for res in self.results_history:
            method = res.get("method")
            if method == "mif":
                shared_da = res["shared_traces"]
                unit_da = res["unit_traces"]
                unit_names = list(unit_da.coords["unit"].values)
                shared_vars = list(shared_da.coords["variable"].values)
                unit_vars = list(unit_da.coords["variable"].values)

                n_rep = shared_da.sizes["replicate"]
                n_iter = shared_da.sizes["iteration"]

                shared_values = shared_da.values
                unit_values = unit_da.values

                shared_param_indices = {
                    name: i for i, name in enumerate(shared_vars[1:])
                }
                unit_param_indices = {name: i for i, name in enumerate(unit_vars[1:])}

                for rep_idx in range(n_rep):
                    if rep_idx not in global_iters:
                        global_iters[rep_idx] = 0

                    for iter_idx in range(n_iter):
                        shared_loglik = float(shared_values[rep_idx, iter_idx, 0])
                        shared_params = {
                            name: float(
                                shared_values[
                                    rep_idx, iter_idx, shared_param_indices[name] + 1
                                ]
                            )
                            for name in shared_param_indices
                        }

                        shared_row = {
                            "replicate": rep_idx,
                            "unit": "shared",
                            "iteration": global_iters[rep_idx],
                            "method": "mif",
                            "logLik": shared_loglik,
                        }
                        for name in all_param_names:
                            shared_row[name] = shared_params.get(name, float("nan"))
                        all_data.append(shared_row)

                        for unit_idx, unit in enumerate(unit_names):
                            unit_loglik = float(
                                unit_values[rep_idx, iter_idx, 0, unit_idx]
                            )
                            unit_params = {
                                name: float(
                                    unit_values[
                                        rep_idx,
                                        iter_idx,
                                        unit_param_indices[name] + 1,
                                        unit_idx,
                                    ]
                                )
                                for name in unit_param_indices
                            }

                            unit_row = {
                                "replicate": rep_idx,
                                "unit": str(unit),
                                "iteration": global_iters[rep_idx],
                                "method": "mif",
                                "logLik": unit_loglik,
                            }
                            for name in all_param_names:
                                if name in unit_params:
                                    unit_row[name] = unit_params[name]
                                elif name in shared_params:
                                    unit_row[name] = shared_params[name]
                                else:
                                    unit_row[name] = float("nan")
                            all_data.append(unit_row)

                        global_iters[rep_idx] += 1

            elif method == "pfilter":
                logLiks = res["logLiks"]
                unit_names = list(logLiks.coords["unit"].values)
                shared_list = res.get("shared")
                unit_list = res.get("unit_specific")

                n_theta = logLiks.sizes["theta"]

                logliks_array = logLiks.values
                unit_avgs = np.array(
                    [
                        [
                            logmeanexp(logliks_array[rep_idx, u_i, :])
                            for u_i in range(len(unit_names))
                        ]
                        for rep_idx in range(n_theta)
                    ]
                )
                shared_totals = np.sum(unit_avgs, axis=1)

                shared_param_data = {}
                if isinstance(shared_list, list):
                    for rep_idx in range(min(n_theta, len(shared_list))):
                        df = shared_list[rep_idx]
                        if hasattr(df, "index") and df.shape[1] >= 1:
                            for name in df.index:
                                if name not in shared_param_data:
                                    shared_param_data[name] = np.full(n_theta, np.nan)
                                shared_param_data[name][rep_idx] = float(
                                    df.loc[name, df.columns[0]]
                                )

                unit_param_data = {}
                if isinstance(unit_list, list):
                    for rep_idx in range(min(n_theta, len(unit_list))):
                        df = unit_list[rep_idx]
                        if hasattr(df, "columns"):
                            for name in df.index:
                                if name not in unit_param_data:
                                    unit_param_data[name] = np.full(
                                        (n_theta, len(unit_names)), np.nan
                                    )
                                for unit_idx, unit in enumerate(unit_names):
                                    if str(unit) in df.columns:
                                        unit_param_data[name][rep_idx, unit_idx] = (
                                            float(df.loc[name, str(unit)])
                                        )

                for rep_idx in range(n_theta):
                    iter_val = global_iters.get(rep_idx, 1) - 1
                    iter_val = iter_val if iter_val > 0 else 1

                    shared_row = {
                        "replicate": rep_idx,
                        "unit": "shared",
                        "iteration": iter_val,
                        "method": "pfilter",
                        "logLik": float(shared_totals[rep_idx]),
                    }
                    for name in all_param_names:
                        if name in shared_param_data:
                            shared_row[name] = shared_param_data[name][rep_idx]
                        else:
                            shared_row[name] = float("nan")
                    all_data.append(shared_row)

                    for u_i, unit in enumerate(unit_names):
                        unit_row = {
                            "replicate": rep_idx,
                            "unit": str(unit),
                            "iteration": iter_val,
                            "method": "pfilter",
                            "logLik": float(unit_avgs[rep_idx, u_i]),
                        }
                        for name in all_param_names:
                            if name in unit_param_data:
                                unit_row[name] = unit_param_data[name][rep_idx, u_i]
                            elif name in shared_param_data:
                                unit_row[name] = shared_param_data[name][rep_idx]
                            else:
                                unit_row[name] = float("nan")
                        all_data.append(unit_row)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        if not df.empty:
            df = df.sort_values(["replicate", "unit", "iteration"]).reset_index(
                drop=True
            )
        return df

    def plot_traces(self, which: str = "shared", show: bool = True):
        traces = self.traces()
        assert isinstance(traces, pd.DataFrame)
        if traces.empty:
            print("No trace data to plot.")
            return

        value_cols = [
            c
            for c in traces.columns
            if c not in ["replicate", "iteration", "method", "unit"]
        ]

        has_shared_rows = bool((traces["unit"] == "shared").any())
        shared_params = []
        if has_shared_rows:
            shared_rows = traces.loc[traces["unit"] == "shared"]
            for c in value_cols:
                if c != "logLik" and pd.notna(shared_rows[c]).any():
                    shared_params.append(c)
        else:
            shared_params = []

        unit_params = [
            c for c in value_cols if c != "logLik" and c not in shared_params
        ]

        if which == "shared":
            if not has_shared_rows:
                print("No shared rows to plot.")
                return None
            shared_vars = (["logLik"] if "logLik" in value_cols else []) + shared_params
            if len(shared_vars) == 0:
                print("No shared parameters or logLik to plot.")
                return None

            df_shared = traces.loc[
                traces["unit"] == "shared",
                ["replicate", "iteration", "method", *shared_vars],
            ]
            assert isinstance(df_shared, pd.DataFrame)
            df_shared_long: pd.DataFrame = df_shared.melt(
                id_vars=["replicate", "iteration", "method"],
                value_vars=shared_vars,
                var_name="variable",
                value_name="value",
            )

            g = sns.FacetGrid(
                df_shared_long,
                col="variable",
                sharex=True,
                sharey=False,
                hue="replicate",
                col_wrap=3,
                height=3.5,
                aspect=1.2,
                palette="tab10",
            )

            def facet_plot_shared(data, color, **kwargs):
                for rep, group in data.groupby("replicate"):
                    for method in ["mif", "train"]:
                        sub = group[group["method"] == method]
                        if len(sub) > 1:
                            plt.plot(
                                sub["iteration"],
                                sub["value"],
                                "-",
                                color=color,
                                alpha=0.8,
                            )
                        elif len(sub) == 1:
                            plt.scatter(
                                sub["iteration"],
                                sub["value"],
                                color=color,
                                marker="o",
                                alpha=0.8,
                            )
                    sub = group[group["method"] == "pfilter"]
                    if not sub.empty:
                        plt.scatter(
                            sub["iteration"],
                            sub["value"],
                            color=color,
                            marker="o",
                            edgecolor="k",
                            zorder=3,
                        )

            g.map_dataframe(facet_plot_shared)
            g.add_legend(title="Replicate")
            g.set_axis_labels("Iteration", "Value")
            g.set_titles(col_template="{col_name}")
            plt.tight_layout()
            if show:
                plt.show()
            return g

        if which == "unitLogLik":
            df_ul = traces.loc[
                traces["unit"] != "shared",
                ["replicate", "iteration", "method", "unit", "logLik"],
            ].rename(columns={"logLik": "value"})
            df_ul = df_ul.loc[pd.notna(df_ul["value"])]
            if bool(df_ul.empty):
                print("No unit-specific logLik data to plot.")
                return None

            g = sns.FacetGrid(
                df_ul,
                col="unit",
                sharex=True,
                sharey=False,
                hue="replicate",
                col_wrap=4,
                height=3.2,
                aspect=1.1,
                palette="tab10",
            )

            def facet_plot_units_ll(data, color, **kwargs):
                for rep, group in data.groupby("replicate"):
                    for method in ["mif", "train"]:
                        sub = group[group["method"] == method]
                        if len(sub) > 1:
                            plt.plot(
                                sub["iteration"],
                                sub["value"],
                                "-",
                                color=color,
                                alpha=0.8,
                            )
                        elif len(sub) == 1:
                            plt.scatter(
                                sub["iteration"],
                                sub["value"],
                                color=color,
                                marker="o",
                                alpha=0.8,
                            )
                    sub = group[group["method"] == "pfilter"]
                    if not sub.empty:
                        plt.scatter(
                            sub["iteration"],
                            sub["value"],
                            color=color,
                            marker="o",
                            edgecolor="k",
                            zorder=3,
                        )

            g.map_dataframe(facet_plot_units_ll)
            g.add_legend(title="Replicate")
            g.set_axis_labels("Iteration", "logLik")
            g.set_titles(col_template="{col_name}")
            plt.tight_layout()
            if show:
                plt.show()
            return g

        if which not in unit_params:
            raise ValueError(
                f"'{which}' not found among unit-specific parameters: {unit_params}"
            )

        df_param = traces.loc[
            :, ["replicate", "iteration", "method", "unit", which]
        ].copy()
        assert isinstance(df_param, pd.DataFrame)
        df_param = df_param.loc[pd.notna(df_param[which])]
        if bool(df_param.empty):
            print(f"No data to plot for unit-specific parameter '{which}'.")
            return None
        df_param = df_param.rename(columns={which: "value"})

        g = sns.FacetGrid(
            df_param,
            col="unit",
            sharex=True,
            sharey=False,
            hue="replicate",
            col_wrap=4,
            height=3.2,
            aspect=1.1,
            palette="tab10",
        )

        def facet_plot_units(data, color, **kwargs):
            for rep, group in data.groupby("replicate"):
                for method in ["mif", "train"]:
                    sub = group[group["method"] == method]
                    if len(sub) > 1:
                        plt.plot(
                            sub["iteration"], sub["value"], "-", color=color, alpha=0.8
                        )
                    elif len(sub) == 1:
                        plt.scatter(
                            sub["iteration"],
                            sub["value"],
                            color=color,
                            marker="o",
                            alpha=0.8,
                        )
                sub = group[group["method"] == "pfilter"]
                if not sub.empty:
                    plt.scatter(
                        sub["iteration"],
                        sub["value"],
                        color=color,
                        marker="o",
                        edgecolor="k",
                        zorder=3,
                    )

        g.map_dataframe(facet_plot_units)
        g.add_legend(title="Replicate")
        g.set_axis_labels("Iteration", which)
        g.set_titles(col_template="{col_name}")
        plt.tight_layout()
        if show:
            plt.show()
        return g
