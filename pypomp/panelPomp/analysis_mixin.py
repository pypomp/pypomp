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
        df = self.results_history[index].to_dataframe()
        if df.empty or "shared logLik" not in df.columns:
            raise ValueError("No log-likelihoods found in results(index).")

        replicate_logliks = df.groupby("replicate")["shared logLik"].first()
        top_indices = replicate_logliks.to_numpy().argsort()[-n:][::-1]
        index_list = list(replicate_logliks.index)
        top_replicates = [index_list[i] for i in top_indices]

        res = self.results_history[index]
        shared_list = res.shared
        unit_specific_list = res.unit_specific

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
        df = self.results_history[index].to_dataframe()
        if df.empty or "shared logLik" not in df.columns:
            raise ValueError("No log-likelihoods found in results(index).")

        res = self.results_history[index]
        shared_list = res.shared
        unit_specific_list = res.unit_specific

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
        return self.results_history.results(index=index, ignore_nan=ignore_nan)


    def time(self):
        return self.results_history.time()

    def traces(self) -> pd.DataFrame:
        return self.results_history.traces()

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
