import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interfaces import PanelPompInterface as Base
else:
    Base = object  # At runtime, this is just a normal class


class PanelAnalysisMixin(Base):
    """
    Handles results processing, pruning, and visualization for PanelPomp.
    """

    def prune(self, n: int = 1, refill: bool = True):
        """
        Prune the parameter sets to the top `n` based on log-likelihoods stored in the PanelParameters object under theta.

        Args:
            n: Number of top parameter sets to keep.
            refill: If True, repeat the top `n` parameter sets to match the
                previous number of replicates. If False, keep only the `n` sets.
        """
        self.theta.prune(n=n, refill=refill)

    def mix_and_match(self):
        """
        Sorts unit-specific parameters and shared parameters in descending order of unit log-likelihood and shared log-likelihood, respectively, then combines them to form new parameter sets. The nth best parameter for a given unit or for the shared parameters is placed in the nth parameter set.
        """
        self.theta.mix_and_match()

    def results(self, index: int = -1, ignore_nan: bool = False) -> pd.DataFrame:
        return self.results_history.results(index=index, ignore_nan=ignore_nan)

    def time(self):
        """
        Return a DataFrame summarizing the execution times of methods run.

        Returns:
            pd.DataFrame: A DataFrame where each row contains:
                - 'method': The name of the method run.
                - 'time': The execution time in seconds.
        """
        return self.results_history.time()

    def traces(self) -> pd.DataFrame:
        """
        Return a tidy DataFrame with the full trace of log-likelihoods and parameters from the entire result history.
        """
        return self.results_history.traces()

    def plot_traces(self, which: str = "shared", show: bool = True):
        """
        Plots the parameter and log-likelihood traces from the entire result history.
        """
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
