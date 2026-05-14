import jax
import pandas as pd
from typing import TYPE_CHECKING, Any
from ..core.viz import plot_traces_internal, plot_panel_simulations_internal

if TYPE_CHECKING:
    from .interfaces import PanelPompInterface as Base
else:
    Base = object  # At runtime, this is just a normal class


class PanelAnalysisMixin(Base):
    """
    Handles results processing, pruning, and visualization for PanelPomp.
    """

    @property
    def ys(self) -> pd.DataFrame:
        """
        Returns a tidy DataFrame with the observations from all units.
        """
        ys_list = []
        for unit, obj in self.unit_objects.items():
            unit_ys = obj.ys.copy()
            unit_ys.insert(0, "unit", unit)
            unit_ys.insert(0, "time", unit_ys.index)
            ys_list.append(unit_ys)
        return pd.concat(ys_list).reset_index(drop=True)

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
        """
        Returns a tidy DataFrame with the results of the method run at the given index.

        Args:
            index (int, optional): The index of the result to retrieve. Defaults to -1.
            ignore_nan (bool, optional): Boolean flag controlling whether to ignore
                NaN values in the log-likelihoods. Defaults to False.
        Returns:
            pd.DataFrame: A DataFrame with the log-likelihoods and parameters used.
        """
        return self.results_history.results(index=index, ignore_nan=ignore_nan)

    def CLL(self, index: int = -1, average: bool = False) -> pd.DataFrame:
        """
        Returns a tidy DataFrame with the conditional log-likelihoods of the method run at the given index.

        Args:
            index (int, optional): The index of the result to retrieve. Defaults to -1.
            average (bool, optional): Boolean flag controlling whether to average
                the conditional log-likelihoods over replicates using logmeanexp.
                Defaults to False.
        Returns:
            pd.DataFrame: A DataFrame with the conditional log-likelihoods.
        """
        return self.results_history.CLL(index=index, average=average)

    def ESS(self, index: int = -1, average: bool = False) -> pd.DataFrame:
        """
        Returns a tidy DataFrame with the effective sample size of the method run at the given index.

        Args:
            index (int, optional): The index of the result to retrieve. Defaults to -1.
            average (bool, optional): Boolean flag controlling whether to average
                the effective sample size over replicates using arithmetic mean.
                Defaults to False.
        Returns:
            pd.DataFrame: A DataFrame with the effective sample size.
        """
        return self.results_history.ESS(index=index, average=average)

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

    def plot_traces(self, which: str = "shared", show: bool = True) -> Any:
        """
        Plots the parameter and log-likelihood traces from the entire result history.
        """
        traces = self.traces()
        assert isinstance(traces, pd.DataFrame)
        if traces.empty:
            print("No trace data to plot.")
            return None

        value_cols = [
            c
            for c in traces.columns
            if c not in ["theta_idx", "iteration", "method", "unit"]
        ]

        has_shared_rows = bool((traces["unit"] == "shared").any())
        shared_params = []
        if has_shared_rows:
            shared_rows = traces.loc[traces["unit"] == "shared"]
            for c in value_cols:
                if c != "logLik" and pd.notna(shared_rows[c]).any():
                    shared_params.append(c)

        unit_params = [
            c for c in value_cols if c != "logLik" and c not in shared_params
        ]

        if which == "shared":
            if not has_shared_rows:
                print("No shared rows to plot.")
                return None
            df_plot = traces[traces["unit"] == "shared"]
            title = "Shared Parameter Traces"
        elif which == "unitLogLik":
            df_plot = traces[traces["unit"] != "shared"][
                ["theta_idx", "iteration", "method", "unit", "logLik"]
            ]
            title = "Unit Log-Likelihood Traces"
        else:
            if which not in unit_params:
                raise ValueError(
                    f"'{which}' not found among unit-specific parameters: {unit_params}"
                )
            df_plot = traces[traces["unit"] != "shared"][
                ["theta_idx", "iteration", "method", "unit", which]
            ]
            title = f"Unit Parameter Traces: {which}"

        fig = plot_traces_internal(df_plot, title=title)

        if fig is not None and show:
            fig.show()
        return fig

    def plot_simulations(
        self,
        key: jax.Array,
        nsim: int = 20,
        mode: str = "lines",
        theta: Any = None,
        show: bool = True,
    ) -> Any:
        """
        Runs simulations for the PanelPomp model and plots them against true data.

        Args:
            key (jax.Array): JAX random key for simulation.
            nsim (int): Number of simulations to perform per parameter set.
            mode (str): Plotting mode, either "lines" or "quantiles".
            theta (PanelParameters, optional): Parameters to use. Defaults to self.theta.
            show (bool): Whether to display the plot.
        """
        if theta is None:
            theta = (
                self.theta.subset([0])
                if self.theta and self.theta.num_replicates() > 1
                else self.theta
            )

        _, sims = self.simulate(nsim=nsim, theta=theta, key=key)
        fig = plot_panel_simulations_internal(sims, self.ys, mode=mode)

        if fig is not None and show:
            fig.show()
        return fig
