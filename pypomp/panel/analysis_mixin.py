import warnings
import jax
import pandas as pd
from typing import TYPE_CHECKING, Any
from ..core.viz import plot_traces_internal, plot_panel_simulations_internal

from ..core.parameters import PanelParameters

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
        """Tidy DataFrame containing observations from all units.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame of observations.
        """
        ys_list = []
        for unit, obj in self.unit_objects.items():
            unit_ys = obj.ys.copy()
            unit_ys.insert(0, "unit", unit)
            unit_ys.insert(0, "time", unit_ys.index)
            ys_list.append(unit_ys)
        return pd.concat(ys_list).reset_index(drop=True)

    def prune(self, n: int = 1, refill: bool = True) -> None:
        """Prune replicates down to the top ``n`` by log-likelihood.

        Parameters
        ----------
        n : int, optional
            Number of top parameter sets to keep.  Defaults to ``1``.
        refill : bool, optional
            If ``True`` (default), duplicate the top ``n`` sets to restore the
            original number of replicates.
        """
        self.theta = self.theta.pruned(n=n, refill=refill)

    def mix_and_match(self) -> None:
        """Sort parameters independently and cross-combine them.

        Ranks the shared parameters of all replicates based on their overall
        panel log-likelihoods.  Independently, for each unit, ranks the
        unit-specific parameters of all replicates based on that unit's
        individual log-likelihoods.

        Finally, reconstructs the parameter set by pairing them by rank: the
        ``n``-th new replicate is formed by combining the ``n``-th best shared
        parameter set with the ``n``-th best unit-specific parameter set for
        each unit.  This cross-combines the best-performing components from
        different replicates to construct potentially superior new parameter sets.

        In particular, if all parameter sets share the same values for the
        shared parameters, then the overall panel likelihood factors
        completely across units.  Consequently, the new best replicate (rank 0)
        is guaranteed to have a panel log-likelihood equal to the sum of the
        maximum log-likelihoods obtained for each unit individually across all
        original replicates.
        """
        self.theta = self.theta.mixed_and_matched()

    def results(self, index: int = -1, ignore_nan: bool = False) -> pd.DataFrame:
        """Get the results DataFrame for the specified history index.

        Parameters
        ----------
        index : int, optional
            History index.  Defaults to ``-1`` (the last result).
        ignore_nan : bool, optional
            Whether to ignore NaNs when computing log-likelihoods and standard errors.  Defaults to ``False``.

        Returns
        -------
        pd.DataFrame
            DataFrame of results.
        """
        return self.results_history.results(index=index, ignore_nan=ignore_nan)

    def CLL(self, index: int = -1, average: bool = False) -> pd.DataFrame:
        """Get conditional log-likelihoods for the specified history index.

        Parameters
        ----------
        index : int, optional
            History index.  Defaults to ``-1`` (the last result).
        average : bool, optional
            Whether to average conditional log-likelihoods over replicates
            in likelihood space.  Defaults to ``False``.

        Returns
        -------
        pd.DataFrame
            Tidy DataFrame of conditional log-likelihoods. The columns appear
            in the following order:

            1. ``theta_idx``: The index of the parameter set.
            2. ``unit``: The unit identifier.
            3. ``rep``: The replicate index (only if ``average=False``).
            4. ``time``: The observation time point.
            5. ``CLL``: The conditional log-likelihood value.
        """
        return self.results_history.CLL(index=index, average=average)

    def ESS(self, index: int = -1, average: bool = False) -> pd.DataFrame:
        """Get Effective Sample Size for the specified history index.

        Parameters
        ----------
        index : int, optional
            History index.  Defaults to ``-1`` (the last result).
        average : bool, optional
            Whether to average ESS values over replicates.  Defaults to
            ``False``.

        Returns
        -------
        pd.DataFrame
            Tidy DataFrame of ESS values. The columns appear in the following order:

            1. ``theta_idx``: The index of the parameter set.
            2. ``unit``: The unit identifier.
            3. ``rep``: The replicate index (only if ``average=False``).
            4. ``time``: The observation time point.
            5. ``ESS``: The Effective Sample Size value.
        """
        return self.results_history.ESS(index=index, average=average)

    def time(self) -> pd.DataFrame:
        """Get a DataFrame summarizing execution times of run methods.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``'method'`` and ``'time'``.
        """
        return self.results_history.time()

    def traces(self) -> pd.DataFrame:
        """Get a tidy DataFrame with the full trace history of replicates.

        Returns
        -------
        pd.DataFrame
            Tidy DataFrame of the traces. The columns appear in the following order:

            1. ``theta_idx``: The index of the parameter set.
            2. ``unit``: The unit identifier (or ``'shared'`` for shared parameter rows).
            3. ``iteration``: The iteration counter.
            4. ``method``: The name of the method.
            5. ``logLik``: The estimated log-likelihood.
            6. ``se``: The standard error of the log-likelihood.
            7. Parameter columns: Shared and unit-specific parameters sharded by unit.
        """
        return self.results_history.traces()

    def plot_traces(self, which: str = "shared", show: bool = True) -> Any:
        """Plot parameter and log-likelihood traces from the result history.

        Parameters
        ----------
        which : str, optional
            Which parameter trace to plot.  Can be ``"shared"`` (default),
            ``"unitLogLik"``, or the name of a unit-specific parameter.
        show : bool, optional
            Whether to display the plot immediately.  Defaults to ``True``.

        Returns
        -------
        plotly.graph_objects.Figure
            The generated figure object.
        """
        traces = self.traces()
        assert isinstance(traces, pd.DataFrame)
        if traces.empty:
            warnings.warn("No trace data to plot.", UserWarning)
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
                warnings.warn("No shared rows to plot.", UserWarning)
                return None
            df_plot = traces.loc[traces["unit"] == "shared"]
            title = "Shared Parameter Traces"
        elif which == "unitLogLik":
            df_plot = traces.loc[
                traces["unit"] != "shared",
                ["theta_idx", "iteration", "method", "unit", "logLik"],
            ]
            title = "Unit Log-Likelihood Traces"
        else:
            if which not in unit_params:
                raise ValueError(
                    f"'{which}' not found among unit-specific parameters: {unit_params}"
                )
            df_plot = traces.loc[
                traces["unit"] != "shared",
                ["theta_idx", "iteration", "method", "unit", which],
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
        theta: PanelParameters | None = None,
        show: bool = True,
    ) -> Any:
        """Simulate and plot observations against real data.

        Parameters
        ----------
        key : jax.Array
            JAX random key.
        nsim : int, optional
            Number of simulations to run per replicate.  Defaults to ``20``.
        mode : {"lines", "quantiles"}, optional
            Plotting style.  Defaults to ``"lines"``.
        theta : PanelParameters or None, optional
            Parameters to simulate from.  If ``None``, defaults to
            ``self.theta``.
        show : bool, optional
            Whether to display the plot immediately.  Defaults to ``True``.

        Returns
        -------
        plotly.graph_objects.Figure
            The generated figure object.
        """
        if theta is None:
            theta = (
                self.theta.subset([0])
                if self.theta and self.theta.num_replicates() > 1
                else self.theta
            )
        elif not isinstance(theta, PanelParameters):
            raise TypeError("theta must be a PanelParameters instance")

        _, sims = self.simulate(nsim=nsim, theta=theta, key=key)
        fig = plot_panel_simulations_internal(sims, self.ys, mode=mode)

        if fig is not None and show:
            fig.show()
        return fig
