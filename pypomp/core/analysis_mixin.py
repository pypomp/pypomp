from __future__ import annotations
from typing import Any
import pandas as pd
import jax

from .parameters import PompParameters
from .viz import plot_traces_internal, plot_simulations_internal

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interfaces import PompInterface as Base
else:
    Base = object


class PompAnalysisMixin(Base):
    """
    Mixin class that implements analysis, diagnostics, and plotting methods for Pomp.
    """

    def traces(self) -> pd.DataFrame:
        """Return the full trace of log-likelihoods and parameters over all runs.

        Concatenates the parameter and log-likelihood histories from every
        :meth:`pfilter`, :meth:`mif`, and :meth:`train` call stored in
        :attr:`results_history`.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame containing concatenated trace data. The columns appear
            in the following order:

            1. ``theta_idx``: The index of the parameter set.
            2. ``iteration``: Counter indicating the global iteration.
            3. ``method``: The name of the method (e.g. ``'pfilter'``, ``'mif'``, ``'train'``).
            4. ``logLik``: The estimated log-likelihood.
            5. ``se``: The standard error of the log-likelihood estimate.
            6. Parameter columns: One column per model parameter in their defined order.
        """
        return self.results_history.traces()

    def results(self, index: int = -1, ignore_nan: bool = False) -> pd.DataFrame:
        """Return a summary DataFrame for one run from the results history.

        Retrieves the final log-likelihoods and parameter values for all
        replicates associated with the run at position ``index`` in
        :attr:`results_history`.

        Parameters
        ----------
        index : int, optional
            Position in :attr:`results_history` to retrieve.  Defaults to
            ``-1`` (the most recent run).
        ignore_nan : bool, optional
            If ``True``, NaN log-likelihoods are excluded when computing
            the summary.  Defaults to ``False``.

        Returns
        -------
        pd.DataFrame
            Tidy DataFrame with columns ``logLik`` and one column per
            parameter, indexed by ``theta_idx``.

        See Also
        --------
        pypomp.core.results.Result.to_dataframe : Dataframe returned by :class:`~pypomp.core.results.Result` class.
        """
        return self.results_history.results(index=index, ignore_nan=ignore_nan)

    def CLL(self, index: int = -1, average: bool = False) -> pd.DataFrame:
        """Return conditional log-likelihoods from a particle filter run.

        Parameters
        ----------
        index : int, optional
            Position in :attr:`results_history` to retrieve.  Defaults to
            ``-1`` (the most recent run).  The indexed result must be a
            :meth:`pfilter` result with ``CLL=True``.
        average : bool, optional
            If ``True``, average the CLLs over replicates using logmeanexp.
            Defaults to ``False``.

        Returns
        -------
        pd.DataFrame
            Tidy DataFrame of conditional log-likelihoods. The columns appear
            in the following order:

            1. ``theta_idx``: The index of the parameter set.
            2. ``rep``: The replicate index (only if ``average=False``).
            3. ``time``: The observation time point.
            4. ``CLL``: The conditional log-likelihood value.
        """
        return self.results_history.CLL(index=index, average=average)

    def ESS(self, index: int = -1, average: bool = False) -> pd.DataFrame:
        """Return effective sample sizes from a particle filter run.

        Parameters
        ----------
        index : int, optional
            Position in :attr:`results_history` to retrieve.  Defaults to
            ``-1`` (the most recent run).  The indexed result must be a
            :meth:`pfilter` result with ``ESS=True``.
        average : bool, optional
            If ``True``, average the ESS over replicates using arithmetic
            mean.  Defaults to ``False``.

        Returns
        -------
        pd.DataFrame
            Tidy DataFrame of ESS values. The columns appear in the following order:

            1. ``theta_idx``: The index of the parameter set.
            2. ``rep``: The replicate index (only if ``average=False``).
            3. ``time``: The observation time point.
            4. ``ESS``: The Effective Sample Size value.
        """
        return self.results_history.ESS(index=index, average=average)

    def time(self) -> pd.DataFrame:
        """Return a summary of wall-clock execution times for all runs.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``method`` (e.g. ``'pfilter'``,
            ``'mif'``) and ``time`` (execution time in seconds).
        """
        return self.results_history.time()

    def prune(self, n: int = 1, refill: bool = True) -> None:
        """Keep the top ``n`` parameter sets by log-likelihood.

        Discards poorly performing starting points after an estimation run,
        focusing subsequent work on the most promising candidates.  If
        ``refill`` is ``True``, the surviving sets are duplicated to restore
        the original number of replicates.

        Parameters
        ----------
        n : int, optional
            Number of top parameter sets to retain.  Defaults to ``1``.
        refill : bool, optional
            If ``True``, repeat the top ``n`` sets to match the previous
            number of replicates.  Defaults to ``True``.
        """
        self.theta = self.theta.pruned(n=n, refill=refill)

    def plot_traces(self, show: bool = True) -> Any:
        """Plot parameter and log-likelihood traces from the results history.

        Produces an interactive Plotly figure with one facet per parameter
        and one for ``logLik``.  Lines connect :meth:`mif` / :meth:`train`
        points for each replicate; :meth:`pfilter` runs appear as dots.
        Replicates are distinguished by colour.

        Parameters
        ----------
        show : bool, optional
            Whether to call ``fig.show()`` before returning.  Defaults to
            ``True``.

        Returns
        -------
        plotly.graph_objects.Figure or None
            The Plotly figure object, or ``None`` if no results are stored.
        """
        traces = self.traces()
        fig = plot_traces_internal(traces, title="Pomp Traces")

        if fig is not None and show:
            fig.show()
        return fig

    def plot_simulations(
        self,
        key: jax.Array,
        nsim: int = 20,
        mode: str = "lines",
        theta: PompParameters | None = None,
        show: bool = True,
    ) -> Any:
        """Plot simulated trajectories alongside the observed data.

        Generates an interactive Plotly figure overlaying ``nsim`` simulated
        observation trajectories on the actual ``ys`` data, helping to
        assess qualitative goodness-of-fit.

        Parameters
        ----------
        key : jax.Array
            JAX random key for simulation.
        nsim : int, optional
            Number of simulation replicates.  Defaults to ``20``.
        mode : str, optional
            Plot mode: ``"lines"`` shows individual trajectories;
            ``"quantiles"`` shows a shaded quantile band.  Defaults to
            ``"lines"``.
        theta : PompParameters or None, optional
            Parameter set to simulate from.  Defaults to the first
            replicate of :attr:`theta`.
        show : bool, optional
            Whether to call ``fig.show()`` before returning.  Defaults to
            ``True``.

        Returns
        -------
        plotly.graph_objects.Figure or None
            The Plotly figure object.
        """
        if theta is None:
            theta = (
                self.theta.subset([0])
                if self.theta and self.theta.num_replicates() > 1
                else self.theta
            )
        elif not isinstance(theta, PompParameters):
            raise TypeError("theta must be a PompParameters instance")

        _, sims = self.simulate(nsim=nsim, theta=theta, key=key)
        fig = plot_simulations_internal(sims, self.ys, mode=mode)

        if fig is not None and show:
            fig.show()
        return fig

    def print_summary(self, n: int = 5) -> None:
        """Print a high-level summary of the model and its estimation history.

        Displays basic model statistics (number of observations, time steps,
        parameters, and parameter replicates) followed by a tabular summary
        of :attr:`results_history` listing each run's method and
        performance metrics.

        Parameters
        ----------
        n : int, optional
            Maximum number of history entries to display.  Defaults to
            ``5``.
        """
        print("Basics:")
        print("-------")
        print(f"Number of observations: {len(self.ys)}")
        print(f"Number of time steps: {len(self._dt_array_extended)}")
        print(f"Number of parameters: {self.theta.num_params()}")
        print(f"Number of parameter sets: {self.theta.num_replicates()}")
        print()
        self.results_history.print_summary(n=n)
