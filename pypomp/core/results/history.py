from __future__ import annotations

from typing import overload, Sequence, Any
import pandas as pd
import warnings
from .result import Result


class ResultsHistory:
    """History log of results from multiple POMP runs.

    Parameters
    ----------
    entries : sequence of Result, optional
        Initial list of results to populate the history.
    """

    def __init__(self, entries: Sequence[Result] | None = None):
        self._entries: list[Result] = list(entries) if entries else []

    def append(self, entry: Result):
        """Add a new result to the history.

        Parameters
        ----------
        entry : Result
            The result object to add.
        """
        self._entries.append(entry)

    def add(self, entry: Result):
        """Alias for append() for backward compatibility."""
        self.append(entry)

    def time(self) -> pd.DataFrame:
        """Return a DataFrame summarizing execution times.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``"method"`` and ``"time"``.
        """
        if not self._entries:
            return pd.DataFrame()
        data = [
            {"method": entry.method, "time": entry.execution_time}
            for entry in self._entries
        ]
        df = pd.DataFrame(data)
        df.index.name = "history_index"
        return df

    def print_summary(self, n: int = 5):
        """Print a summary of the results history."""
        print("Results History:")
        print("----------------")
        if not self._entries:
            print("No results recorded yet.")
            return
        for i, entry in enumerate(self._entries):
            print(f"[{i}] {entry.method.upper()} Result:")
            entry.print_summary(n=n)
            print()

    def __eq__(self, other: Any) -> bool:
        """Structural equality for history."""
        if not isinstance(other, ResultsHistory):
            return False
        if len(self) != len(other):
            return False
        return all(a == b for a, b in zip(self._entries, other._entries))

    @staticmethod
    def merge(*histories: "ResultsHistory") -> "ResultsHistory":
        """Merge multiple histories into one by merging entries at each index.

        Parameters
        ----------
        *histories : ResultsHistory
            One or more histories to merge.  Must all be of the same length.

        Returns
        -------
        ResultsHistory
            A new history containing the pairwise-merged results.

        """
        if not histories:
            return ResultsHistory()
        first = histories[0]
        for h in histories:
            if len(h) != len(first):
                raise ValueError(
                    "All histories must have the same number of entries to be merged."
                )

        merged_entries = []
        for i in range(len(first)):
            entries_to_merge = [h[i] for h in histories]
            merged_entries.append(entries_to_merge[0].merge(*entries_to_merge))
        return ResultsHistory(merged_entries)

    @overload
    def __getitem__(self, index: int) -> Result: ...

    @overload
    def __getitem__(self, index: slice) -> ResultsHistory: ...

    def __getitem__(self, index: int | slice) -> Result | ResultsHistory:
        if isinstance(index, slice):
            return ResultsHistory(self._entries[index])
        return self._entries[index]

    def __len__(self):
        """Get number of entries."""
        return len(self._entries)

    def __iter__(self):
        """Iterate over entries."""
        return iter(self._entries)

    def clear(self):
        """Clear all entries from the history."""
        self._entries.clear()

    def last(self) -> Result:
        """Get last entry."""
        if not self._entries:
            raise ValueError("History is empty")
        return self._entries[-1]

    def results(self, index: int = -1, ignore_nan: bool = False) -> pd.DataFrame:
        """Get the results DataFrame for the entry at the specified index.

        Parameters
        ----------
        index : int, optional
            Index of the result entry.  Defaults to ``-1`` (the last entry).
        ignore_nan : bool, optional
            Whether to ignore NaNs when computing log-likelihoods and standard errors.  Defaults to ``False``.

        Returns
        -------
        pd.DataFrame
            DataFrame of results.
        """
        if not self._entries:
            return pd.DataFrame()
        result = self._entries[index]
        return result.to_dataframe(ignore_nan=ignore_nan)

    def CLL(self, index: int = -1, average: bool = False) -> pd.DataFrame:
        """Get conditional log-likelihoods for the entry at the specified index.

        Parameters
        ----------
        index : int, optional
            Index of the result entry.  Defaults to ``-1`` (the last entry).
        average : bool, optional
            Whether to average over replicates.  Defaults to ``False``.

        Returns
        -------
        pd.DataFrame
            Tidy DataFrame of conditional log-likelihoods. The columns appear
            in the following order:

            For single-unit models:
                - ``theta_idx``: Index of the parameter set.
                - ``rep``: Replicate index (only if ``average=False``).
                - ``time``: Observation time point.
                - ``CLL``: Conditional log-likelihood value.

            For panel models:
                - ``theta_idx``: Index of the parameter set.
                - ``unit``: Unit name/identifier.
                - ``rep``: Replicate index (only if ``average=False``).
                - ``time``: Observation time point.
                - ``CLL``: Conditional log-likelihood value.
        """
        if not self._entries:
            return pd.DataFrame()
        result = self._entries[index]
        return result.CLL(average=average)

    def ESS(self, index: int = -1, average: bool = False) -> pd.DataFrame:
        """Get Effective Sample Size for the entry at the specified index.

        Parameters
        ----------
        index : int, optional
            Index of the result entry.  Defaults to ``-1`` (the last entry).
        average : bool, optional
            Whether to average over replicates.  Defaults to ``False``.

        Returns
        -------
        pd.DataFrame
            Tidy DataFrame of ESS values. The columns appear in the following order:

            For single-unit models:
                - ``theta_idx``: Index of the parameter set.
                - ``rep``: Replicate index (only if ``average=False``).
                - ``time``: Observation time point.
                - ``ESS``: Effective Sample Size value.

            For panel models:
                - ``theta_idx``: Index of the parameter set.
                - ``unit``: Unit name/identifier.
                - ``rep``: Replicate index (only if ``average=False``).
                - ``time``: Observation time point.
                - ``ESS``: Effective Sample Size value.
        """
        if not self._entries:
            return pd.DataFrame()
        result = self._entries[index]
        return result.ESS(average=average)

    def traces(self) -> pd.DataFrame:
        """Return a combined trace of parameters/likelihoods over all entries.

        Returns
        -------
        pd.DataFrame
            DataFrame containing concatenated and iteration-aligned trace data.

            For single-unit history:
                - ``theta_idx``: Index of the parameter set.
                - ``iteration``: Counter indicating the global iteration.
                - ``method``: Estimation/filtering method.
                - ``logLik``: Estimated log-likelihood.
                - ``se``: Standard error of the log-likelihood estimate.
                - Parameter columns: One column per model parameter in defined order.

            For panel history:
                - ``theta_idx``: Index of the parameter set.
                - ``unit``: Unit identifier (or ``'shared'`` for shared parameter rows).
                - ``iteration``: Counter indicating the global iteration.
                - ``method``: Estimation/filtering method.
                - ``logLik``: Estimated log-likelihood.
                - ``se``: Standard error of the log-likelihood.
                - Parameter columns: Shared and unit-specific parameters sharded by unit.
        """
        if not self._entries:
            return pd.DataFrame()

        all_traces = []
        last_iter = 0
        for entry in self._entries:
            t = entry.traces()
            if t.empty:
                continue

            if "iteration" in t.columns:
                shift = last_iter - t["iteration"].min()
                t["iteration"] += shift
                last_iter = int(t["iteration"].max())
            else:
                last_iter += 1
                t["iteration"] = last_iter

            all_traces.append(t)

        if not all_traces:
            return pd.DataFrame()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            df = pd.concat(all_traces, ignore_index=True)
        sort_cols = ["theta_idx"]
        if "unit" in df.columns:
            sort_cols.append("unit")
        sort_cols.append("iteration")
        return df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
