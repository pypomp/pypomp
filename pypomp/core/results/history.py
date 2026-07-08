from typing import overload, Sequence, Any
import pandas as pd
import warnings
from .base import BaseResult


class ResultsHistory:
    """
    Stores and manages a history of results from multiple pypomp runs.
    """

    def __init__(self, entries: Sequence[BaseResult] | None = None):
        self._entries: list[BaseResult] = list(entries) if entries else []

    def append(self, entry: BaseResult):
        """Add a new result to the history."""
        self._entries.append(entry)

    def add(self, entry: BaseResult):
        """Alias for append() for backward compatibility."""
        self.append(entry)

    def time(self) -> pd.DataFrame:
        """Return a DataFrame summarizing execution times."""
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
        """
        Merge multiple histories into one by merging entries at each index.
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
    def __getitem__(self, index: int) -> BaseResult: ...

    @overload
    def __getitem__(self, index: slice) -> "ResultsHistory": ...

    def __getitem__(self, index: int | slice) -> "BaseResult | ResultsHistory":
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

    def last(self) -> BaseResult:
        """Get last entry."""
        if not self._entries:
            raise ValueError("History is empty")
        return self._entries[-1]

    def results(self, index: int = -1, ignore_nan: bool = False) -> pd.DataFrame:
        """Get results DataFrame for entry at index."""
        if not self._entries:
            return pd.DataFrame()
        result = self._entries[index]
        return result.to_dataframe(ignore_nan=ignore_nan)

    def CLL(self, index: int = -1, average: bool = False) -> pd.DataFrame:
        """Get conditional log-likelihoods for entry at index."""
        if not self._entries:
            return pd.DataFrame()
        result = self._entries[index]
        return result.CLL(average=average)

    def ESS(self, index: int = -1, average: bool = False) -> pd.DataFrame:
        """Get Effective Sample Size for entry at index."""
        if not self._entries:
            return pd.DataFrame()
        result = self._entries[index]
        return result.ESS(average=average)

    def traces(self) -> pd.DataFrame:
        """
        Return a DataFrame with the full trace of log-likelihoods and parameters
        from the entire result history.
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
