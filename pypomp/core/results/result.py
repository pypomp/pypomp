"""The single :class:`Result` container used by every inference method.

All state lives on this one dataclass:

* ``config`` — a mapping of the run's hyperparameters (``J``, ``M``, ``thresh``,
  ``rw_sd``, ...). These are used only for display (:mod:`.render`) and as
  merge-equality guards; algorithms never read them back out of a result.
* ``payload`` — a single :class:`xarray.Dataset` holding every numeric array the
  method produced (log-likelihoods, CLL, ESS, traces, ...). Variables may have
  differing dimensions; they all share ``theta_idx``.

Rendering (``to_dataframe``/``traces``/``CLL``/``ESS``/``print_summary``) is
delegated to free functions in :mod:`.render`. Legacy per-method result classes
survive as empty tag-subclasses (see :mod:`.pomp` / :mod:`.panel`) purely so that
``isinstance`` checks and ``X.merge(...)`` calls keep working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np
import pandas as pd
import xarray as xr

# Payload variables surfaced under a legacy attribute name.
_PAYLOAD_ALIASES = {"CLL_da": "CLL", "ESS_da": "ESS", "traces_da": "traces"}

# Payload variables that are optional; accessing them returns ``None`` when the
# variable is absent (matches the historical ``result.CLL_da is None`` contract).
_OPTIONAL_PAYLOAD_VARS = {"CLL", "ESS", "filter_mean", "prediction_mean"}

# Panel "trace" results hold three arrays whose ``variable``/``unit`` dims would
# collide inside a single Dataset (``shared_traces`` and ``unit_traces`` both use
# ``variable`` with different labels; ``logLiks`` uses a ``unit`` dim one longer
# than ``unit_traces``). They are stored under collision-free internal dim names
# and restored to their public names on access, so the payload stays a single
# Dataset while accessors keep their historical dimensions.
_DIM_UNALIAS = {"unit_variable": "variable", "ll_unit": "unit"}


def _unalias(da: xr.DataArray) -> xr.DataArray:
    """Restore public dimension names on a payload variable."""
    rename = {
        d: _DIM_UNALIAS[d] for d in da.dims if isinstance(d, str) and d in _DIM_UNALIAS
    }
    return da.rename(rename) if rename else da


@dataclass(eq=False)
class Result:
    """Container for the output of a single inference run.

    Parameters
    ----------
    method : str
        Name of the method that produced this result (``"pfilter"``, ``"mif"``,
        ``"train"``, ``"pmcmc"``, ``"abc"``, ``"dpop_train"``).
    kind : str
        Payload shape: ``"table"`` (pfilter-style log-likelihood table) or
        ``"trace"`` (iteration-by-iteration parameter/likelihood traces).
    panel : bool
        Whether this result came from a :class:`~pypomp.panel.panel.PanelPomp`.
    execution_time : float or None
        Total wall-clock execution time in seconds.
    key : jax.Array
        The JAX random key used for the run.
    theta : object, optional
        The parameter object (``PompParameters`` / ``PanelParameters`` / list).
    config : dict
        Run hyperparameters, used for summaries and merge-equality guards.
    payload : xarray.Dataset
        All numeric arrays produced by the run, keyed by variable name.
    timestamp : pandas.Timestamp
        When the result was created.
    """

    method: str
    kind: str
    panel: bool
    execution_time: float | None
    key: jax.Array
    theta: Any = None
    config: dict[str, Any] = field(default_factory=dict)
    payload: xr.Dataset = field(default_factory=xr.Dataset)
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)

    # ------------------------------------------------------------------
    # Accessors: surface payload variables and config keys as attributes.
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        # Only called when normal attribute lookup fails. Guard against
        # recursion / partially-initialised state (e.g. during unpickling) by
        # reading from ``__dict__`` directly.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        payload = self.__dict__.get("payload")
        config = self.__dict__.get("config")
        var = _PAYLOAD_ALIASES.get(name, name)
        if payload is not None and var in payload.data_vars:
            return _unalias(payload[var])
        if config is not None and name in config:
            return config[name]
        if var in _OPTIONAL_PAYLOAD_VARS:
            return None
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    # ------------------------------------------------------------------
    # Computed diagnostics (MCMC-family results).
    # ------------------------------------------------------------------
    @property
    def n_chains(self) -> int:
        """Number of chains / parameter sets (size of the ``theta_idx`` dim)."""
        payload = self.__dict__.get("payload")
        if payload is None:
            return 0
        return int(payload.sizes.get("theta_idx", 0))

    @property
    def acceptance_rate(self) -> np.ndarray:
        """Per-chain acceptance rate for MCMC-family results."""
        denom = self.config.get("Nmcmc") or self.config.get("Nabc")
        accepts = self.payload["accepts"].values if "accepts" in self.payload else None
        if accepts is None:
            return np.zeros(0, dtype=float)
        if not denom:
            return np.zeros_like(np.asarray(accepts), dtype=float)
        return np.asarray(accepts, dtype=float) / float(denom)

    # ------------------------------------------------------------------
    # Equality and pickling.
    # ------------------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        assert isinstance(other, Result)
        if (self.method, self.kind, self.panel) != (
            other.method,
            other.kind,
            other.panel,
        ):
            return False
        if not _config_equal(self.config, other.config):
            return False
        if self.theta != other.theta:
            return False
        if not self.payload.equals(other.payload):
            return False
        return bool(
            jax.numpy.array_equal(
                jax.random.key_data(self.key), jax.random.key_data(other.key)
            )
        )

    def __getstate__(self) -> dict[str, Any]:
        """Custom pickling: store the JAX key as raw bits."""
        state = vars(self).copy()
        if self.key is not None:
            state["_key_data"] = jax.random.key_data(self.key)
        state.pop("key", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Custom unpickling: reconstruct the JAX key from raw bits."""
        vars(self).update(state)
        if "_key_data" in state:
            self.key = jax.random.wrap_key_data(state["_key_data"])
        vars(self).pop("_key_data", None)

    # ------------------------------------------------------------------
    # Merge.
    # ------------------------------------------------------------------
    @classmethod
    def merge(cls, *results: "Result") -> "Result":
        """Merge results of the same type by concatenating along ``theta_idx``."""
        if not results:
            raise ValueError(f"At least one {cls.__name__} object must be provided.")
        for r in results:
            if type(r) is not type(results[0]):
                raise TypeError(
                    f"All merged objects must be of type {type(results[0]).__name__}."
                )
        first = results[0]
        for r in results:
            if (r.method, r.kind, r.panel) != (first.method, first.kind, first.panel):
                raise ValueError("All merged results must share method/kind/panel.")
            key = _first_config_mismatch(r.config, first.config)
            if key is not None:
                raise ValueError(f"All merged results must have the same {key}.")

        merged_theta = _merge_theta([r.theta for r in results])
        payloads = [r.payload for r in results if r.payload.sizes.get("theta_idx", 0)]
        if payloads:
            merged_payload = xr.concat(payloads, dim="theta_idx").assign_coords(
                theta_idx=np.arange(sum(p.sizes["theta_idx"] for p in payloads))
            )
        else:
            merged_payload = first.payload
        times = [r.execution_time for r in results if r.execution_time is not None]

        return cls(
            method=first.method,
            kind=first.kind,
            panel=first.panel,
            execution_time=max(times) if times else None,
            key=first.key,
            theta=merged_theta,
            config=dict(first.config),
            payload=merged_payload,
        )

    # ------------------------------------------------------------------
    # Rendering delegations (implemented in .render).
    # ------------------------------------------------------------------
    def to_dataframe(self, ignore_nan: bool = False) -> pd.DataFrame:
        """Convert this result to a tidy :class:`pandas.DataFrame`."""
        from . import render

        return render.to_dataframe(self, ignore_nan=ignore_nan)

    def traces(self) -> pd.DataFrame:
        """Return the parameter/likelihood trace as a :class:`pandas.DataFrame`."""
        from . import render

        return render.traces(self)

    def CLL(self, average: bool = False) -> pd.DataFrame:
        """Return conditional log-likelihoods as a :class:`pandas.DataFrame`."""
        from . import render

        return render.CLL(self, average=average)

    def ESS(self, average: bool = False) -> pd.DataFrame:
        """Return effective sample sizes as a :class:`pandas.DataFrame`."""
        from . import render

        return render.ESS(self, average=average)

    def print_summary(self, n: int = 5) -> None:
        """Print a human-readable summary of this result."""
        from . import render

        render.print_summary(self, n=n)


def _values_equal(va: Any, vb: Any) -> bool:
    if isinstance(va, (np.ndarray, jax.Array)) or isinstance(
        vb, (np.ndarray, jax.Array)
    ):
        return bool(np.array_equal(np.asarray(va), np.asarray(vb)))
    return bool(va == vb)


def _first_config_mismatch(a: dict[str, Any], b: dict[str, Any]) -> str | None:
    """Return the name of the first config key that differs, or ``None``."""
    if a.keys() != b.keys():
        differing = set(a).symmetric_difference(b)
        return sorted(differing)[0]
    for key in a:
        if not _values_equal(a[key], b[key]):
            return key
    return None


def _config_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Compare two config dicts, tolerating array-valued entries."""
    return _first_config_mismatch(a, b) is None


def _merge_theta(thetas: list[Any]) -> Any:
    """Merge parameter objects the way the historical ``_merge_results`` did."""
    present = [t for t in thetas if t is not None]
    if not present:
        return None
    first = present[0]
    if isinstance(first, list):
        merged: list[Any] = []
        for t in thetas:
            merged.extend(t or [])
        return merged
    if hasattr(type(first), "merge"):
        return type(first).merge(*present)
    return first
