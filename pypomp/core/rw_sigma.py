from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import cloudpickle
from typing import Callable, Any, Sequence


def default_cooling(nt: Any, m: Any, ntimes: Any) -> float:
    """Default flat cooling schedule that does not reduce the random walk standard deviations."""
    return 1.0


class RWSigma:
    """Random walk standard deviation configuration for IF2 parameter perturbation.

    Stores per-parameter random walk standard deviations and a cooling
    schedule used by the iterated filtering algorithms (IF2, MPIF, PIF).  The
    cooling schedule reduces the perturbation magnitude over iterations
    so parameters converge to their maximum likelihood estimates.

    Parameters
    ----------
    sigmas : dict of str to float
        Mapping from parameter names to non-negative standard deviations
        for the perturbation random walk.
    init_names : sequence of str, optional
        Subset of parameter names to treat as "initial" parameters (e.g.
        initial state parameters).  These are perturbed with the
        initial-value sigmas instead of the ordinary sigmas at iteration 0.
        Defaults to an empty sequence.
    cooling_fn : callable or None, optional
        Custom cooling function of signature ``(nt, m, ntimes) -> float``.
        If ``None``, defaults to a geometric schedule with ``a=0.5``.

    Examples
    --------
    >>> rw = RWSigma({"beta": 0.02, "gamma": 0.01}).geometric_cooling(0.5)
    >>> rw  # doctest: +SKIP
    RWSigma(sigmas={'beta': 0.02, 'gamma': 0.01}, init_names=(), cooling='geometric')

    See Also
    --------
    :meth:`Pomp.mif <pypomp.Pomp.mif>` : Uses RWSigma to run IF2.

    :meth:`PanelPomp.mif <pypomp.PanelPomp.mif>` : Uses RWSigma to run PIF/MPIF.
    """

    param_names: tuple[str, ...]
    """Tuple of all parameter names, defining the array ordering (a PyTree leaf order)."""
    init_names: tuple[str, ...]
    """Tuple of parameter names that are considered initial parameters."""
    sigmas_all_arr: np.ndarray
    """Array of every sigma in ``param_names`` order (a PyTree leaf)."""
    init_mask: np.ndarray
    """0/1 array flagging the initial parameters, in ``param_names`` order (a PyTree leaf)."""
    cooling_type: str
    """One of ``geometric``, ``cosine``, ``hyperbolic``, ``custom``, or ``none``."""
    a: float | None
    s: float | None
    c: float | None
    M: int | None
    _custom_fn: Callable | None
    """User-supplied cooling function, stored only when ``cooling_type == 'custom'``."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        sigmas: dict[str, float],
        init_names: Sequence[str] = (),
        cooling_fn: Callable | None = None,
    ):
        if not isinstance(init_names, (list, tuple)):
            raise ValueError("init_names must be a list or tuple")
        param_names, init_names_t, sigmas_arr, init_mask = self._validate(
            sigmas, list(init_names)
        )
        object.__setattr__(self, "param_names", param_names)
        object.__setattr__(self, "init_names", init_names_t)
        object.__setattr__(self, "sigmas_all_arr", sigmas_arr)
        object.__setattr__(self, "init_mask", init_mask)
        if cooling_fn is not None:
            self._set_cooling("custom", custom_fn=cooling_fn)
        else:
            self._set_cooling("geometric", a=0.5)

    @classmethod
    def _from_leaves(
        cls,
        param_names: tuple[str, ...],
        init_names: tuple[str, ...],
        sigmas_all_arr: Any,
        init_mask: Any,
        cooling_type: str,
        a: float | None,
        s: float | None,
        c: float | None,
        M: int | None,
        custom_fn: Callable | None,
    ) -> RWSigma:
        """Rebuild an instance directly from leaves + aux (bypasses validation).

        Used by PyTree unflattening (where leaves may be tracers) and by the
        array-transforming helpers (:meth:`_canonicalize`, :meth:`_permuted`).
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "param_names", tuple(param_names))
        object.__setattr__(obj, "init_names", tuple(init_names))
        object.__setattr__(obj, "sigmas_all_arr", sigmas_all_arr)
        object.__setattr__(obj, "init_mask", init_mask)
        object.__setattr__(obj, "cooling_type", cooling_type)
        object.__setattr__(obj, "a", a)
        object.__setattr__(obj, "s", s)
        object.__setattr__(obj, "c", c)
        object.__setattr__(obj, "M", M)
        object.__setattr__(obj, "_custom_fn", custom_fn)
        return obj

    def _set_cooling(
        self,
        cooling_type: str,
        a: float | None = None,
        s: float | None = None,
        c: float | None = None,
        M: int | None = None,
        custom_fn: Callable | None = None,
    ) -> None:
        object.__setattr__(self, "cooling_type", cooling_type)
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "s", s)
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "M", M)
        object.__setattr__(self, "_custom_fn", custom_fn)

    @staticmethod
    def _validate(
        sigmas: dict[str, float], init_names: list[str]
    ) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray]:
        if not isinstance(sigmas, dict):
            raise ValueError("sigmas must be a dictionary")
        clean: dict[str, float] = {}
        for name, value in sigmas.items():
            if not isinstance(name, str):
                raise ValueError("All keys in sigmas must be strings")
            if isinstance(value, bool):
                raise ValueError(f"Value for '{name}' must be a float, got bool")
            try:
                fval = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Value for parameter '{name}' in sigmas must be a float: "
                    f"got {type(value).__name__}"
                )
            if fval < 0:
                raise ValueError("All values in sigmas dictionary must be non-negative")
            clean[name] = fval

        if not all(isinstance(n, str) for n in init_names):
            raise ValueError("All values in init_names list must be strings")
        if len(init_names) != len(set(init_names)):
            raise ValueError("Duplicate names found in init_names")
        if not all(n in clean for n in init_names):
            raise ValueError("All init_names names must be in sigmas dictionary")

        param_names = tuple(clean.keys())
        sigmas_arr = np.asarray([clean[n] for n in param_names], dtype=float)
        init_set = set(init_names)
        init_mask = np.asarray(
            [1.0 if n in init_set else 0.0 for n in param_names], dtype=float
        )
        return param_names, tuple(init_names), sigmas_arr, init_mask

    # ------------------------------------------------------------------
    # Device-side interface
    # ------------------------------------------------------------------

    @property
    def sigmas_array(self) -> Any:
        """Sigmas for the non-initial parameters (0 elsewhere), in ``param_names`` order."""
        return self.sigmas_all_arr * (1.0 - self.init_mask)

    @property
    def sigmas_init_array(self) -> Any:
        """Sigmas for the initial parameters (0 elsewhere), in ``param_names`` order."""
        return self.sigmas_all_arr * self.init_mask

    def cooling_factor(self, nt: Any, m: Any, ntimes: Any) -> Any:
        """Evaluate the cooling schedule (device-side).

        Parameters
        ----------
        nt : int or jax.Array
            Current time-step index within an iteration.
        m : int or jax.Array
            Current IF2 iteration index.
        ntimes : int or jax.Array
            Total number of observation time steps.

        Returns
        -------
        The multiplicative cooling factor applied to the random-walk sigmas.
        """
        frac = nt / ntimes + m
        if self.cooling_type == "geometric":
            assert self.a is not None
            factor = self.a ** (1.0 / 50.0)
            return factor**frac
        elif self.cooling_type == "cosine":
            assert self.c is not None and self.M is not None
            progress = frac / self.M
            return self.c + (1.0 - self.c) * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        elif self.cooling_type == "hyperbolic":
            assert self.s is not None
            return 1.0 / (1.0 + self.s * frac)
        elif self.cooling_type == "custom":
            assert self._custom_fn is not None
            return self._custom_fn(nt, m, ntimes)
        else:
            return 1.0

    # ------------------------------------------------------------------
    # Host-side transforms (return new instances)
    # ------------------------------------------------------------------

    def _canonicalize(self, canonical_names: Sequence[str]) -> RWSigma:
        """Reorder the sigmas to match the model's canonical parameter vector.

        Parameters
        ----------
        canonical_names : sequence of str
            The model's full canonical parameter names.  Must equal this
            object's parameter names up to reordering.

        Returns
        -------
        RWSigma
            A new instance whose arrays are ordered to match ``canonical_names``.
        """
        names = tuple(canonical_names)
        if set(names) != set(self.param_names):
            raise ValueError(
                "RWSigma parameter names must match canonical_param_names up to "
                f"reordering. Got {sorted(self.param_names)}, expected {sorted(names)}."
            )
        idx = {n: i for i, n in enumerate(self.param_names)}
        order = np.asarray([idx[n] for n in names], dtype=int)
        return RWSigma._from_leaves(
            param_names=names,
            init_names=self.init_names,
            sigmas_all_arr=self.sigmas_all_arr[order],
            init_mask=self.init_mask[order],
            cooling_type=self.cooling_type,
            a=self.a,
            s=self.s,
            c=self.c,
            M=self.M,
            custom_fn=self._custom_fn,
        )

    def _permuted(self, permutation: Any) -> RWSigma:
        """Reorder only the sigma arrays by ``permutation`` (device-side, panel use).

        The parameter-name tuples are left unchanged because ``permutation`` may
        be a tracer; the resulting instance is only consumed on-device, where the
        names are not read.
        """
        return RWSigma._from_leaves(
            param_names=self.param_names,
            init_names=self.init_names,
            sigmas_all_arr=self.sigmas_all_arr[permutation],
            init_mask=self.init_mask[permutation],
            cooling_type=self.cooling_type,
            a=self.a,
            s=self.s,
            c=self.c,
            M=self.M,
            custom_fn=self._custom_fn,
        )

    def geometric_cooling(self, a: float) -> RWSigma:
        """Return a copy using geometric cooling with 50-iteration factor ``a``."""
        if not (0 <= a <= 1):
            raise ValueError("a should be between 0 and 1")
        obj = self.copy()
        obj._set_cooling("geometric", a=float(a))
        return obj

    def cosine_cooling(self, c: float, M: int) -> RWSigma:
        """Return a copy using cosine annealing cooling."""
        if not (0 <= c <= 1):
            raise ValueError("c should be between 0 and 1")
        if M <= 0:
            raise ValueError("M must be positive")
        obj = self.copy()
        obj._set_cooling("cosine", c=float(c), M=int(M))
        return obj

    def hyperbolic_cooling(self, s: float) -> RWSigma:
        """Return a copy using hyperbolic cooling."""
        if s < 0:
            raise ValueError("s must be non-negative")
        obj = self.copy()
        obj._set_cooling("hyperbolic", s=float(s))
        return obj

    def custom_cooling(self, cooling_fn: Callable) -> RWSigma:
        """Return a copy using a custom ``(nt, m, ntimes) -> float`` cooling function."""
        obj = self.copy()
        obj._set_cooling("custom", custom_fn=cooling_fn)
        return obj

    def cooled(self, factor: float) -> RWSigma:
        """Scale all standard deviations by ``factor`` and return a new instance."""
        if factor < 0:
            raise ValueError("factor must be >= 0")
        obj = self.copy()
        object.__setattr__(
            obj, "sigmas_all_arr", np.asarray(self.sigmas_all_arr) * factor
        )
        return obj

    def copy(self) -> RWSigma:
        """Return a copy of this instance."""
        return RWSigma._from_leaves(
            param_names=self.param_names,
            init_names=self.init_names,
            sigmas_all_arr=np.array(self.sigmas_all_arr),
            init_mask=np.array(self.init_mask),
            cooling_type=self.cooling_type,
            a=self.a,
            s=self.s,
            c=self.c,
            M=self.M,
            custom_fn=self._custom_fn,
        )

    # ------------------------------------------------------------------
    # Pickling: only the (optional) custom cooling function needs cloudpickle;
    # numpy arrays, tuples, and floats pickle natively.
    # ------------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        fn = state.pop("_custom_fn", None)
        if self.cooling_type == "custom" and fn is not None:
            state["_custom_fn_bytes"] = cloudpickle.dumps(fn)
        return state

    def __setstate__(self, state):
        fn_bytes = state.pop("_custom_fn_bytes", None)
        self.__dict__.update(state)
        object.__setattr__(
            self,
            "_custom_fn",
            cloudpickle.loads(fn_bytes) if fn_bytes is not None else None,
        )

    # ------------------------------------------------------------------
    # Dict-like ergonomics (read-only views)
    # ------------------------------------------------------------------

    @property
    def sigmas(self) -> dict[str, float]:
        """Read-only dict view of ``{param_name: sigma}`` (host-side only)."""
        return {
            n: float(v)
            for n, v in zip(self.param_names, np.asarray(self.sigmas_all_arr))
        }

    def __getitem__(self, param_name: str) -> float:
        if param_name not in self.param_names:
            raise KeyError(f"Parameter '{param_name}' not found in sigmas.")
        return float(self.sigmas_all_arr[self.param_names.index(param_name)])

    def __contains__(self, param_name: str) -> bool:
        return param_name in self.param_names

    def __len__(self) -> int:
        return len(self.param_names)

    def __iter__(self):
        return iter(self.param_names)

    def keys(self):
        return self.sigmas.keys()

    def values(self):
        return self.sigmas.values()

    def items(self):
        return self.sigmas.items()

    def get(self, param_name: str, default: float | None = None) -> float | None:
        if param_name in self.param_names:
            return self[param_name]
        return default

    # ------------------------------------------------------------------
    # Representation and equality
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        sigmas_str = ", ".join(f"'{k}': {v:.4g}" for k, v in self.sigmas.items())
        return (
            f"RWSigma(sigmas={{{sigmas_str}}}, init_names={self.init_names}, "
            f"cooling='{self.cooling_type}')"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False
        if self.param_names != other.param_names:
            return False
        if self.init_names != other.init_names:
            return False
        if not np.array_equal(
            np.asarray(self.sigmas_all_arr), np.asarray(other.sigmas_all_arr)
        ):
            return False
        if self.cooling_type != other.cooling_type:
            return False
        if (self.a, self.s, self.c, self.M) != (other.a, other.s, other.c, other.M):
            return False
        if self.cooling_type == "custom":
            return _callables_equal(self._custom_fn, other._custom_fn)
        return True


def _callables_equal(fn1: Callable | None, fn2: Callable | None) -> bool:
    if fn1 is fn2:
        return True
    if fn1 is None or fn2 is None:
        return False
    if not (hasattr(fn1, "__code__") and hasattr(fn2, "__code__")):
        return False
    if fn1.__code__ != fn2.__code__:
        return False
    cells1 = getattr(fn1, "__closure__", None)
    cells2 = getattr(fn2, "__closure__", None)
    if (cells1 is None) != (cells2 is None):
        return False
    if cells1 is not None and cells2 is not None:
        if len(cells1) != len(cells2):
            return False
        for c1, c2 in zip(cells1, cells2):
            if c1.cell_contents != c2.cell_contents:
                return False
    return True


# ----------------------------------------------------------------------
# PyTree registration: sigma arrays are leaves; names + cooling spec are aux.
# ----------------------------------------------------------------------

jax.tree_util.register_pytree_node(
    RWSigma,
    lambda rw: (
        (rw.sigmas_all_arr, rw.init_mask),
        (
            rw.param_names,
            rw.init_names,
            rw.cooling_type,
            rw.a,
            rw.s,
            rw.c,
            rw.M,
            rw._custom_fn,
        ),
    ),
    lambda aux, children: RWSigma._from_leaves(
        param_names=aux[0],
        init_names=aux[1],
        sigmas_all_arr=children[0],
        init_mask=children[1],
        cooling_type=aux[2],
        a=aux[3],
        s=aux[4],
        c=aux[5],
        M=aux[6],
        custom_fn=aux[7],
    ),
)
