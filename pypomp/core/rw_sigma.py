from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Any, cast, Sequence


def default_cooling(nt: Any, m: Any, ntimes: Any) -> float:
    """Default flat cooling schedule that does not reduce the random walk standard deviations."""
    return 1.0


class RWSigma:
    """Random walk standard deviation configuration for IF2 parameter perturbation.

    Stores per-parameter random walk standard deviations and a cooling
    schedule used by the Iterated Filtering 2 (IF2) algorithm.  The
    cooling schedule reduces the perturbation magnitude over iterations
    so parameters converge to their maximum likelihood estimates.

    Parameters
    ----------
    sigmas : dict of str to float
        Mapping from parameter names to non-negative standard deviations
        for the perturbation random walk.
    init_names : sequence of str, optional
        Subset of parameter names to treat as "initial" parameters (e.g.
        initial state parameters).  These are perturbed with
        ``sigmas_init`` instead of ``sigmas`` in the algorithm.  Defaults
        to an empty sequence.
    cooling_fn : callable or None, optional
        Custom cooling function of signature
        ``(nt, m, ntimes) -> float``.  If ``None``, defaults to a
        geometric schedule with ``a=0.5``.

    Examples
    --------
    >>> rw = RWSigma({"beta": 0.02, "gamma": 0.01}).geometric_cooling(0.5)
    >>> rw  # doctest: +SKIP
    RWSigma({'beta': 0.02, 'gamma': 0.01}, geometric a=0.5)

    See Also
    --------
    :meth:`Pomp.mif <pypomp.Pomp.mif>` : Uses RWSigma to run IF2.

    :meth:`PanelPomp.mif <pypomp.PanelPomp.mif>` : Uses RWSigma to run PIF/MPIF.
    """

    sigmas: dict[str, float]
    """Dictionary mapping parameter names to sigma values."""
    init_names: tuple[str, ...]
    """Tuple of parameter names that are considered initial parameters."""
    not_init_names: tuple[str, ...]
    """Tuple of parameter names that are not considered initial parameters."""
    all_names: tuple[str, ...]
    """Tuple of all parameter names."""
    cooling_fn: Callable
    """A Callable taking (nt, m, ntimes) and returning a float cooling factor."""
    a: float | None
    """The geometric cooling parameter if configured."""
    s: float | None
    """The hyperbolic cooling parameter if configured."""
    c: float | None
    """The cosine cooling minimum factor if configured."""
    M: int | None
    """The cosine cooling duration if configured."""
    _cooling_info: tuple[Any, ...]
    """Tuple storing cooling type and arguments for pickling."""

    def __init__(
        self,
        sigmas: dict[str, float],
        init_names: Sequence[str] = (),
        cooling_fn: Callable | None = None,
    ):
        if not isinstance(init_names, (list, tuple)):
            raise ValueError("init_names must be a list or tuple")
        self.sigmas, self.init_names, self.not_init_names, self.all_names = (
            self._validate_attributes(sigmas, list(init_names))
        )
        if cooling_fn is not None:
            self.cooling_fn = cooling_fn
            self.a = None
            self.s = None
            self.c = None
            self.M = None
            self._cooling_info = cast(tuple[Any, ...], ("custom", cooling_fn))
        else:
            self.a = 0.5
            self.s = None
            self.c = None
            self.M = None
            self._cooling_info = cast(tuple[Any, ...], ("geometric", 0.5))
            factor = 0.5 ** (1 / 50)

            def geometric_fn(nt, m, ntimes):
                return factor ** (nt / ntimes + m)

            self.cooling_fn = geometric_fn

    def __getstate__(self):
        state = self.__dict__.copy()
        if "cooling_fn" in state:
            del state["cooling_fn"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        info = cast(tuple[Any, ...], getattr(self, "_cooling_info", ("none",)))
        ctype = info[0]
        if ctype == "geometric":
            a = info[1]
            self.a = a
            self.s = None
            self.c = None
            self.M = None
            factor = a ** (1 / 50)

            def geometric_fn(nt, m, ntimes):
                return factor ** (nt / ntimes + m)

            self.cooling_fn = geometric_fn
        elif ctype == "cosine":
            c, M = info[1], info[2]
            self.a = None
            self.s = None
            self.c = c
            self.M = M

            def cosine_fn(nt, m, ntimes):
                return c + (1.0 - c) * 0.5 * (
                    1.0 + jnp.cos(jnp.pi * ((nt / ntimes + m) / M))
                )

            self.cooling_fn = cosine_fn
        elif ctype == "hyperbolic":
            s = info[1]
            self.a = None
            self.s = s
            self.c = None
            self.M = None

            def hyperbolic_fn(nt, m, ntimes):
                return 1.0 / (1.0 + s * (nt / ntimes + m))

            self.cooling_fn = hyperbolic_fn
        elif ctype == "custom":
            self.a = None
            self.s = None
            self.c = None
            self.M = None
            self.cooling_fn = info[1]
        else:
            self.a = None
            self.s = None
            self.c = None
            self.M = None
            self.cooling_fn = default_cooling

    def geometric_cooling(self, a: float) -> RWSigma:
        """Return a copy of this instance using geometric cooling.

        The cooling factor at iteration ``m``, time step ``nt``, with
        ``ntimes`` total steps is:

        .. math::

           f = (a^{1/50})^{\\text{nt}/\\text{ntimes} + m}

        Parameters
        ----------
        a : float
            Cumulative cooling factor after 50 algorithm iterations.
            Must be in the interval :math:`[0, 1]`.

        Returns
        -------
        RWSigma
            A new :class:`RWSigma` instance with geometric cooling
            configured.
        """
        if not (0 <= a <= 1):
            raise ValueError("a should be between 0 and 1")
        factor = a ** (1 / 50)

        def fn(nt, m, ntimes):
            return factor ** (nt / ntimes + m)

        obj = RWSigma(self.sigmas, init_names=self.init_names, cooling_fn=fn)
        obj.a = a
        obj.s = None
        obj.c = None
        obj.M = None
        obj._cooling_info = cast(tuple[Any, ...], ("geometric", a))
        return obj

    def cosine_cooling(self, c: float, M: int) -> RWSigma:
        """Return a copy of this instance using cosine annealing cooling.

        The cooling factor at progress ``p = (nt / ntimes + m) / M`` is:

        .. math::

           \\text{factor} = c + (1-c) \\cdot 0.5 \\cdot (1 + \\cos(\\pi p))

        Parameters
        ----------
        c : float
            Minimum cooling factor reached at ``p >= 1``.  Must be in the
            interval :math:`[0, 1]`.
        M : int
            Number of iterations over which the cosine schedule is
            defined.  Typically set to the total number of IF2
            iterations.  Must be positive.

        Returns
        -------
        RWSigma
            A new :class:`RWSigma` instance with cosine cooling
            configured.
        """
        if not (0 <= c <= 1):
            raise ValueError("c should be between 0 and 1")
        if M <= 0:
            raise ValueError("M must be positive")

        def fn(nt, m, ntimes):
            progress = (nt / ntimes + m) / M
            return c + (1.0 - c) * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))

        obj = RWSigma(self.sigmas, init_names=self.init_names, cooling_fn=fn)
        obj.a = None
        obj.s = None
        obj.c = c
        obj.M = M
        obj._cooling_info = cast(tuple[Any, ...], ("cosine", c, M))
        return obj

    def hyperbolic_cooling(self, s: float) -> RWSigma:
        """Return a copy of this instance using hyperbolic cooling.

        The cooling factor at iteration ``m``, time step ``nt``, with
        ``ntimes`` total steps is:

        .. math::

           \\text{factor} = \\frac{1}{1 + s \\cdot (\\text{nt}/\\text{ntimes} + m)}

        Parameters
        ----------
        s : float
            Hyperbolic decay rate.  Larger ``s`` gives faster cooling.
            Must be non-negative.

        Returns
        -------
        RWSigma
            A new :class:`RWSigma` instance with hyperbolic cooling
            configured.
        """
        if s < 0:
            raise ValueError("s must be non-negative")

        def fn(nt, m, ntimes):
            return 1.0 / (1.0 + s * (nt / ntimes + m))

        obj = RWSigma(self.sigmas, init_names=self.init_names, cooling_fn=fn)
        obj.a = None
        obj.s = s
        obj.c = None
        obj.M = None
        obj._cooling_info = cast(tuple[Any, ...], ("hyperbolic", s))
        return obj

    def custom_cooling(self, cooling_fn: Callable) -> RWSigma:
        """Return a copy of this instance using a custom cooling function.

        Parameters
        ----------
        cooling_fn : callable
            A function with signature ``(nt, m, ntimes) -> float``
            where ``nt`` is the current time step, ``m`` is the current
            iteration, and ``ntimes`` is the total number of time steps.

        Returns
        -------
        RWSigma
            A new :class:`RWSigma` instance with the custom schedule.
        """
        obj = RWSigma(self.sigmas, init_names=self.init_names, cooling_fn=cooling_fn)
        obj.a = None
        obj.s = None
        obj.c = None
        obj.M = None
        return obj

    def _validate_attributes(
        self, sigmas: dict[str, float], init_names: list[str]
    ) -> tuple[dict[str, float], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """
        Validates the attributes of the RWSigma object and returns prepared attributes.
        """
        if not isinstance(sigmas, dict):
            raise ValueError("sigmas must be a dictionary")
        for param_name, value in sigmas.items():
            if isinstance(value, (int, np.number, jax.Array)) and not isinstance(
                value, bool
            ):
                try:
                    sigmas[param_name] = float(value)
                except (TypeError, ValueError):
                    pass

            if not isinstance(sigmas[param_name], float):
                raise ValueError(
                    f"Value for parameter '{param_name}' in sigmas dictionary must be a float: "
                    f"got {type(sigmas[param_name]).__name__}"
                )
        if not isinstance(init_names, list):
            raise ValueError("init_names must be a list")
        if not all(isinstance(param_name, str) for param_name in init_names):
            raise ValueError("All values in init_names list must be strings")
        if not all(param_name in sigmas.keys() for param_name in init_names):
            raise ValueError("All init_names names must be in sigmas dictionary")
        if len(init_names) != len(set(init_names)):
            raise ValueError("Duplicate names found in init_names")
        if not all(sigmas[param_name] >= 0 for param_name in sigmas.keys()):
            raise ValueError("All values in sigmas dictionary must be non-negative")

        not_init_names = [name for name in sigmas.keys() if name not in init_names]
        if len(not_init_names) != len(set(not_init_names)):
            raise ValueError("Duplicate names found in not_init_names")

        all_names = not_init_names + init_names
        if len(all_names) != len(set(all_names)):
            raise ValueError("Duplicate names found in all_names")

        return sigmas, tuple(init_names), tuple(not_init_names), tuple(all_names)

    def _return_arrays(
        self, param_names: Sequence[str] | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """
        Returns the sigmas and sigmas_init arrays. If param_names is provided, only
        returns the arrays if the parameter names in the object match those in the
        param_names argument.

        Returns:
            sigmas_array: Array of sigmas for non-initial parameters. Shape (d,).
            Contains 0 for initial parameters.
            sigmas_init_array: Array of sigmas for initial parameters. Shape (d,).
            Contains 0 for non-initial parameters.
        """
        if param_names is None:
            param_names = self.all_names
        else:
            if not (
                all(param_name in self.all_names for param_name in param_names)
                and all(param_name in param_names for param_name in self.all_names)
            ):
                raise ValueError("All param_names must be in all_names and vice versa")

        all_sigmas_array = jnp.array(
            [self.sigmas[param_name] for param_name in param_names]
        )
        not_init_mask = jnp.array(
            [
                1 if param_name in self.not_init_names else 0
                for param_name in param_names
            ]
        )
        init_mask = jnp.array(
            [1 if param_name in self.init_names else 0 for param_name in param_names]
        )
        sigmas_array = all_sigmas_array * not_init_mask
        sigmas_init_array = all_sigmas_array * init_mask
        return sigmas_array, sigmas_init_array

    def copy(self) -> RWSigma:
        """Return a copy of the RWSigma instance."""
        obj = RWSigma(
            self.sigmas.copy(),
            init_names=self.init_names,
            cooling_fn=self.cooling_fn,
        )
        obj.a = self.a
        obj.s = self.s
        obj.c = self.c
        obj.M = self.M
        obj._cooling_info = self._cooling_info
        return obj

    def cooled(self, factor: float) -> RWSigma:
        """Scale all standard deviations by ``factor`` and return a new instance.

        Parameters
        ----------
        factor : float
            Multiplicative scaling applied to every sigma.  Must be
            non-negative.

        Returns
        -------
        RWSigma
            A new :class:`RWSigma` instance with scaled sigmas.
        """
        if factor < 0:
            raise ValueError("factor must be >= 0")
        obj = self.copy()
        for key in obj.sigmas:
            obj.sigmas[key] *= factor
        return obj

    def __getitem__(self, param_name: str) -> float:
        """
        Get the sigma value for a given parameter name using index syntax.
        """
        if param_name not in self.sigmas:
            raise KeyError(f"Parameter '{param_name}' not found in sigmas.")
        return self.sigmas[param_name]

    def __setitem__(self, param_name: str, value: float) -> None:
        """
        Set the value of a sigma for a given parameter name using the indexing syntax.

        Args:
            param_name (str): The name of the parameter whose sigma value you wish to set.
            value (float): The new sigma value.
        """
        if param_name not in self.sigmas:
            raise KeyError(f"Parameter '{param_name}' not found in sigmas.")
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise TypeError(
                "Sigma value must be a float or numeric type that can be coerced to float."
            )
        if value < 0:
            raise ValueError("Sigma value must be non-negative.")

        self.sigmas[param_name] = value

    def __contains__(self, param_name: str) -> bool:
        """Check if a parameter name is present in sigmas."""
        return param_name in self.sigmas

    def __len__(self) -> int:
        """Return the number of parameters in sigmas."""
        return len(self.sigmas)

    def __iter__(self):
        """Iterate over all parameter names."""
        return iter(self.all_names)

    def keys(self):
        """Return a view of the parameter names."""
        return self.sigmas.keys()

    def values(self):
        """Return a view of the sigma values."""
        return self.sigmas.values()

    def items(self):
        """Return a view of the parameter-sigma pairs."""
        return self.sigmas.items()

    def get(self, param_name: str, default: float | None = None) -> float | None:
        """Get the sigma value, or default if the parameter is not present."""
        return self.sigmas.get(param_name, default)

    def __str__(self) -> str:
        cooling_type = (
            self._cooling_info[0] if hasattr(self, "_cooling_info") else "none"
        )
        sigmas_str = ", ".join(f"'{k}': {v:.4g}" for k, v in self.sigmas.items())
        return f"RWSigma(sigmas={{{sigmas_str}}}, init_names={self.init_names}, cooling='{cooling_type}')"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        """
        Check equality with another :class:`~pypomp.core.rw_sigma.RWSigma` object.

        Two :class:`~pypomp.core.rw_sigma.RWSigma` instances are equal if they have the same sigmas,
        init_names, and cooling configuration.
        """
        if not isinstance(other, type(self)):
            return False
        if self.sigmas != other.sigmas:
            return False
        if self.init_names != other.init_names:
            return False
        if getattr(self, "a", None) != getattr(other, "a", None):
            return False

        info1 = cast(tuple[Any, ...], getattr(self, "_cooling_info", ("none",)))
        info2 = cast(tuple[Any, ...], getattr(other, "_cooling_info", ("none",)))
        if info1[0] != info2[0]:
            return False

        if info1[0] == "geometric":
            return info1[1] == info2[1]
        elif info1[0] == "cosine":
            return info1[1] == info2[1] and info1[2] == info2[2]
        elif info1[0] == "hyperbolic":
            return info1[1] == info2[1]
        elif info1[0] == "none":
            return True
        elif info1[0] == "custom":
            fn1, fn2 = info1[1], info2[1]
            if fn1 == fn2:
                return True
            if hasattr(fn1, "__code__") and hasattr(fn2, "__code__"):
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
            return False
        return True
