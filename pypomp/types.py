"""
Type aliases for user-defined model component function arguments.

These aliases use :data:`typing.Annotated` so the library can identify
argument roles by tag (regardless of argument order) when wrapping
user-supplied ``rinit``, ``rproc``, ``dmeas``, and ``rmeas`` functions.
Users see the underlying base type (e.g. ``dict[str, float]``); the tag
string is consumed internally during function introspection.
"""

from typing import Annotated, TypeAlias

import jax
import numpy as np

Numeric: TypeAlias = int | float | np.number | jax.Array
"""A scalar numeric value: Python int/float, NumPy scalar, or JAX array."""

# Annotated[base_type, tag] — tag matches the type name; used for mapping
# Maps to 'X_'
StateDict = Annotated[dict[str, float | jax.Array], "StateDict"]
"""Latent state dictionary mapping state variable names to current values.

Used as the ``X_`` argument in ``rproc`` and ``rmeas`` user functions.
"""

# Maps to 'theta_'
ParamDict = Annotated[dict[str, float | jax.Array], "ParamDict"]
"""Parameter dictionary mapping parameter names to current values.

Used as the ``theta_`` argument in all user-defined model component functions.
"""

# Maps to 'covars'
CovarDict = Annotated[dict[str, float | jax.Array], "CovarDict"]
"""Covariate dictionary mapping covariate names to values at the current time.

Used as the ``covars`` argument in model component functions when covariates
are provided.
"""

# Maps to 't'
TimeFloat = Annotated[float, "TimeFloat"]
"""Current observation time, passed as the ``t`` argument."""

# Maps to 'dt'
StepSizeFloat = Annotated[float, "StepSizeFloat"]
"""Euler approximation step size, passed as the ``dt`` argument in ``rproc``."""

# Maps to 'key'
RNGKey = Annotated[jax.Array, "RNGKey"]
"""JAX PRNG key for random number generation, passed as the ``key`` argument."""

# Maps to 'Y_' (for DMeas)
ObservationDict = Annotated[dict[str, float | jax.Array], "ObservationDict"]
"""Observation dictionary mapping measurement names to observed values.

Used as the ``Y_`` argument in ``dmeas`` user functions.
"""

# Maps to 't0' (for RInit)
InitialTimeFloat = Annotated[float, "InitialTimeFloat"]
"""Initial time ``t0``, passed as the ``t0`` argument in ``rinit``."""
