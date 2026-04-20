"""
Type aliases for user function arguments.

Uses Annotated so the library can match arguments by tag regardless of order.
Users see the underlying type (e.g. dict[str, float]); the tag is used internally.
"""

from typing import Annotated, TypeAlias, Mapping, Sequence, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .core.parameters import PompParameters
import jax
import numpy as np

Numeric: TypeAlias = int | float | np.number | jax.Array

# Annotated[base_type, tag] — tag matches the type name; used for mapping
# Maps to 'X_'
StateDict = Annotated[dict[str, float | jax.Array], "StateDict"]

# Maps to 'theta_'
ParamDict = Annotated[dict[str, float | jax.Array], "ParamDict"]

# Maps to 'covars'
CovarDict = Annotated[dict[str, float | jax.Array], "CovarDict"]

# Maps to 't'
TimeFloat = Annotated[float, "TimeFloat"]

# Maps to 'dt'
StepSizeFloat = Annotated[float, "StepSizeFloat"]

# Maps to 'key'
RNGKey = Annotated[jax.Array, "RNGKey"]

# Maps to 'Y_' (for DMeas)
ObservationDict = Annotated[dict[str, float | jax.Array], "ObservationDict"]

# Maps to 't0' (for RInit)
InitialTimeFloat = Annotated[float, "InitialTimeFloat"]

ThetaInput: TypeAlias = Union[
    Mapping[str, Numeric],
    Sequence[Mapping[str, Numeric]],
    "PompParameters",
    None,
]
