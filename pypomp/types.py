"""
Type aliases for user function arguments.

Uses Annotated so the library can match arguments by tag regardless of order.
Users see the underlying type (e.g. dict[str, float]); the tag is used internally.
"""

from typing import Annotated, Dict
import jax

# Annotated[base_type, tag] — tag matches the type name; used for mapping
# Maps to 'X_'
StateDict = Annotated[Dict[str, float], "StateDict"]

# Maps to 'theta_'
ParamDict = Annotated[Dict[str, float], "ParamDict"]

# Maps to 'covars'
CovarDict = Annotated[Dict[str, float], "CovarDict"]

# Maps to 't'
TimeFloat = Annotated[float, "TimeFloat"]

# Maps to 'dt'
StepSizeFloat = Annotated[float, "StepSizeFloat"]

# Maps to 'key'
RNGKey = Annotated[jax.Array, "RNGKey"]

# Maps to 'Y_' (for DMeas)
ObservationDict = Annotated[Dict[str, float], "ObservationDict"]

# Maps to 't0' (for RInit)
InitialTimeFloat = Annotated[float, "InitialTimeFloat"]
