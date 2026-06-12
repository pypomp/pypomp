"""
This module defines the parameter classes for Pomp and PanelPomp models.
It handles input validation, standardization, and conversion to JAX arrays.
"""

from .base import ParameterSet
from .pomp import PompParameters
from .panel import PanelParameters

__all__ = ["ParameterSet", "PompParameters", "PanelParameters"]
