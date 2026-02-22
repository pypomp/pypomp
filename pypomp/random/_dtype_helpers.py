"""Helper functions for dtype handling in JAX random number generators."""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
from jax._src import dtypes


def _get_available_dtype(requested_dtype):
    """Return the dtype that JAX actually uses (handles truncation when jax_enable_x64=False)."""
    if requested_dtype is None:
        return None
    test_val = 1.0 if dtypes.issubdtype(requested_dtype, np.floating) else 1
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*dtype.*not available.*")
        test_arr = jnp.array(test_val, dtype=requested_dtype)
    return test_arr.dtype


def check_and_canonicalize_user_dtype(dtype):
    """Canonicalize dtype and return what JAX actually uses."""
    if dtype is None:
        return None
    # Use JAX's canonicalize if available, otherwise pass through
    if hasattr(dtypes, "canonicalize_dtype"):
        try:
            dtype = dtypes.canonicalize_dtype(dtype)  # type: ignore[attr-defined]
        except Exception:
            pass
    # Get the actual dtype JAX will use
    return _get_available_dtype(dtype)
