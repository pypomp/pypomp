from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Sequence, Callable, Any, List

import numpy as np
import jax
import jax.numpy as jnp

# MCAP result container
@dataclass
class MCAPResult:
    level: float
    mle: float
    ci: Tuple[Optional[float], Optional[float]]
    delta: float
    se_stat: float
    se_mc: float
    se_total: float
    fit: Dict[str, np.ndarray]
    quadratic_max: float
    quadratic_coef: Dict[str, float]
    vcov: np.ndarray 
