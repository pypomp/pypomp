from .history import ResultsHistory
from .panel import (
    build_panel_dpop_train_result,
    build_panel_mif_result,
    build_panel_pfilter_result,
    build_panel_train_result,
)
from .pomp import (
    build_abc_result,
    build_mif_result,
    build_pfilter_result,
    build_pmcmc_result,
    build_train_result,
)
from .result import Result

# ``BaseResult`` is retained as an alias of the unified ``Result`` for any
# external code that imported the old base type.
BaseResult = Result

__all__ = [
    "BaseResult",
    "Result",
    "ResultsHistory",
    "build_abc_result",
    "build_mif_result",
    "build_panel_dpop_train_result",
    "build_panel_mif_result",
    "build_panel_pfilter_result",
    "build_panel_train_result",
    "build_pfilter_result",
    "build_pmcmc_result",
    "build_train_result",
]
