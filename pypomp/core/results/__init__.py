from .result import Result
from .pomp import (
    build_pfilter_result,
    build_mif_result,
    build_train_result,
    build_pmcmc_result,
    build_abc_result,
)
from .panel import (
    build_panel_pfilter_result,
    build_panel_mif_result,
    build_panel_train_result,
    build_panel_dpop_train_result,
)
from .history import ResultsHistory

# ``BaseResult`` is retained as an alias of the unified ``Result`` for any
# external code that imported the old base type.
BaseResult = Result

__all__ = [
    "Result",
    "BaseResult",
    "build_pfilter_result",
    "build_mif_result",
    "build_train_result",
    "build_pmcmc_result",
    "build_abc_result",
    "build_panel_pfilter_result",
    "build_panel_mif_result",
    "build_panel_train_result",
    "build_panel_dpop_train_result",
    "ResultsHistory",
]
