"""Skip modules whose optional dependencies are missing, e.g. in doctest runs."""

import importlib.util

collect_ignore = (
    []
    if importlib.util.find_spec("cutlass")
    else ["pypomp/cute/kernel.py", "pypomp/cute/translate.py"]
)
