"""Collection gate for the CPU parallel-scaling tests.

These tests measure wall-clock time, so they only mean anything when they have
the machine's cores to themselves.  A normal `pytest` run uses xdist with one
worker per core (see pytest.ini), which would make any timing here pure noise.
Rather than emit skips on every run, the tests are not collected at all unless
`PYPOMP_CPU_SCALING=1` is set:

    PYPOMP_CPU_SCALING=1 .venv/bin/pytest tests/test_cpu_parallel -p no:xdist

or, equivalently, `make test-cpu-scaling`.
"""

import os

if os.environ.get("PYPOMP_CPU_SCALING") != "1":
    collect_ignore_glob = ["test_*.py"]
