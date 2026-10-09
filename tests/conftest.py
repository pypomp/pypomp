import ctypes
import gc
import os
import resource
import sys
import time

import jax
import pytest

# Configure JAX persistent compilation cache to avoid duplicate compilations
# across parallel xdist workers and subsequent pytest runs.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache_dir = os.path.join(project_root, ".pytest_cache", "jax_cache")

# Test writeability of the cache directory and only configure cache if successful
try:
    if os.environ.get("DISABLE_JAX_CACHE") != "1":
        os.makedirs(cache_dir, exist_ok=True)
        test_file = os.path.join(cache_dir, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)

        # Update JAX configuration
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        # Cache all compilations, including fast ones (default is 1.0 second min compile time)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        # Bound the cache size to enable LRU eviction. This also makes JAX's
        # LRUCache take a real file lock around every read/write; without it
        # (max_size == -1, the default), concurrent xdist workers can write
        # the same cache entry at once with no locking and corrupt it,
        # producing spurious "Error reading persistent compilation cache
        # entry ... ZstdError" warnings.
        jax.config.update("jax_compilation_cache_max_size", 5 * 1024**3)
except Exception:
    # If cache directory is not writeable (e.g. in sandbox environment), JAX caching is disabled.
    pass


def _load_malloc_trim():
    if not sys.platform.startswith("linux"):
        return None
    try:
        return ctypes.CDLL("libc.so.6").malloc_trim
    except (OSError, AttributeError):
        return None


_malloc_trim = _load_malloc_trim()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """Release each test module's compiled executables on Linux.

    JAX's jit caches keep every executable a module compiles (~100 MB each for
    the larger models), and once released glibc keeps the pages until told to
    trim. Both steps are needed to bound xdist worker RSS on the CI runners.
    """
    yield
    if nextitem is not None and nextitem.module is item.module:
        return
    _log_module_stats(item)
    if _malloc_trim is None:
        return
    jax.clear_caches()
    gc.collect()
    _malloc_trim(0)


# Per-module timing log, enabled by PYPOMP_TEST_DIAG_LOG=<path>. Lines are
# appended as each module finishes so they survive a job killed mid-run.
# cpu much less than wall points to a starved runner (steal or swap); many
# misses point to a cold or mismatched persistent compilation cache.
_diag_log = os.environ.get("PYPOMP_TEST_DIAG_LOG")
_cache_events = {"hits": 0, "misses": 0}
_module_start: dict[str, float] = {}


def _count_cache_event(event: str, **kwargs: object) -> None:
    if event == "/jax/compilation_cache/cache_hits":
        _cache_events["hits"] += 1
    elif event == "/jax/compilation_cache/cache_misses":
        _cache_events["misses"] += 1


if _diag_log:
    jax.monitoring.register_event_listener(_count_cache_event)


def _reset_module_stats() -> None:
    _module_start.update(
        wall=time.monotonic(), cpu=time.process_time(), **_cache_events
    )


def pytest_collection_finish(session):
    _reset_module_stats()


def _log_module_stats(item) -> None:
    if not _diag_log:
        return
    # ru_maxrss is in KB on Linux and bytes on macOS.
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    maxrss_mb = maxrss / 1024**2 if sys.platform == "darwin" else maxrss / 1024
    line = (
        f"{time.strftime('%H:%M:%S')} "
        f"{os.environ.get('PYTEST_XDIST_WORKER', 'main')} "
        f"{item.module.__name__} "
        f"wall={time.monotonic() - _module_start['wall']:.0f}s "
        f"cpu={time.process_time() - _module_start['cpu']:.0f}s "
        f"hits={_cache_events['hits'] - _module_start['hits']:.0f} "
        f"misses={_cache_events['misses'] - _module_start['misses']:.0f} "
        f"maxrss={maxrss_mb:.0f}MB\n"
    )
    with open(_diag_log, "a") as f:
        f.write(line)
    _reset_module_stats()
