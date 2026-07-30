import ctypes
import ctypes.util
import os
import sys

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


def _get_malloc_trim():
    """Return glibc's ``malloc_trim``, or None where it does not apply.

    The suite frees its memory correctly -- profiling a sequential run on macOS
    shows per-process RSS oscillating between ~0.9 and ~1.7 GB as module-scoped
    model fixtures are torn down. glibc, though, does not hand those freed
    chunks back to the kernel on its own, so on Linux the same run ratchets
    upward instead: on a GitHub runner each of the four xdist workers passed
    4 GB and the 16 GB VM died mid-suite (`mem_avail=21M`, 3 GB of swap in use)
    with every test still passing.

    ``malloc_trim(0)`` releases the free chunks in every arena. Returns None
    off Linux, where macOS's allocator already gives the memory back, and on
    Linux libcs that do not export the call (musl).
    """
    if not sys.platform.startswith("linux"):
        return None
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        trim = libc.malloc_trim
    except (OSError, AttributeError):
        return None
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return trim


_malloc_trim = _get_malloc_trim()


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    # `trylast` puts this after pytest's own implementation, so the test's
    # fixtures have already been finalised and whatever they released is
    # trimmable. Sub-millisecond -- it only walks glibc's free lists.
    del item, nextitem
    if _malloc_trim is not None:
        _malloc_trim(0)
