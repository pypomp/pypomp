"""Drop JAX persistent-cache entries not read or written since a given time.

CI restores the compilation cache, adds to it and saves it again, so without
pruning it keeps every entry ever compiled. With a max size set (see
tests/conftest.py), JAX scans the whole cache directory under a shared file
lock on every write, so each cache miss costs time proportional to the cache
size and stalls the other xdist workers. Keeping only the entries the current
run touched holds the cache at one suite's working set.

Usage: python tools/prune_jax_cache.py CACHE_DIR SINCE_NS
"""

import sys
from pathlib import Path

# File suffixes used by jax._src.lru_cache.LRUCache.
CACHE_SUFFIX = "-cache"
ATIME_SUFFIX = "-atime"


def last_used_ns(atime_path: Path) -> int:
    try:
        return int.from_bytes(atime_path.read_bytes(), "little")
    except FileNotFoundError:
        return 0


def main() -> None:
    cache_dir = Path(sys.argv[1])
    since_ns = int(sys.argv[2])
    if not cache_dir.is_dir():
        print(f"{cache_dir} does not exist; nothing to prune")
        return

    kept = removed = kept_bytes = removed_bytes = 0
    for cache_path in cache_dir.glob(f"*{CACHE_SUFFIX}"):
        key = cache_path.name.removesuffix(CACHE_SUFFIX)
        atime_path = cache_dir / f"{key}{ATIME_SUFFIX}"
        size = cache_path.stat().st_size
        if last_used_ns(atime_path) >= since_ns:
            kept += 1
            kept_bytes += size
        else:
            cache_path.unlink()
            atime_path.unlink(missing_ok=True)
            removed += 1
            removed_bytes += size

    print(
        f"kept {kept} entries ({kept_bytes / 1e6:.0f} MB), "
        f"removed {removed} ({removed_bytes / 1e6:.0f} MB)"
    )


if __name__ == "__main__":
    main()
