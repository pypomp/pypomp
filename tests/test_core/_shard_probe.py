"""Report how many replicates `run_jax_batch_sharded` actually computes.

Run as a script, one process per measurement::

    python _shard_probe.py 1 3 4

The JAX device count is frozen when JAX is first imported, and the test suite
itself runs on a single device, so a separate process is the only way to put
`run_jax_batch_sharded` in front of several devices.  Everything about the
sharding decision that can be checked without devices is covered by unit tests
of :func:`~pypomp.core.algorithms.helpers.plan_sharding` instead; this only
confirms that the plan is what actually reaches the sharded function.

Results are written to stdout as a JSON object::

    {"devices": 4, "sizes": {"1": {"computed": [1], "out": [0.0]}, ...}}

`computed` holds the axis lengths the sharded function was handed, which is the
replicate count plus whatever padding was added.
"""

import json
import os
import sys
from typing import Any

DEVICES = 4


def _probe(size: int) -> dict[str, Any]:
    """Shard `size` replicates and report what the sharded function received."""
    import jax.numpy as jnp

    from pypomp.core.algorithms.helpers import run_jax_batch_sharded

    computed: list[int] = []

    def double(x):
        # len(x) is what actually gets computed: `size` plus any padding.
        computed.append(x.shape[0])
        return x * 2.0

    x = jnp.arange(size, dtype=float).reshape(size, 1)
    out = run_jax_batch_sharded(double, {0: 0}, 0, x)
    return {"computed": computed, "out": out.ravel().tolist()}


def main(argv: list[str]) -> int:
    sizes = [int(arg) for arg in argv[1:]]

    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + f" --xla_force_host_platform_device_count={DEVICES}"
    )

    import jax

    json.dump(
        {
            "devices": len(jax.devices()),
            "sizes": {str(size): _probe(size) for size in sizes},
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
