"""Splitting the CuTe kernels' filters over the devices of the active mesh.

XLA's partitioner can't split a custom call, so a kernel whose inputs are sharded
over several GPUs would otherwise gather every filter onto each GPU and run them
all there. Under a mesh (``jax.set_mesh``, which
:func:`pypomp.core.algorithms.helpers.run_jax_batch_sharded` enters when it spreads
the parameter sets over devices), :func:`by_filter` runs each kernel under
``shard_map`` instead, so every device launches it for its own filters.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import jax


def by_filter(ndim: int) -> Callable[[Callable], Callable]:
    """Return a decorator that splits a kernel's call over the active mesh.

    The decorated function takes arrays whose first argument has `ndim` dimensions
    for one filter; any more are batch axes, shared by every argument and output.
    Under a mesh of more than one device whose size divides the first batch axis,
    the function runs under ``shard_map``, every argument and output split along
    that axis over all of the mesh's axes. Otherwise it runs as it is.
    """

    def decorate(f: Callable) -> Callable:
        @functools.wraps(f)
        def split(*args: Any) -> Any:
            mesh = jax.sharding.get_abstract_mesh()
            if jax.numpy.ndim(args[0]) <= ndim or mesh.empty or mesh.size == 1:
                return f(*args)
            if jax.numpy.shape(args[0])[0] % mesh.size:
                return f(*args)
            spec = jax.sharding.PartitionSpec(tuple(mesh.axis_names))
            split_f = jax.shard_map(f, in_specs=spec, out_specs=spec, check_vma=False)
            return split_f(*args)

        return split

    return decorate
