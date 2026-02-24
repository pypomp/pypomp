"""
Tests for sharding functionality in pfilter and mif methods.

This isn't yet able to be run as part of the larger test suite because JAX initializes
XLA once per Python process. This should be ran separately for now.
"""

import os

# Force JAX to use exactly 2 CPU devices for testing sharding
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=2"
)

import jax
import jax.numpy as jnp
import pytest

import pypomp as pp

# Global parameters
J_DEFAULT = 3
KEY_DEFAULT = jax.random.key(111)
REPS_DEFAULT = 2
M_DEFAULT = 2
A_DEFAULT = 0.5

RW_SD_DEFAULT = pp.RWSigma(
    sigmas={
        "A1": 0.02,
        "A2": 0.02,
        "A3": 0.02,
        "A4": 0.02,
        "C1": 0.02,
        "C2": 0.02,
        "C3": 0.02,
        "C4": 0.02,
        "Q1": 0.02,
        "Q2": 0.02,
        "Q3": 0.02,
        "Q4": 0.02,
        "R1": 0.02,
        "R2": 0.02,
        "R3": 0.02,
        "R4": 0.0,
    },
    init_names=[],
)


@pytest.fixture(scope="function")
def pomp():
    """Fixture that returns a Pomp object."""
    return pp.LG()


def _test_pfilter_sharding(pomp):
    """Test that pfilter works correctly across available devices."""
    num_devices = max(1, len(jax.devices()))
    theta_orig = pomp.theta.to_list()[0]

    # JAX's NamedSharding requires the sharded dimension to be perfectly divisible
    # by the number of devices.
    test_cases = [
        # (number of thetas, reps)
        (num_devices, 1),
        (num_devices * 2, 1),
        (1, num_devices),
        (1, num_devices * 2),
        (num_devices, num_devices),
    ]

    for n_thetas, n_reps in test_cases:
        pomp.results_history.clear()

        theta_list = [
            {k: v * (1 + 0.1 * i) for k, v in theta_orig.items()}
            for i in range(n_thetas)
        ]

        pomp.pfilter(J=J_DEFAULT, key=KEY_DEFAULT, theta=theta_list, reps=n_reps)

        result = pomp.results_history[-1]
        assert result.method == "pfilter"
        assert result.logLiks.shape == (n_thetas, n_reps)
        assert jnp.all(jnp.isfinite(result.logLiks.data))


def _test_mif_sharding(pomp):
    """Test that mif works correctly across available devices."""
    num_devices = max(1, len(jax.devices()))
    theta_orig = pomp.theta.to_list()[0]

    test_cases = [
        num_devices,
        num_devices * 2,
    ]

    for n_thetas in test_cases:
        pomp.results_history.clear()

        theta_list = [
            {k: v * (1 + 0.1 * i) for k, v in theta_orig.items()}
            for i in range(n_thetas)
        ]

        pomp.mif(
            J=J_DEFAULT,
            M=M_DEFAULT,
            rw_sd=RW_SD_DEFAULT,
            a=A_DEFAULT,
            key=KEY_DEFAULT,
            theta=theta_list,
        )

        result = pomp.results_history[-1]
        assert result.method == "mif"
        traces = result.traces_da
        assert traces.shape[0] == n_thetas
        assert traces.shape[1] == M_DEFAULT + 1
        assert traces.shape[2] == len(theta_orig) + 1
