import jax
import jax.numpy as jnp
import pypomp as pp
from pypomp.binominvf import rbinom
import time
import matplotlib.pyplot as plt


def test_poissoninvf():
    key = jax.random.PRNGKey(0)
    lam = jnp.array([1.0, 2.0, 3.0])
    x = pp.rpoisson(key, lam)
    assert x.shape == (3,)
    assert x.dtype == jnp.float32
    assert x.min() >= 0


def test_poissoninvf_performance():
    # Prepare parameters
    n = 100_000
    key = jax.random.PRNGKey(42)
    lam = jnp.array([0.01, 0.2, 1.0, 10.0, 50.0, 100.0], dtype=jnp.float32)
    # lam = jnp.array([100.0], dtype=jnp.float32)
    lam_samples = jnp.repeat(lam, n // len(lam))
    key1, key2 = jax.random.split(key)

    # Warmup to trigger JITs
    _ = pp.rpoisson(key1, lam_samples).block_until_ready()
    _ = jax.random.poisson(key2, lam_samples).block_until_ready()

    # JAX's .block_until_ready() ensures we measure actual compute time
    key1, key2 = jax.random.split(key)
    t0 = time.time()
    x_pp = pp.rpoisson(key1, lam_samples).block_until_ready()
    t1 = time.time()
    x_jax = jax.random.poisson(key2, lam_samples).block_until_ready()
    t2 = time.time()

    print(f"pp.rpoisson: {t1 - t0:.4f} seconds for {n} samples")
    print(f"jax.random.poisson: {t2 - t1:.4f} seconds for {n} samples")
    pass


def test_binominvf_performance():
    # Prepare parameters
    n = 100_000
    key = jax.random.PRNGKey(43)
    trials = jnp.array([1, 5, 20, 50, 100, 200], dtype=jnp.float32)
    # trials = jnp.array([100], dtype=jnp.float32)
    p = jnp.array([0.01, 0.2, 0.5, 0.7, 0.9, 0.99], dtype=jnp.float32)
    # Create all (trial, p) combinations and tile to reach n samples
    trial_grid, p_grid = jnp.meshgrid(trials, p, indexing="ij")
    trial_flat = trial_grid.reshape(-1)
    p_flat = p_grid.reshape(-1)
    n_repeat = n // len(trial_flat)
    n_samples = n_repeat * len(trial_flat)  # actual total count (may be just under n)
    trial_samples = jnp.tile(trial_flat, n_repeat)
    p_samples = jnp.tile(p_flat, n_repeat)
    key1, key2 = jax.random.split(key)

    # Warmup to trigger JITs
    _ = rbinom(key1, trial_samples, p_samples).block_until_ready()
    _ = jax.random.binomial(key2, trial_samples, p_samples).block_until_ready()

    # JAX's .block_until_ready() ensures we measure actual compute time
    key1, key2 = jax.random.split(key)
    t0 = time.time()
    x_pp = rbinom(key1, trial_samples, p_samples).block_until_ready()
    t1 = time.time()
    x_jax = jax.random.binomial(key2, trial_samples, p_samples).block_until_ready()
    t2 = time.time()

    print(f"rbinom: {t1 - t0:.4f} seconds for {n_samples} samples")
    print(f"jax.random.binomial: {t2 - t1:.4f} seconds for {n_samples} samples")
    pass


# Convenience function for examining the distributions of the Poisson random variables.
# Remove test_ when actually testing.
def compare_rpoisson_and_jax_poisson(
    seed=42, lam_vals=[0.01, 0.1, 1.0, 4.0, 10.0, 70.0, 100.0, 500.0]
):
    """
    Compare distributions of pypomp.rpoisson (inverse CDF, continuous-valued Poisson)
    and jax.random.poisson (discrete Poisson), for various rates, using histograms.
    """
    # Activate .venv if running as script, per custom rules - this must be done outside of the script in bash.
    key = jax.random.PRNGKey(seed)
    lam = jnp.array(lam_vals)
    n_samples = 10000

    fig, axes = plt.subplots(
        1, len(lam_vals), figsize=(5 * len(lam_vals), 4), sharey=True
    )
    if len(lam_vals) == 1:
        axes = [axes]

    for i, lam_val in enumerate(lam_vals):
        # Draw samples for the i-th lambda
        lam_arr = jnp.full((n_samples,), lam_val)
        # Use a new PRNG split for each
        key_rpoisson, key = jax.random.split(key)
        rpoisson_samples = pp.rpoisson(key_rpoisson, lam_arr)
        key_jax, key = jax.random.split(key)
        jax_poisson_samples = jax.random.poisson(key_jax, lam_arr)
        # For rpoisson, values may be fractional, but for comparison, let's plot histogram
        ax = axes[i]
        # Compute shared bin edges that span both sample sets (from min to max of both)
        min_bin = min(
            float(jnp.min(jax_poisson_samples)),
            float(jnp.min(rpoisson_samples)),
        )
        max_bin = max(
            float(jnp.max(jax_poisson_samples)),
            float(jnp.max(rpoisson_samples)),
        )
        # Make bin edges as integer steps
        bin_edges = jnp.arange(jnp.floor(min_bin), jnp.ceil(max_bin) + 1)
        ax.hist(
            jax_poisson_samples,
            bins=bin_edges,
            alpha=0.5,
            label="jax.random.poisson",
            density=True,
            color="C1",
            edgecolor="k",
        )
        ax.hist(
            rpoisson_samples,
            bins=bin_edges,
            alpha=0.5,
            label="pp.rpoisson",
            density=True,
            color="C0",
            edgecolor="k",
        )
        ax.set_title(r"$\lambda$ = {:.2f}".format(lam_val))
        ax.set_xlabel("Sample value")
        if i == 0:
            ax.set_ylabel("Density")
        ax.legend()
    plt.tight_layout()
    plt.show()


# Convenience function for examining the distributions of the binomial random variables.
# Remove test_ when actually testing.
def compare_rbinom_and_jax_binom(
    seed=42,
    n_trials_list=[3, 20, 100, 2000],
    prob_vals=[0.02 / 365.25, 0.01, 0.1, 0.3, 0.5, 0.8, 0.95],
):
    """
    Compare distributions of pypomp.rbinom (inverse CDF, Binomial) and
    jax.random.binomial for various probabilities and numbers of trials,
    using histograms in a grid.

    This test compares the outputs of a custom binomial sampler and JAX's built-in binomial sampler,
    over the cross product of n_trials_list and prob_vals.
    """
    # Activate .venv if running as script, per custom rules - this must be done outside of the script in bash.
    key = jax.random.PRNGKey(seed)
    probs = jnp.array(prob_vals, dtype=jnp.float32)
    n_trials_arr = jnp.array(n_trials_list, dtype=jnp.int32)
    n_samples = 10000

    n_rows = len(n_trials_list)
    n_cols = len(prob_vals)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey="row", squeeze=False
    )

    for row, n in enumerate(n_trials_list):
        for col, p_val in enumerate(prob_vals):
            # Draw samples for the given n and p
            p_arr = jnp.full((n_samples,), p_val, dtype=jnp.float32)
            n_arr = jnp.full((n_samples,), n, dtype=jnp.int32)
            # Use a new PRNG split for each
            key_rbinom, key = jax.random.split(key)
            rbinom_samples = rbinom(key_rbinom, n_arr, p_arr)
            key_jax, key = jax.random.split(key)
            jax_binom_samples = jax.random.binomial(key_jax, n=n_arr, p=p_arr)
            # For comparison, plot histogram
            ax = axes[row, col]
            # Compute shared bin edges that span both sample sets (from min to max of both)
            min_bin = min(
                float(jnp.min(jax_binom_samples)),
                float(jnp.min(rbinom_samples)),
            )
            max_bin = max(
                float(jnp.max(jax_binom_samples)),
                float(jnp.max(rbinom_samples)),
            )
            # Make bin edges as integer steps
            bin_edges = jnp.arange(jnp.floor(min_bin), jnp.ceil(max_bin) + 1)
            ax.hist(
                jax_binom_samples,
                bins=bin_edges,
                alpha=0.5,
                label="jax.random.binomial",
                density=True,
                color="C1",
                edgecolor="k",
            )
            ax.hist(
                rbinom_samples,
                bins=bin_edges,
                alpha=0.5,
                label="pp.rbinom",
                density=True,
                color="C0",
                edgecolor="k",
            )
            ax.set_title(r"$n$ = {}, $p$ = {:.2f}".format(n, p_val))
            ax.set_xlabel("Sample value")
            if col == 0:
                ax.set_ylabel("Density")
            ax.legend()
    plt.tight_layout()
    plt.show()
