"""
This file contains extra tests for the random number generators in pypomp. These tests have a subjective component, so they should only be run manually, not as part of the test suite.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import matplotlib.pyplot as plt
import pypomp as pp
from pypomp.binominvf import rbinom
from pypomp.gammainvf import rgamma


def poissoninvf_performance():
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


def binominvf_performance():
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


def gammainvf_performance():
    # Prepare parameters
    n = 100_000
    key = jax.random.PRNGKey(44)
    alpha = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0], dtype=jnp.float32)
    # alpha = jnp.array([100.0], dtype=jnp.float32)
    alpha_samples = jnp.repeat(alpha, n // len(alpha))
    key1, key2 = jax.random.split(key)

    # Warmup to trigger JITs
    _ = rgamma(key1, alpha_samples).block_until_ready()
    _ = jax.random.gamma(key2, alpha_samples).block_until_ready()

    # JAX's .block_until_ready() ensures we measure actual compute time
    key1, key2 = jax.random.split(key)
    t0 = time.time()
    x_pp = rgamma(key1, alpha_samples).block_until_ready()
    t1 = time.time()
    x_jax = jax.random.gamma(key2, alpha_samples).block_until_ready()
    t2 = time.time()

    print(f"rgamma: {t1 - t0:.4f} seconds for {n} samples")
    print(f"jax.random.gamma: {t2 - t1:.4f} seconds for {n} samples")
    pass


def compare_rpoisson_and_jax_poisson(
    seed=42, lam_vals=[0.0001, 0.01, 0.1, 1.0, 4.0, 10.0, 70.0, 100.0, 500.0]
):
    """
    Compare distributions of pypomp.rpoisson (inverse CDF, continuous-valued Poisson)
    and jax.random.poisson (discrete Poisson), for various rates, using histograms and qq-plots.
    """
    key = jax.random.PRNGKey(seed)
    lam = jnp.array(lam_vals)
    n_samples = 10000

    fig, axes = plt.subplots(2, len(lam_vals), figsize=(5 * len(lam_vals), 8))
    if len(lam_vals) == 1:
        axes = axes.reshape(2, 1)
    hist_axes = axes[0, :]
    qq_axes = axes[1, :]

    for i, lam_val in enumerate(lam_vals):
        # Draw samples for the i-th lambda
        lam_arr = jnp.full((n_samples,), lam_val)
        # Use a new PRNG split for each
        key_rpoisson, key = jax.random.split(key)
        rpoisson_samples = pp.rpoisson(key_rpoisson, lam_arr)
        key_jax, key = jax.random.split(key)
        jax_poisson_samples = jax.random.poisson(key_jax, lam_arr)

        # Histogram plot
        ax_hist = hist_axes[i]
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
        ax_hist.hist(
            jax_poisson_samples,
            bins=bin_edges,
            alpha=0.5,
            label="jax.random.poisson",
            density=True,
            color="C1",
            edgecolor="k",
        )
        ax_hist.hist(
            rpoisson_samples,
            bins=bin_edges,
            alpha=0.5,
            label="pp.rpoisson",
            density=True,
            color="C0",
            edgecolor="k",
        )
        ax_hist.set_title(r"$\lambda$ = {:.2f}".format(lam_val))
        ax_hist.set_xlabel("Sample value")
        if i == 0:
            ax_hist.set_ylabel("Density")
        if i == len(lam_vals) - 1:
            ax_hist.legend()

        # QQ-plot
        ax_qq = qq_axes[i]
        # Sort both samples
        sorted_jax = np.sort(np.array(jax_poisson_samples))
        sorted_rpoisson = np.sort(np.array(rpoisson_samples))
        # Plot quantiles
        ax_qq.scatter(sorted_jax, sorted_rpoisson, alpha=0.5, s=10, color="C0")
        # Add y=x reference line
        min_val = min(float(sorted_jax[0]), float(sorted_rpoisson[0]))
        max_val = max(float(sorted_jax[-1]), float(sorted_rpoisson[-1]))
        ax_qq.plot(
            [min_val, max_val], [min_val, max_val], "r--", alpha=0.7, label="y=x"
        )
        ax_qq.set_xlabel("jax.random.poisson quantiles")
        ax_qq.set_ylabel("pp.rpoisson quantiles")
        if i == len(lam_vals) - 1:
            ax_qq.legend()
    plt.tight_layout()
    plt.show()


def compare_rbinom_and_jax_binom(
    seed=42,
    n_trials_list=[3, 20, 100, 2000],
    prob_vals=[0.02 / 365.25, 0.01, 0.1, 0.3, 0.5, 0.8, 0.95],
):
    """
    Compare distributions of pypomp.rbinom (inverse CDF, Binomial) and
    jax.random.binomial for various probabilities and numbers of trials,
    using histograms and qq-plots in a grid.

    This test compares the outputs of a custom binomial sampler and JAX's built-in binomial sampler,
    over the cross product of n_trials_list and prob_vals.
    """
    key = jax.random.PRNGKey(seed)
    probs = jnp.array(prob_vals, dtype=jnp.float32)
    n_trials_arr = jnp.array(n_trials_list, dtype=jnp.int32)
    n_samples = 10000

    n_rows = len(n_trials_list)
    n_cols = len(prob_vals)

    # Store samples for both plots
    all_samples = {}

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
            all_samples[(row, col)] = (rbinom_samples, jax_binom_samples, n, p_val)

    # Create separate figure for histograms
    fig_hist, hist_axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for row, n in enumerate(n_trials_list):
        for col, p_val in enumerate(prob_vals):
            rbinom_samples, jax_binom_samples, n, p_val = all_samples[(row, col)]

            # Histogram plot
            ax_hist = hist_axes[row, col]
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
            ax_hist.hist(
                jax_binom_samples,
                bins=bin_edges,
                alpha=0.5,
                label="jax.random.binomial",
                density=True,
                color="C1",
                edgecolor="k",
            )
            ax_hist.hist(
                rbinom_samples,
                bins=bin_edges,
                alpha=0.5,
                label="pp.rbinom",
                density=True,
                color="C0",
                edgecolor="k",
            )
            ax_hist.set_title(r"$n$ = {}, $p$ = {:.2f}".format(n, p_val))
            ax_hist.set_xlabel("Sample value")
            if col == 0:
                ax_hist.set_ylabel("Density")
            if row == len(n_trials_list) - 1 and col == len(prob_vals) - 1:
                ax_hist.legend()

    fig_hist.tight_layout()

    # Create separate figure for qq-plots
    fig_qq, qq_axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for row, n in enumerate(n_trials_list):
        for col, p_val in enumerate(prob_vals):
            rbinom_samples, jax_binom_samples, n, p_val = all_samples[(row, col)]

            # QQ-plot
            ax_qq = qq_axes[row, col]
            # Sort both samples
            sorted_jax = np.sort(np.array(jax_binom_samples))
            sorted_rbinom = np.sort(np.array(rbinom_samples))
            # Plot quantiles
            ax_qq.scatter(sorted_jax, sorted_rbinom, alpha=0.5, s=10, color="C0")
            # Add y=x reference line
            min_val = min(float(sorted_jax[0]), float(sorted_rbinom[0]))
            max_val = max(float(sorted_jax[-1]), float(sorted_rbinom[-1]))
            ax_qq.plot(
                [min_val, max_val], [min_val, max_val], "r--", alpha=0.7, label="y=x"
            )
            ax_qq.set_xlabel("jax.random.binomial quantiles")
            ax_qq.set_ylabel("pp.rbinom quantiles")
            ax_qq.set_title(r"$n$ = {}, $p$ = {:.2f}".format(n, p_val))
            if row == len(n_trials_list) - 1 and col == len(prob_vals) - 1:
                ax_qq.legend()

    fig_qq.tight_layout()
    # Show both figures at the same time
    plt.show()


def compare_rgamma_and_jax_gamma(
    seed=42, alpha_vals=[0.5, 1.0, 1.01, 1.1, 1.5, 2.0, 5.0, 10.0, 50.0, 100.0]
):
    """
    Compare distributions of pypomp.rgamma (inverse CDF, Gamma) and
    jax.random.gamma for various shape parameters, using histograms and qq-plots.

    This test compares the outputs of a custom gamma sampler and JAX's built-in gamma sampler.
    """
    key = jax.random.PRNGKey(seed)
    n_samples = 10000

    fig, axes = plt.subplots(2, len(alpha_vals), figsize=(5 * len(alpha_vals), 8))
    if len(alpha_vals) == 1:
        axes = axes.reshape(2, 1)
    hist_axes = axes[0, :]
    qq_axes = axes[1, :]

    for i, alpha_val in enumerate(alpha_vals):
        # Draw samples for the i-th alpha
        alpha_arr = jnp.full((n_samples,), alpha_val, dtype=jnp.float32)
        # Use a new PRNG split for each
        key_rgamma, key = jax.random.split(key)
        rgamma_samples = rgamma(key_rgamma, alpha_arr)
        key_jax, key = jax.random.split(key)
        jax_gamma_samples = jax.random.gamma(key_jax, alpha_arr)

        # Histogram plot
        ax_hist = hist_axes[i]
        # Compute shared bin edges that span both sample sets (from min to max of both)
        min_bin = min(
            float(jnp.min(jax_gamma_samples)),
            float(jnp.min(rgamma_samples)),
        )
        max_bin = max(
            float(jnp.max(jax_gamma_samples)),
            float(jnp.max(rgamma_samples)),
        )
        # Use continuous bins for gamma (not integer steps)
        n_bins = 50
        bin_edges = jnp.linspace(min_bin, max_bin, n_bins + 1)
        ax_hist.hist(
            jax_gamma_samples,
            bins=bin_edges,
            alpha=0.5,
            label="jax.random.gamma",
            density=True,
            color="C1",
            edgecolor="k",
        )
        ax_hist.hist(
            rgamma_samples,
            bins=bin_edges,
            alpha=0.5,
            label="pp.rgamma",
            density=True,
            color="C0",
            edgecolor="k",
        )
        ax_hist.set_title(r"$\alpha$ = {:.2f}".format(alpha_val))
        ax_hist.set_xlabel("Sample value")
        if i == 0:
            ax_hist.set_ylabel("Density")
        if i == len(alpha_vals) - 1:
            ax_hist.legend()

        # QQ-plot
        ax_qq = qq_axes[i]
        # Sort both samples
        sorted_jax = np.sort(np.array(jax_gamma_samples))
        sorted_rgamma = np.sort(np.array(rgamma_samples))
        # Plot quantiles
        ax_qq.scatter(sorted_jax, sorted_rgamma, alpha=0.5, s=10, color="C0")
        # Add y=x reference line
        min_val = min(float(sorted_jax[0]), float(sorted_rgamma[0]))
        max_val = max(float(sorted_jax[-1]), float(sorted_rgamma[-1]))
        ax_qq.plot(
            [min_val, max_val], [min_val, max_val], "r--", alpha=0.7, label="y=x"
        )
        ax_qq.set_xlabel("jax.random.gamma quantiles")
        ax_qq.set_ylabel("pp.rgamma quantiles")
        if i == len(alpha_vals) - 1:
            ax_qq.legend()
    plt.tight_layout()
    plt.show()
