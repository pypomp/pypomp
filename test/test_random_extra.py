"""
This file contains extra tests for the random number generators in pypomp. These tests have a subjective component, so they should only be run manually, not as part of the test suite.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import matplotlib.pyplot as plt
import pypomp.random as ppr
import warnings
from scipy import stats
from jax.scipy.stats import binom as jax_binom


def poissoninvf_performance():
    # Prepare parameters
    n = 1_000_000
    key = jax.random.key(42)
    lam = jnp.array([0.01, 0.2, 1.0, 8.0, 10.0, 12.0, 50.0, 100.0], dtype=jnp.float32)
    lam_samples = jnp.repeat(lam, n // len(lam))
    reps = 20

    # Warmup to trigger JITs
    key1, key2 = jax.random.split(key)
    _ = ppr.fast_approx_rpoisson(key1, lam_samples).block_until_ready()
    _ = jax.random.poisson(key2, lam_samples).block_until_ready()

    # Run fast_approx_rpoisson reps times
    t_pp_total = 0.0
    for i in range(reps):
        key1, _ = jax.random.split(key1)
        t0 = time.time()
        _ = ppr.fast_approx_rpoisson(key1, lam_samples).block_until_ready()
        t1 = time.time()
        t_pp_total += t1 - t0
    avg_t_pp = t_pp_total / reps

    # Run jax.random.poisson reps times
    t_jax_total = 0.0
    for i in range(reps):
        key2, _ = jax.random.split(key2)
        t0 = time.time()
        _ = jax.random.poisson(key2, lam_samples).block_until_ready()
        t1 = time.time()
        t_jax_total += t1 - t0
    avg_t_jax = t_jax_total / reps

    print(
        f"pp.fast_approx_rpoisson: {avg_t_pp:.4f} seconds (avg over {reps} runs, {n} samples each)"
    )
    print(
        f"jax.random.poisson: {avg_t_jax:.4f} seconds (avg over {reps} runs, {n} samples each)"
    )
    print(f"ratio: {avg_t_jax / avg_t_pp:.4f}")
    pass


def binominvf_performance():
    # Prepare parameters
    n = 1_000_000
    key = jax.random.key(43)
    trials = jnp.array([1, 5, 20, 50, 100, 200], dtype=jnp.float32)
    p = jnp.array([0.01, 0.2, 0.5, 0.7, 0.9, 0.99], dtype=jnp.float32)
    trial_grid, p_grid = jnp.meshgrid(trials, p, indexing="ij")
    trial_flat = trial_grid.reshape(-1)
    p_flat = p_grid.reshape(-1)
    n_repeat = n // len(trial_flat)
    n_samples = n_repeat * len(trial_flat)  # actual total count (may be just under n)
    trial_samples = jnp.tile(trial_flat, n_repeat)
    p_samples = jnp.tile(p_flat, n_repeat)
    reps = 20

    # Warmup to trigger JITs
    key1, key2 = jax.random.split(key)
    _ = ppr.fast_approx_rbinom(key1, trial_samples, p_samples).block_until_ready()
    _ = jax.random.binomial(key2, trial_samples, p_samples).block_until_ready()

    # Run fast_approx_rbinom reps times
    t_pp_total = 0.0
    for i in range(reps):
        key1, _ = jax.random.split(key1)
        t0 = time.time()
        _ = ppr.fast_approx_rbinom(key1, trial_samples, p_samples).block_until_ready()
        t1 = time.time()
        t_pp_total += t1 - t0
    avg_t_pp = t_pp_total / reps

    # Run jax.random.binomial reps times
    t_jax_total = 0.0
    for i in range(reps):
        key2, _ = jax.random.split(key2)
        t0 = time.time()
        _ = jax.random.binomial(key2, trial_samples, p_samples).block_until_ready()
        t1 = time.time()
        t_jax_total += t1 - t0
    avg_t_jax = t_jax_total / reps

    print(
        f"fast_approx_rbinom: {avg_t_pp:.4f} seconds (avg over {reps} runs, {n_samples} samples each)"
    )
    print(
        f"jax.random.binomial: {avg_t_jax:.4f} seconds (avg over {reps} runs, {n_samples} samples each)"
    )
    print(f"ratio: {avg_t_jax / avg_t_pp:.4f}")
    pass


def gammainvf_performance():
    # Prepare parameters
    n = 1_000_000
    key = jax.random.key(44)
    alpha = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0], dtype=jnp.float32)
    alpha_samples = jnp.repeat(alpha, n // len(alpha))
    reps = 20

    # Warmup to trigger JITs
    key1, key2 = jax.random.split(key)
    t0 = time.time()
    _ = ppr.rgamma(key1, alpha_samples).block_until_ready()
    t1 = time.time()
    print(f"Warmup ppr.rgamma: {t1 - t0:.4f} seconds")

    t2 = time.time()
    _ = jax.random.gamma(key2, alpha_samples).block_until_ready()
    t3 = time.time()
    print(f"Warmup jax.random.gamma: {t3 - t2:.4f} seconds")

    # Run rgamma reps times
    t_pp_total = 0.0
    for i in range(reps):
        key1, _ = jax.random.split(key1)
        t0 = time.time()
        _ = ppr.rgamma(key1, alpha_samples).block_until_ready()
        t1 = time.time()
        t_pp_total += t1 - t0
    avg_t_pp = t_pp_total / reps

    # Run jax.random.gamma reps times
    t_jax_total = 0.0
    for i in range(reps):
        key2, _ = jax.random.split(key2)
        t0 = time.time()
        _ = jax.random.gamma(key2, alpha_samples).block_until_ready()
        t1 = time.time()
        t_jax_total += t1 - t0
    avg_t_jax = t_jax_total / reps

    print(f"rgamma: {avg_t_pp:.4f} seconds (avg over {reps} runs, {n} samples each)")
    print(
        f"jax.random.gamma: {avg_t_jax:.4f} seconds (avg over {reps} runs, {n} samples each)"
    )
    print(f"ratio: {avg_t_jax / avg_t_pp:.4f}")
    pass


def compare_rpoisson_and_jax_poisson(
    seed=42,
    lam_vals=[0.0001, 0.1, 1.0, 4.0, 4.01, 8.0, 15, 19.9, 20.1, 25, 30, 100.0, 500.0],
):
    """
    Compare distributions of pypomp.fast_approx_rpoisson (inverse CDF, continuous-valued Poisson)
    and jax.random.poisson (discrete Poisson), for various rates, using histograms and qq-plots.
    """
    key = jax.random.key(seed)
    n_samples = 100000

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
        rpoisson_samples = ppr.fast_approx_rpoisson(key_rpoisson, lam_arr)
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
            label="pp.fast_approx_rpoisson",
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
        if i == 1:
            ax_qq.set_xlabel("jax.random.poisson quantiles")
        if i == 0:
            ax_qq.set_ylabel("pp.fast_approx_rpoisson quantiles")
        if i == len(lam_vals) - 1:
            ax_qq.legend()
    plt.tight_layout()
    plt.show()


def compare_rbinom_and_jax_binom(
    seed=42,
    n_trials_list=[3, 20, 100, 2000],
    prob_vals=[0.02 / 365.25, 0.01, 0.1, 0.3, 0.5, 0.8, 0.95, 0.99],
    jitter_scale=0.01,  # % of the data range
):
    """
    Compare distributions of pypomp.fast_approx_rbinom (inverse CDF, Binomial) and
    jax.random.binomial for various probabilities and numbers of trials,
    using histograms and qq-plots in a grid.

    This test compares the outputs of a custom binomial sampler and JAX's built-in binomial sampler,
    over the cross product of n_trials_list and prob_vals.
    """
    key = jax.random.key(seed)
    n_samples = 100000

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
            rbinom_samples = ppr.fast_approx_rbinom(
                key_rbinom, n_arr, p_arr, exact_max=5
            )
            key_jax, key = jax.random.split(key)
            jax_binom_samples = jax.random.binomial(key_jax, n=n_arr, p=p_arr)
            all_samples[(row, col)] = (rbinom_samples, jax_binom_samples, n, p_val)

    # Use smaller figure heights to reduce vertical space
    fig_hist, hist_axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 2 * n_rows),
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
                label="pp.fast_approx_rbinom",
                density=True,
                color="C0",
                edgecolor="k",
            )
            ax_hist.set_title(r"$n$ = {}, $p$ = {:.2f}".format(n, p_val), fontsize=10)
            ax_hist.set_xlabel("Sample value", fontsize=8)
            if col == 0:
                ax_hist.set_ylabel("Density")
            if row == len(n_trials_list) - 1 and col == len(prob_vals) - 1:
                ax_hist.legend()

    fig_hist.tight_layout()

    # Create separate figure for qq-plots
    fig_qq, qq_axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 2 * n_rows),
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
            # Add jitter to reduce overplotting
            x_range = (
                float(sorted_jax[-1] - sorted_jax[0]) if len(sorted_jax) > 1 else 1.0
            )
            y_range = (
                float(sorted_rbinom[-1] - sorted_rbinom[0])
                if len(sorted_rbinom) > 1
                else 1.0
            )
            np.random.seed(42)  # For reproducibility
            x_jitter = np.random.normal(0, x_range * jitter_scale, size=len(sorted_jax))
            y_jitter = np.random.normal(
                0, y_range * jitter_scale, size=len(sorted_rbinom)
            )
            # Plot quantiles with jitter
            ax_qq.scatter(
                sorted_jax + x_jitter,
                sorted_rbinom + y_jitter,
                alpha=0.2,
                s=10,
                color="C0",
            )
            # Add y=x reference line
            min_val = min(float(sorted_jax[0]), float(sorted_rbinom[0]))
            max_val = max(float(sorted_jax[-1]), float(sorted_rbinom[-1]))
            ax_qq.plot(
                [min_val, max_val], [min_val, max_val], "r--", alpha=0.7, label="y=x"
            )
            if row == len(n_trials_list) - 1 and col == 1:
                ax_qq.set_xlabel("jax.random.binomial quantiles")
            if col == 0:
                ax_qq.set_ylabel("pp.fast_approx_rbinom quantiles")
            ax_qq.set_title(r"$n$ = {}, $p$ = {:.2f}".format(n, p_val))
            if row == len(n_trials_list) - 1 and col == len(prob_vals) - 1:
                ax_qq.legend()

    fig_qq.tight_layout()
    # Show both figures at the same time
    plt.show()


def compare_rgamma_and_jax_gamma(
    seed=42, alpha_vals=[0.01, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 50.0, 100.0]
):
    """
    Compare distributions of pypomp.rgamma (inverse CDF, Gamma) and
    jax.random.gamma for various shape parameters, using histograms and qq-plots.

    This test compares the outputs of a custom gamma sampler and JAX's built-in gamma sampler.
    """
    key = jax.random.key(seed)
    n_samples = 100000

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
        rgamma_samples = ppr.rgamma(key_rgamma, alpha_arr)
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


def rpoisson_goodness_of_fit():
    """Goodness of fit test for fast_approx_rpoisson using chi-square test."""
    lam = [0.0001, 0.1, 1.0, 4.0, 4.01, 8.0, 15, 19.9, 20.1, 25, 30, 100.0, 500.0]
    n_samples = 10000
    significance_level = 0.1

    for lam_val in lam:
        key = jax.random.key(int(lam_val * 1000) % 2**31)
        lam_arr = jnp.full((n_samples,), lam_val, dtype=jnp.float32)
        samples = np.array(ppr.fast_approx_rpoisson(key, lam_arr))

        # Convert to integers for chi-square test (round to nearest integer)
        samples_int = np.round(samples).astype(int)
        samples_int = np.maximum(samples_int, 0)  # Ensure non-negative

        # Get observed frequencies - use a range that covers all samples
        max_observed = int(np.max(samples_int))
        # Extend range to include tail of distribution
        max_val = max(max_observed + 1, int(lam_val * 3) + 10)
        bin_edges = np.arange(-0.5, max_val + 0.5)
        observed, _ = np.histogram(samples_int, bins=bin_edges)

        # Calculate expected frequencies from Poisson distribution for all k values
        k_values = np.arange(len(observed))
        expected = stats.poisson.pmf(k_values, lam_val) * n_samples

        # Normalize expected to sum to n_samples (account for tail beyond max_val)
        expected_sum = np.sum(expected)
        if expected_sum > 0:
            expected = expected * (n_samples / expected_sum)

        # Combine bins with expected frequency < 5 (chi-square requirement)
        min_expected = 5
        combined_observed = []
        combined_expected = []
        current_obs = 0
        current_exp = 0

        for i in range(len(observed)):
            current_obs += observed[i]
            current_exp += expected[i]
            if current_exp >= min_expected or i == len(observed) - 1:
                if current_exp > 0:  # Only add if expected > 0
                    combined_observed.append(current_obs)
                    combined_expected.append(current_exp)
                current_obs = 0
                current_exp = 0

        # Normalize combined arrays to ensure sums match exactly
        if len(combined_observed) > 1:
            obs_sum = sum(combined_observed)
            exp_sum = sum(combined_expected)
            if exp_sum > 0 and obs_sum > 0:
                # Scale expected to match observed sum
                combined_expected = [x * (obs_sum / exp_sum) for x in combined_expected]

                chi2_stat, p_value = stats.chisquare(
                    combined_observed, f_exp=combined_expected, ddof=0
                )
                # We use a lenient significance level since we're testing approximations
                if p_value <= significance_level:
                    warnings.warn(
                        f"fast_approx_rpoisson failed chi-square test for lambda={lam_val}: "
                        f"chi2={chi2_stat:.4f}, p={p_value:.4f}",
                        UserWarning,
                    )


def rbinom_goodness_of_fit():
    """Goodness of fit test for fast_approx_rbinom using chi-square test."""
    n = [3, 20, 100, 2000]
    p = [0.02 / 365.25, 0.01, 0.1, 0.3, 0.5, 0.8, 0.95, 0.99]
    n_samples = 10000
    significance_level = 0.1

    for n_val in n:
        for p_val in p:
            # Skip extreme cases that might have numerical issues
            # if p_val < 1e-6 or p_val > 1 - 1e-6:
            #     continue
            # if n_val * p_val < 1 or n_val * (1 - p_val) < 1:
            #     continue

            key = jax.random.key((int(n_val * 100) + int(p_val * 10000)) % 2**31)
            n_arr = jnp.full((n_samples,), n_val, dtype=jnp.float32)
            p_arr = jnp.full((n_samples,), p_val, dtype=jnp.float32)
            samples = np.array(ppr.fast_approx_rbinom(key, n_arr, p_arr))

            # Convert to integers
            samples_int = np.round(samples).astype(int)
            samples_int = np.clip(samples_int, 0, n_val)

            # Get observed frequencies
            observed, _ = np.histogram(samples_int, bins=np.arange(-0.5, n_val + 1.5))

            # Calculate expected frequencies from Binomial distribution
            k_values = np.arange(len(observed))
            expected = stats.binom.pmf(k_values, n_val, p_val) * n_samples

            # Normalize expected to sum to n_samples
            expected_sum = np.sum(expected)
            if expected_sum > 0:
                expected = expected * (n_samples / expected_sum)

            # Combine bins with expected frequency < 5
            min_expected = 5
            combined_observed = []
            combined_expected = []
            current_obs = 0
            current_exp = 0

            for i in range(len(observed)):
                current_obs += observed[i]
                current_exp += expected[i]
                if current_exp >= min_expected or i == len(observed) - 1:
                    if current_exp > 0:  # Only add if expected > 0
                        combined_observed.append(current_obs)
                        combined_expected.append(current_exp)
                    current_obs = 0
                    current_exp = 0

            # Normalize combined arrays to ensure sums match exactly
            if len(combined_observed) > 1:
                obs_sum = sum(combined_observed)
                exp_sum = sum(combined_expected)
                if exp_sum > 0 and obs_sum > 0:
                    # Scale expected to match observed sum
                    combined_expected = [
                        x * (obs_sum / exp_sum) for x in combined_expected
                    ]

                    chi2_stat, p_value = stats.chisquare(
                        combined_observed, f_exp=combined_expected, ddof=0
                    )
                    # We use a lenient significance level since we're testing approximations
                    if p_value <= significance_level:
                        warnings.warn(
                            f"fast_approx_rbinom failed chi-square test for n={n_val}, p={p_val}: "
                            f"chi2={chi2_stat:.4f}, p={p_value:.4f}",
                            UserWarning,
                        )


def rgamma_goodness_of_fit():
    """Goodness of fit test for rgamma using Kolmogorov-Smirnov test."""
    alpha = [0.01, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 50.0, 100.0]
    n_samples = 10000
    significance_level = 0.1

    for alpha_val in alpha:
        key = jax.random.key(int(alpha_val * 1000) % 2**31)
        alpha_arr = jnp.full((n_samples,), alpha_val, dtype=jnp.float32)
        samples = np.array(ppr.rgamma(key, alpha_arr))

        # Perform Kolmogorov-Smirnov test
        # Gamma distribution with shape=alpha, scale=1 (rate=1)
        ks_stat, p_value = stats.kstest(
            samples, lambda x: stats.gamma.cdf(x, a=alpha_val, scale=1.0)
        )

        # We use a lenient significance level since we're testing approximations
        if p_value <= significance_level:
            warnings.warn(
                f"rgamma failed KS test for alpha={alpha_val}: "
                f"KS={ks_stat:.4f}, p={p_value:.4f}",
                UserWarning,
            )
