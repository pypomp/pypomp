import os
import time
from typing import Any, Callable
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import stats
import pypomp.random as ppr

# Mark all tests in this module as heavy
pytestmark = pytest.mark.heavy

# Plot directory setup
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


def check_and_skip_ci() -> bool:
    """Check if we are in a CI/GitHub Actions environment."""
    return os.getenv("GITHUB_ACTIONS") is not None


def run_performance_test(
    name: str,
    reps: int,
    fast_sampler: Callable[[], Any],
    ref_sampler: Callable[[], Any] | None = None,
) -> None:
    """Generic performance test wrapper that runs warmup, timings, and prints statistics.

    Args:
        name: Name of the distribution.
        reps: Number of timing repetitions.
        fast_sampler: Function returning a block-until-ready JAX array.
        ref_sampler: Reference function returning a block-until-ready JAX array.
    """
    # Warmup to trigger JIT compilation
    res_fast = fast_sampler()
    _ = res_fast.block_until_ready()

    if ref_sampler is not None:
        res_ref = ref_sampler()
        _ = res_ref.block_until_ready()
    else:
        res_ref = None

    # Time the fast sampler
    t_fast_total = 0.0
    for _ in range(reps):
        t0 = time.time()
        _ = fast_sampler().block_until_ready()
        t1 = time.time()
        t_fast_total += t1 - t0
    avg_t_fast = t_fast_total / reps

    # Verify return shape sanity
    assert len(res_fast) > 0

    avg_t_ref = None
    if ref_sampler is not None:
        # Time the reference sampler
        t_ref_total = 0.0
        for _ in range(reps):
            t0 = time.time()
            _ = ref_sampler().block_until_ready()
            t1 = time.time()
            t_ref_total += t1 - t0
        avg_t_ref = t_ref_total / reps

        assert res_ref is not None
        print(f"\n[{name} Performance]")
        print(
            f"pp.fast_{name.lower()}: {avg_t_fast:.4f} seconds (avg over {reps} runs, {len(res_fast)} samples each)"
        )
        print(
            f"jax.random.{name.lower()}: {avg_t_ref:.4f} seconds (avg over {reps} runs, {len(res_ref)} samples each)"
        )
        print(f"ratio (ref / fast): {avg_t_ref / avg_t_fast:.4f}")
    else:
        print(f"\n[{name} Performance]")
        print(
            f"pp.fast_{name.lower()}: {avg_t_fast:.4f} seconds (avg over {reps} runs, {len(res_fast)} samples each)"
        )


# =====================================================================
# Performance Tests
# =====================================================================


def test_poisson_performance() -> None:
    n = 1_000_000
    key = jax.random.key(1001)
    lam = jnp.array([0.01, 0.2, 1.0, 8.0, 10.0, 12.0, 50.0, 100.0], dtype=jnp.float32)
    lam_samples = jnp.repeat(lam, n // len(lam))

    key1, key2 = jax.random.split(key)
    state_key1 = [key1]
    state_key2 = [key2]

    # JIT compilable closures for JAX compatibility
    @jax.jit
    def run_fast() -> jax.Array:
        state_key1[0], subkey = jax.random.split(state_key1[0])
        return ppr.fast_poisson(subkey, lam_samples)

    @jax.jit
    def run_ref() -> jax.Array:
        state_key2[0], subkey = jax.random.split(state_key2[0])
        return jax.random.poisson(subkey, lam_samples)

    run_performance_test("Poisson", 20, run_fast, run_ref)


def test_binomial_performance() -> None:
    n = 1_000_000
    key = jax.random.key(1002)
    trials = jnp.array([1, 5, 20, 50, 100, 200], dtype=jnp.float32)
    p = jnp.array([0.01, 0.2, 0.5, 0.7, 0.9, 0.99], dtype=jnp.float32)
    trial_grid, p_grid = jnp.meshgrid(trials, p, indexing="ij")
    trial_flat = trial_grid.reshape(-1)
    p_flat = p_grid.reshape(-1)
    n_repeat = n // len(trial_flat)
    trial_samples = jnp.tile(trial_flat, n_repeat)
    p_samples = jnp.tile(p_flat, n_repeat)

    key1, key2 = jax.random.split(key)
    state_key1 = [key1]
    state_key2 = [key2]

    @jax.jit
    def run_fast() -> jax.Array:
        state_key1[0], subkey = jax.random.split(state_key1[0])
        return ppr.fast_binomial(subkey, trial_samples, p_samples)

    @jax.jit
    def run_ref() -> jax.Array:
        state_key2[0], subkey = jax.random.split(state_key2[0])
        return jax.random.binomial(subkey, trial_samples, p_samples)

    run_performance_test("Binomial", 20, run_fast, run_ref)


def test_gamma_performance() -> None:
    n = 1_000_000
    key = jax.random.key(1003)
    alpha = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0], dtype=jnp.float32)
    alpha_samples = jnp.repeat(alpha, n // len(alpha))

    key1, key2 = jax.random.split(key)
    state_key1 = [key1]
    state_key2 = [key2]

    @jax.jit
    def run_fast() -> jax.Array:
        state_key1[0], subkey = jax.random.split(state_key1[0])
        return ppr.fast_gamma(subkey, alpha_samples)

    @jax.jit
    def run_ref() -> jax.Array:
        state_key2[0], subkey = jax.random.split(state_key2[0])
        return jax.random.gamma(subkey, alpha_samples)

    run_performance_test("Gamma", 20, run_fast, run_ref)


def test_nbinomial_performance() -> None:
    n = 1_000_000
    key = jax.random.key(1004)
    size = jnp.array([1.0, 5.0, 20.0, 100.0], dtype=jnp.float32)
    p = jnp.array([0.1, 0.5, 0.9], dtype=jnp.float32)
    size_grid, p_grid = jnp.meshgrid(size, p, indexing="ij")
    size_flat = size_grid.reshape(-1)
    p_flat = p_grid.reshape(-1)
    n_repeat = max(1, n // len(size_flat))
    size_samples = jnp.tile(size_flat, n_repeat)
    p_samples = jnp.tile(p_flat, n_repeat)

    key1 = key
    state_key1 = [key1]

    @jax.jit
    def run_fast() -> jax.Array:
        state_key1[0], subkey = jax.random.split(state_key1[0])
        return ppr.fast_nbinomial(subkey, size_samples, p=p_samples)

    run_performance_test("NBinomial", 20, run_fast, None)


# =====================================================================
# Distribution & Comparison Plots
# =====================================================================


def test_poisson_plots() -> None:
    if check_and_skip_ci():
        pytest.skip("Skipping plots in GITHUB_ACTIONS environment")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    seed = 1005
    key = jax.random.key(seed)
    lam_vals = [0.0001, 0.1, 1.0, 4.0, 4.01, 8.0, 15, 19.9, 20.1, 25, 30, 100.0, 500.0]
    n_samples = 1_000_000

    fig, axes = plt.subplots(2, len(lam_vals), figsize=(5 * len(lam_vals), 8))
    if len(lam_vals) == 1:
        axes = axes.reshape(2, 1)
    hist_axes = axes[0, :]
    qq_axes = axes[1, :]

    for i, lam_val in enumerate(lam_vals):
        lam_arr = jnp.full((n_samples,), lam_val)
        key_fast, key = jax.random.split(key)
        fast_samples = ppr.fast_poisson(key_fast, lam_arr)
        key_ref, key = jax.random.split(key)
        ref_samples = jax.random.poisson(key_ref, lam_arr)

        # Plot Histogram
        ax_hist = hist_axes[i]
        min_bin = min(float(jnp.min(ref_samples)), float(jnp.min(fast_samples)))
        max_bin = max(float(jnp.max(ref_samples)), float(jnp.max(fast_samples)))
        bin_edges = jnp.arange(jnp.floor(min_bin), jnp.ceil(max_bin) + 1)

        ax_hist.hist(
            ref_samples,
            bins=bin_edges,
            alpha=0.5,
            label="jax.random.poisson",
            density=True,
            color="C1",
            edgecolor="k",
        )
        ax_hist.hist(
            fast_samples,
            bins=bin_edges,
            alpha=0.5,
            label="pp.fast_poisson",
            density=True,
            color="C0",
            edgecolor="k",
        )
        ax_hist.set_title(r"$\lambda$ = {:.2f}".format(lam_val), fontsize=9)
        ax_hist.set_xlabel("Sample value", fontsize=9)
        if i == 0:
            ax_hist.set_ylabel("Density", fontsize=9)
        ax_hist.tick_params(axis="both", which="major", labelsize=8)
        if i == len(lam_vals) - 1:
            ax_hist.legend(fontsize=8)

        # Plot QQ-Plot
        ax_qq = qq_axes[i]
        sorted_ref = np.sort(np.array(ref_samples))
        sorted_fast = np.sort(np.array(fast_samples))

        # Add jitter to discrete values for QQ plot to show density
        np.random.seed(seed)
        x_jitter = np.random.normal(0, 0.1, size=sorted_ref.size)
        y_jitter = np.random.normal(0, 0.1, size=sorted_fast.size)

        ax_qq.scatter(
            sorted_ref + x_jitter,
            sorted_fast + y_jitter,
            alpha=0.3,
            s=5,
            color="C0",
        )
        min_val = min(float(sorted_ref[0]), float(sorted_fast[0]))
        max_val = max(float(sorted_ref[-1]), float(sorted_fast[-1]))
        ax_qq.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="red",
            linestyle="--",
            alpha=0.7,
            lw=1.5,
            label="y=x",
        )
        ax_qq.set_xlabel("jax.random.poisson quantiles", fontsize=9)
        if i == 0:
            ax_qq.set_ylabel("pp.fast_poisson quantiles", fontsize=9)
        ax_qq.tick_params(axis="both", which="major", labelsize=8)
        if i == len(lam_vals) - 1:
            ax_qq.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "poisson_comparison.png"))
    plt.close()


def test_gamma_plots() -> None:
    if check_and_skip_ci():
        pytest.skip("Skipping plots in GITHUB_ACTIONS environment")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    seed = 1006
    key = jax.random.key(seed)
    alpha_vals = [0.01, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 50.0, 100.0]
    n_samples = 1_000_000

    fig, axes = plt.subplots(2, len(alpha_vals), figsize=(3.5 * len(alpha_vals), 7))
    if len(alpha_vals) == 1:
        axes = axes.reshape(2, 1)
    hist_axes = axes[0, :]
    qq_axes = axes[1, :]

    for i, alpha_val in enumerate(alpha_vals):
        alpha_arr = jnp.full((n_samples,), alpha_val, dtype=jnp.float32)
        key_fast, key = jax.random.split(key)
        fast_samples = ppr.fast_gamma(key_fast, alpha_arr)
        key_ref, key = jax.random.split(key)
        ref_samples = jax.random.gamma(key_ref, alpha_arr)

        # Plot Density Estimate
        ax_hist = hist_axes[i]
        if alpha_val == 0.01:
            import matplotlib.ticker as ticker

            ref_pos = np.array(ref_samples[ref_samples > 0])
            fast_pos = np.array(fast_samples[fast_samples > 0])
            log10_ref = np.log10(ref_pos)
            log10_fast = np.log10(fast_pos)

            kde_ref = stats.gaussian_kde(log10_ref[:20000])
            kde_fast = stats.gaussian_kde(log10_fast[:20000])

            min_log = min(float(np.min(log10_ref)), float(np.min(log10_fast)))
            max_log = max(float(np.max(log10_ref)), float(np.max(log10_fast)))
            grid = np.linspace(min_log, max_log, 200)

            ax_hist.plot(
                grid, kde_ref(grid), label="jax.random.gamma", color="C1", lw=2
            )
            ax_hist.plot(
                grid,
                kde_fast(grid),
                label="pp.fast_gamma",
                color="C0",
                lw=2,
                linestyle="--",
            )
            ax_hist.fill_between(grid, kde_ref(grid), alpha=0.15, color="C1")
            ax_hist.fill_between(grid, kde_fast(grid), alpha=0.15, color="C0")
            ax_hist.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, pos: f"$10^{{{int(np.round(x))}}}$")
            )
        else:
            min_bin = min(float(jnp.min(ref_samples)), float(jnp.min(fast_samples)))
            max_bin = max(float(jnp.max(ref_samples)), float(jnp.max(fast_samples)))
            grid = np.linspace(min_bin, max_bin, 200)

            kde_ref = stats.gaussian_kde(np.array(ref_samples[:20000]))
            kde_fast = stats.gaussian_kde(np.array(fast_samples[:20000]))

            ax_hist.plot(
                grid, kde_ref(grid), label="jax.random.gamma", color="C1", lw=2
            )
            ax_hist.plot(
                grid,
                kde_fast(grid),
                label="pp.fast_gamma",
                color="C0",
                lw=2,
                linestyle="--",
            )
            ax_hist.fill_between(grid, kde_ref(grid), alpha=0.15, color="C1")
            ax_hist.fill_between(grid, kde_fast(grid), alpha=0.15, color="C0")
        ax_hist.set_title(r"$\alpha$ = {:.2f}".format(alpha_val), fontsize=9)
        ax_hist.set_xlabel("Sample value", fontsize=9)
        if i == 0:
            ax_hist.set_ylabel("Density", fontsize=9)
        ax_hist.tick_params(axis="both", which="major", labelsize=8)
        if i == len(alpha_vals) - 1:
            ax_hist.legend(fontsize=8)

        # Plot QQ-Plot
        ax_qq = qq_axes[i]
        sorted_ref = np.sort(np.array(ref_samples))
        sorted_fast = np.sort(np.array(fast_samples))
        ax_qq.scatter(sorted_ref, sorted_fast, alpha=0.3, s=5, color="C0")
        min_val = min(float(sorted_ref[0]), float(sorted_fast[0]))
        max_val = max(float(sorted_ref[-1]), float(sorted_fast[-1]))
        ax_qq.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="red",
            linestyle="--",
            alpha=0.7,
            lw=1.5,
            label="y=x",
        )
        if alpha_val == 0.01:
            ax_qq.set_xscale("log")
            ax_qq.set_yscale("log")
        if i == 0:
            ax_qq.set_ylabel("pp.fast_gamma quantiles", fontsize=9)
        ax_qq.set_xlabel("jax.random.gamma quantiles", fontsize=9)
        ax_qq.tick_params(axis="both", which="major", labelsize=8)
        if i == len(alpha_vals) - 1:
            ax_qq.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "gamma_comparison.png"))
    plt.close()


def test_nbinomial_plots() -> None:
    if check_and_skip_ci():
        pytest.skip("Skipping plots in GITHUB_ACTIONS environment")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    seed = 1007
    key = jax.random.key(seed)
    n_vals = [1.0, 5.0, 20.0, 100.0]
    p_vals = [0.1, 0.5, 0.9]
    n_samples = 1_000_000

    total_plots = len(n_vals) * len(p_vals)
    fig, axes = plt.subplots(2, total_plots, figsize=(3.5 * total_plots, 7))
    hist_axes = axes[0, :]
    qq_axes = axes[1, :]

    plot_idx = 0
    for n_val in n_vals:
        for p_val in p_vals:
            n_arr = jnp.full((n_samples,), n_val, dtype=jnp.float32)
            p_arr = jnp.full((n_samples,), p_val, dtype=jnp.float32)
            key_fast, key = jax.random.split(key)
            fast_samples = ppr.fast_nbinomial(
                key_fast,
                n_arr,
                p=p_arr,
                # gamma_newton_loops=6,
                # poisson_newton_loops=10,
                # poisson_inverse_cdf_loops=20,
            )

            # Histogram comparison with theoretical Scipy PMF
            ax_hist = hist_axes[plot_idx]
            min_bin = int(jnp.min(fast_samples))
            max_bin = int(jnp.max(fast_samples))
            bin_edges = np.arange(min_bin, max_bin + 2) - 0.5

            ax_hist.hist(
                fast_samples,
                bins=bin_edges,
                alpha=0.5,
                label="pp.fast_nbinomial",
                density=True,
                color="C0",
                edgecolor="k",
            )
            x_theory = np.arange(min_bin, max_bin + 1)
            pmf_theory = stats.nbinom.pmf(x_theory, n_val, p_val)
            ax_hist.step(
                x_theory,
                pmf_theory,
                where="mid",
                color="red",
                alpha=0.8,
                label="Exact PMF (SciPy)",
            )
            ax_hist.set_title(f"n={n_val}, p={p_val}", fontsize=9)
            ax_hist.set_xlabel("Sample value", fontsize=9)
            if plot_idx == 0:
                ax_hist.set_ylabel("Density", fontsize=9)
            ax_hist.tick_params(axis="both", which="major", labelsize=8)
            if plot_idx == total_plots - 1:
                ax_hist.legend(fontsize=8)

            # QQ plot against theoretical Scipy quantiles with jitter
            ax_qq = qq_axes[plot_idx]
            sorted_samples = np.sort(np.array(fast_samples))

            quantiles = np.linspace(1e-5, 1 - 1e-5, len(sorted_samples))
            theo_quants = stats.nbinom.ppf(quantiles, n_val, p_val)

            # Add jitter to discrete values for QQ plot to show density
            np.random.seed(seed)
            x_jitter = np.random.normal(0, 0.1, size=theo_quants.size)
            y_jitter = np.random.normal(0, 0.1, size=sorted_samples.size)

            ax_qq.scatter(
                theo_quants + x_jitter,
                sorted_samples + y_jitter,
                alpha=0.3,
                s=5,
                color="C0",
            )
            limit = [
                min(float(theo_quants[0]), float(sorted_samples[0])),
                max(float(theo_quants[-1]), float(sorted_samples[-1])),
            ]
            ax_qq.plot(
                limit,
                limit,
                color="red",
                linestyle="--",
                alpha=0.7,
                lw=1.5,
                label="y=x",
            )
            ax_qq.set_xlabel("scipy.stats.nbinom quantiles", fontsize=9)
            if plot_idx == 0:
                ax_qq.set_ylabel("pp.fast_nbinomial quantiles", fontsize=9)
            ax_qq.tick_params(axis="both", which="major", labelsize=8)
            if plot_idx == total_plots - 1:
                ax_qq.legend(fontsize=8)

            plot_idx += 1

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "nbinomial_comparison.png"))
    plt.close()


def test_binomial_plots() -> None:
    if check_and_skip_ci():
        pytest.skip("Skipping plots in GITHUB_ACTIONS environment")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    seed = 1008
    key = jax.random.key(seed)
    n_trials_list = [3, 20, 100, 2000]
    prob_vals = [0.02 / 365.25, 0.01, 0.1, 0.3, 0.5, 0.8, 0.95, 0.99]
    n_samples = 1_000_000

    plot_data = {}
    for row, n in enumerate(n_trials_list):
        for col, p_val in enumerate(prob_vals):
            p_arr = jnp.full((n_samples,), p_val, dtype=jnp.float32)
            n_arr = jnp.full((n_samples,), n, dtype=jnp.int32)
            key_fast, key = jax.random.split(key)
            fast_samples = ppr.fast_binomial(key_fast, n_arr, p_arr, exact_max=5)

            key_ref, key = jax.random.split(key)
            ref_samples = jax.random.binomial(key_ref, n=n_arr, p=p_arr)
            plot_data[(row, col)] = (fast_samples, ref_samples, n, p_val)

    n_rows = len(n_trials_list)
    n_cols = len(prob_vals)

    # --- Histogram Grid Figure ---
    fig_hist, hist_axes = plt.subplots(
        n_rows, n_cols, figsize=(3 * n_cols, 1.8 * n_rows), squeeze=False
    )
    for row in range(n_rows):
        for col in range(n_cols):
            fast_samples, ref_samples, n, p_val = plot_data[(row, col)]
            ax = hist_axes[row, col]

            min_v = min(int(jnp.min(fast_samples)), int(jnp.min(ref_samples)))
            max_v = max(int(jnp.max(fast_samples)), int(jnp.max(ref_samples)))
            bins = np.arange(min_v, max_v + 2) - 0.5

            ax.hist(
                ref_samples,
                bins=bins,
                alpha=0.5,
                label="jax.random.binomial",
                density=True,
                color="C1",
                edgecolor="k",
            )
            ax.hist(
                fast_samples,
                bins=bins,
                alpha=0.5,
                label="pp.fast_binomial",
                density=True,
                color="C0",
                edgecolor="k",
            )
            ax.set_title(f"n={n}, p={p_val:.4f}", fontsize=8)
            if col == 0:
                ax.set_ylabel("Density", fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel("Value", fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=7)
            if row == 0 and col == n_cols - 1:
                ax.legend(fontsize=7)

    fig_hist.tight_layout()
    fig_hist.savefig(os.path.join(PLOTS_DIR, "binomial_histogram.png"))
    plt.close(fig_hist)

    # --- QQ Plot Grid Figure ---
    fig_qq, qq_axes = plt.subplots(
        n_rows, n_cols, figsize=(3 * n_cols, 1.8 * n_rows), squeeze=False
    )
    for row in range(n_rows):
        for col in range(n_cols):
            fast_samples, ref_samples, n, p_val = plot_data[(row, col)]
            ax = qq_axes[row, col]

            sorted_fast = np.sort(np.array(fast_samples))
            sorted_ref = np.sort(np.array(ref_samples))

            # Add jitter to reduce overplotting for discrete values
            np.random.seed(seed)
            x_jitter = np.random.normal(0, 0.1, size=sorted_ref.size)
            y_jitter = np.random.normal(0, 0.1, size=sorted_fast.size)

            ax.scatter(
                sorted_ref + x_jitter,
                sorted_fast + y_jitter,
                alpha=0.3,
                s=5,
                color="C0",
            )
            if row == n_rows - 1:
                ax.set_xlabel("jax.random.binomial quantiles", fontsize=8)
            if col == 0:
                ax.set_ylabel("pp.fast_binomial quantiles", fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=7)

            limit = [
                min(float(sorted_ref[0]), float(sorted_fast[0])),
                max(float(sorted_ref[-1]), float(sorted_fast[-1])),
            ]
            ax.plot(
                limit,
                limit,
                color="red",
                linestyle="--",
                alpha=0.7,
                lw=1.5,
                label="y=x",
            )
            ax.set_title(f"n={n}, p={p_val:.4f}", fontsize=8)

    fig_qq.tight_layout()
    fig_qq.savefig(os.path.join(PLOTS_DIR, "binomial_qq.png"))
    plt.close(fig_qq)
