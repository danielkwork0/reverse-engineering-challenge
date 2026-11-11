import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from common import (DETRENDED_COLOR, RAW_COLOR, REFRACTORY_MS, SURFACE, TEXT_PRIMARY, THRESHOLD, detect_spikes,
                    load, refractory_samples_from, style_axis)
from drift_model import X_TICK_STEP, YLIM, fit_drift, sine, spike_mask

HERE = Path(__file__).resolve().parent

SEED = 42
PANEL_HEIGHT = 2.8
HIST_BINS = 60
HIST_RANGE = (-1.5, 1.5)
GAUSS_COLOR = "#1baf7a"


def real_noise(t, x):
    # Raw voltage minus the fitted drift; samples around spikes are NaN so only the noise is left.
    idx = detect_spikes(x, THRESHOLD, refractory_samples_from(t, REFRACTORY_MS))
    keep = spike_mask(t.size, idx)
    params, _ = fit_drift(t, x, keep)
    return np.where(keep, x - sine(t, *params), np.nan), idx.size


def fit_noise(real):
    return real[~np.isnan(real)].std(ddof=1)


def describe(y):
    y = y[~np.isnan(y)]
    lag1 = np.corrcoef(y[:-1], y[1:])[0, 1]
    return [
        f"{y.size}",
        f"{y.mean():.3f}",
        f"{y.std(ddof=1):.3f}",
        f"{stats.skew(y):.3f}",
        f"{stats.kurtosis(y):.3f}",
        f"{y.min():.2f}",
        f"{y.max():.2f}",
        f"{lag1:.3f}",
    ]


ROWS = ["Samples", "Mean (a.u.)", "SD (a.u.)", "Skewness", "Excess kurtosis", "Min (a.u.)", "Max (a.u.)",
        "Autocorrelation, lag 1"]


def finish(ax, t, title):
    ax.set_title(title, color=TEXT_PRIMARY, loc="left")
    ax.set_xlim(t[0], t[-1] + (t[1] - t[0]))
    ax.set_xticks(np.arange(0, t[-1] + X_TICK_STEP, X_TICK_STEP))
    ax.set_ylim(*YLIM)
    style_axis(ax, "Time (s)", "Voltage (a.u.)")


def save(fig, name, seed):
    # Default seed keeps the plain file names; any other seed gets its own files instead of overwriting them.
    if seed != SEED:
        name = name.replace(".png", f"_seed{seed}.png")
    fig.tight_layout()
    fig.savefig(HERE / name, dpi=150)
    print(f"Saved {name}")


def main():
    parser = argparse.ArgumentParser(description="Fit and simulate the noise as Gaussian white noise.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    t, x = load()
    real, n_spikes = real_noise(t, x)
    clean = real[~np.isnan(real)]
    sigma = fit_noise(real)
    sim = np.random.default_rng(args.seed).normal(0, sigma, t.size)

    print(f"Noise model           Gaussian, mean 0, SD {sigma:.3f} a.u., independent samples (seed {args.seed})")
    print(f"Real noise            raw voltage minus fitted drift, {np.isnan(real).sum()} samples around "
          f"{n_spikes} spikes removed")
    print()
    print(f"Normality tests on the real noise (p > 0.05: no evidence against Gaussian)")
    print(f"  D'Agostino-Pearson  p = {stats.normaltest(clean).pvalue:.2f}")
    print(f"  Kolmogorov-Smirnov  p = {stats.kstest(clean, 'norm', args=(0, sigma)).pvalue:.2f}")
    print()
    table = [describe(real), describe(sim)]
    width = max(len(r) for r in ROWS)
    print(f"{'':<{width}}{'Recording':>12}{'Model':>12}")
    for i, row in enumerate(ROWS):
        print(f"{row:<{width}}{table[0][i]:>12}{table[1][i]:>12}")
    print()

    model_title = f"Noise model: Gaussian, mean 0, SD {sigma:.3f} a.u. (seed {args.seed})"
    real_title = "Recording noise: raw voltage minus fitted drift, spikes removed"

    fig, axes = plt.subplots(3, 1, figsize=(14, PANEL_HEIGHT * 3 + 0.6), facecolor=SURFACE)
    axes[0].plot(t, x, color=RAW_COLOR, linewidth=0.6)
    finish(axes[0], t, "Recording, full raw voltage")
    axes[1].plot(t, real, color=RAW_COLOR, linewidth=0.6)
    finish(axes[1], t, real_title)
    axes[2].plot(t, sim, color=DETRENDED_COLOR, linewidth=0.6)
    finish(axes[2], t, model_title)
    save(fig, "noise_side_by_side.png", args.seed)

    fig, ax = plt.subplots(figsize=(14, PANEL_HEIGHT + 0.6), facecolor=SURFACE)
    ax.plot(t, real, color=RAW_COLOR, linewidth=0.6, label="Recording noise")
    ax.plot(t, sim, color=DETRENDED_COLOR, linewidth=0.6, alpha=0.6, label="Noise model")
    finish(ax, t, "Recording noise vs noise model")
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=8)
    save(fig, "noise_overlay.png", args.seed)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True, facecolor=SURFACE)
    grid = np.linspace(*HIST_RANGE, 300)
    for ax, data, color, title in [(axes[0], clean, RAW_COLOR, "Recording noise"),
                                   (axes[1], sim, DETRENDED_COLOR, f"Noise model (seed {args.seed})")]:
        ax.hist(data, bins=HIST_BINS, range=HIST_RANGE, density=True, color=color, alpha=0.8)
        ax.plot(grid, stats.norm.pdf(grid, 0, sigma), color=GAUSS_COLOR, linewidth=1.8,
                label=f"Gaussian, SD {sigma:.3f}")
        ax.set_title(f"{title}: SD {data.std(ddof=1):.3f} a.u.", color=TEXT_PRIMARY, loc="left")
        ax.set_xlim(*HIST_RANGE)
        style_axis(ax, "Voltage (a.u.)", "Density")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    save(fig, "noise_histogram.png", args.seed)


if __name__ == "__main__":
    main()
