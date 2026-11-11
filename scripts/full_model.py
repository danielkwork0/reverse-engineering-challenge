import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import noise_model
import spike_model
from common import (DETRENDED_COLOR, RAW_COLOR, REFRACTORY_MS, SURFACE, TEXT_PRIMARY, THRESHOLD, baseline,
                    detect_spikes, load, refractory_samples_from, sampling_interval, style_axis)
from drift_model import X_TICK_STEP, YLIM, fit_drift, sine, spike_mask
from noise_model import fit_noise
from spike_model import PEAK_SEARCH, fit_waveform, simulate, spike_train

HERE = Path(__file__).resolve().parent

PANEL_HEIGHT = 2.8
HIST_BINS = 120
HIST_RANGE = (-2, 6.5)


def fit_signal(t, x):
    # Same steps as the spikes/, drift/ and noise/ scripts, but on any signal, so the tests can refit synthetic data.
    dt = sampling_interval(t)
    idx = detect_spikes(x, THRESHOLD, refractory_samples_from(t, REFRACTORY_MS))
    y = x - baseline(t, x)
    peaks = np.array([i + np.argmax(y[i:i + PEAK_SEARCH]) for i in idx], dtype=int)
    amplitude, tau = fit_waveform(y, peaks, dt)
    keep = spike_mask(t.size, idx)
    drift, _ = fit_drift(t, x, keep)
    sigma = fit_noise(np.where(keep, x - sine(t, *drift), np.nan))
    return peaks, dt, t.size * dt, amplitude, tau, drift, sigma


def fit_all():
    t, x = load()
    return (t, x, *fit_signal(t, x))


def synthesize(t, peaks, dt, duration, amplitude, tau, drift, sigma, window_s, spike_seed, noise_seed):
    rates, times = simulate(t[peaks], duration, window_s, REFRACTORY_MS / 1e3, np.random.default_rng(spike_seed))
    spikes = spike_train(times, t.size, dt, amplitude, tau)
    noise = np.random.default_rng(noise_seed).normal(0, sigma, t.size)
    return spikes + sine(t, *drift) + noise, rates, times


def describe(t, x):
    idx = detect_spikes(x, THRESHOLD, refractory_samples_from(t, REFRACTORY_MS))
    noise = x[spike_mask(t.size, idx)]
    isi = np.diff(t[idx]) * 1e3
    return [
        f"{x.min():.2f}",
        f"{x.max():.2f}",
        f"{x.mean():.3f}",
        f"{np.median(x):.3f}",
        f"{x.std(ddof=1):.3f}",
        f"{idx.size}",
        f"{idx.size / (t.size * (t[1] - t[0])):.2f}",
        f"{isi.mean():.0f}",
        f"{isi.std() / isi.mean():.2f}",
        f"{noise.std(ddof=1):.3f}",
    ]


ROWS = ["Min (a.u.)", "Max (a.u.)", "Mean (a.u.)", "Median (a.u.)", "SD (a.u.)",
        f"Spikes detected (threshold {THRESHOLD:g}, {REFRACTORY_MS:g} ms)", "Rate (Hz)", "ISI mean (ms)", "ISI CV",
        "SD without spikes, drift left in (a.u.)"]


def finish(ax, t, title):
    ax.set_title(title, color=TEXT_PRIMARY, loc="left")
    ax.set_xlim(t[0], t[-1] + (t[1] - t[0]))
    ax.set_xticks(np.arange(0, t[-1] + X_TICK_STEP, X_TICK_STEP))
    ax.set_ylim(*YLIM)
    style_axis(ax, "Time (s)", "Voltage (a.u.)")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(HERE / name, dpi=150)
    print(f"Saved {name}")


def main():
    parser = argparse.ArgumentParser(description="Synthesize the full signal (spikes + drift + noise) from the model.")
    parser.add_argument("--window-s", type=float, default=spike_model.WINDOW_S)
    parser.add_argument("--spike-seed", type=int, default=spike_model.SEED)
    parser.add_argument("--noise-seed", type=int, default=noise_model.SEED)
    args = parser.parse_args()

    t, x, peaks, dt, duration, amplitude, tau, drift, sigma = fit_all()
    if not 0 < args.window_s <= duration:
        parser.error(f"--window-s must be between 0 and {duration:g} s")
    synth, rates, times = synthesize(t, peaks, dt, duration, amplitude, tau, drift, sigma, args.window_s,
                                     args.spike_seed, args.noise_seed)

    a_d, f_d, p_d, c_d = drift
    print("Model coefficients (all fitted from the recording)")
    print(f"  Spikes   A {amplitude:.2f} a.u., tau {tau * 1e3:.2f} ms, {args.window_s:g} s windows at "
          f"{', '.join(f'{r:.1f}' for r in rates)} Hz, dead time {REFRACTORY_MS:g} ms (seed {args.spike_seed})")
    print(f"  Drift    {a_d:.3f} sin(2 pi {f_d:.4f} t + {p_d:.2f}) + {c_d:.3f}  (period {1 / f_d:.2f} s)")
    print(f"  Noise    Gaussian, mean 0, SD {sigma:.3f} a.u. (seed {args.noise_seed})")
    print(f"  Spikes generated: {times.size}")
    print()
    table = [describe(t, x), describe(t, synth)]
    width = max(len(r) for r in ROWS)
    print(f"{'':<{width}}{'Recording':>12}{'Model':>12}")
    for i, row in enumerate(ROWS):
        print(f"{row:<{width}}{table[0][i]:>12}{table[1][i]:>12}")
    ks = stats.ks_2samp(x, synth)
    print()
    print(f"Two-sample KS test on all voltage values: D = {ks.statistic:.3f}, p = {ks.pvalue:.2f}")
    print()

    model_title = f"Model: spikes + drift + noise (spike seed {args.spike_seed}, noise seed {args.noise_seed})"
    fig, axes = plt.subplots(2, 1, figsize=(14, PANEL_HEIGHT * 2 + 0.6), facecolor=SURFACE)
    axes[0].plot(t, x, color=RAW_COLOR, linewidth=0.6)
    finish(axes[0], t, "Recording, full raw voltage")
    axes[1].plot(t, synth, color=DETRENDED_COLOR, linewidth=0.6)
    finish(axes[1], t, model_title)
    save(fig, "full_side_by_side.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True, facecolor=SURFACE)
    for ax, data, color, title in [(axes[0], x, RAW_COLOR, "Recording"), (axes[1], synth, DETRENDED_COLOR, "Model")]:
        ax.hist(data, bins=HIST_BINS, range=HIST_RANGE, color=color, alpha=0.8)
        ax.set_yscale("log")
        ax.set_title(f"{title}: all voltage values", color=TEXT_PRIMARY, loc="left")
        ax.set_xlim(*HIST_RANGE)
        style_axis(ax, "Voltage (a.u.)", "Samples (log scale)")
    save(fig, "full_histogram.png")


if __name__ == "__main__":
    main()
