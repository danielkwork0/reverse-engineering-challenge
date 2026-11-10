import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Part 5"))
from common import (BASELINE_WINDOW_MS, DETRENDED_COLOR, RAW_COLOR, REFRACTORY_MS, SURFACE, TEXT_PRIMARY,  # noqa: E402
                    THRESHOLD, baseline, detect_spikes, load, refractory_samples_from, style_axis)

HERE = Path(__file__).resolve().parent
# Samples removed around each detected spike: from just before the onset to past the end of the decay (tau ~3 ms).
MASK_BEFORE = 2
MASK_AFTER = 12
P0 = (0.65, 0.1, 0.0, 0.0)
PANEL_HEIGHT = 2.8
X_TICK_STEP = 1
# Fixed range that fits the raw voltage (-1.7 to 6.2), so every drift panel is on the same scale as the full data.
YLIM = (-2, 6.5)


def sine(t, amplitude, frequency, phase, offset):
    return amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset


def spike_mask(n_samples, idx):
    keep = np.ones(n_samples, bool)
    for i in idx:
        keep[max(0, i - MASK_BEFORE):i + MASK_AFTER] = False
    return keep


def fit_drift(t, x, keep):
    params, cov = curve_fit(sine, t[keep], x[keep], p0=P0)
    return params, np.sqrt(np.diag(cov))


def main():
    t, x = load()
    idx = detect_spikes(x, THRESHOLD, refractory_samples_from(t, REFRACTORY_MS))
    keep = spike_mask(t.size, idx)
    (amplitude, frequency, phase, offset), (e_a, e_f, e_p, e_c) = fit_drift(t, x, keep)
    drift = sine(t, amplitude, frequency, phase, offset)
    real = baseline(t, x)
    residual = x - drift

    print(f"Samples used          {keep.sum()} of {t.size} ({(~keep).sum()} masked around {idx.size} spikes)")
    print(f"Amplitude A           {amplitude:.3f} ± {e_a:.3f} a.u.")
    print(f"Frequency f           {frequency:.4f} ± {e_f:.4f} Hz (period {1 / frequency:.2f} s)")
    print(f"Phase                 {phase:.2f} ± {e_p:.2f} rad")
    print(f"Offset c              {offset:.3f} ± {e_c:.3f} a.u.")
    print(f"Residual SD           {residual[keep].std():.3f} a.u. (spikes masked)")
    for lo, hi in [(0, 5), (5, 10)]:
        part = keep & (t >= lo) & (t < hi)
        print(f"Residual mean {lo}-{hi} s".ljust(22) + f"{residual[part].mean():.3f} a.u.")

    print(f"Model vs running median   mean difference {(drift - real).mean():.3f} a.u., "
          f"largest {np.abs(drift - real).max():.3f} a.u.")

    model_title = (f"Drift model: {amplitude:.3f} sin(2\u03c0 \u00b7 {frequency:.4f} t + {phase:.2f}) + {offset:.3f}  "
                   f"(period {1 / frequency:.2f} s)")
    real_title = f"Recording drift: {BASELINE_WINDOW_MS} ms running median of the raw voltage"

    fig, ax = plt.subplots(figsize=(14, PANEL_HEIGHT + 0.6), facecolor=SURFACE)
    ax.plot(t, drift, color=DETRENDED_COLOR, linewidth=1.8)
    finish(ax, t, model_title)
    save(fig, "drift_fit.png")

    raw_title = "Recording, full raw voltage"
    fig, axes = plt.subplots(3, 1, figsize=(14, PANEL_HEIGHT * 3 + 0.6), facecolor=SURFACE)
    axes[0].plot(t, x, color=RAW_COLOR, linewidth=0.6)
    finish(axes[0], t, raw_title)
    axes[1].plot(t, real, color=RAW_COLOR, linewidth=1.8)
    finish(axes[1], t, real_title)
    axes[2].plot(t, drift, color=DETRENDED_COLOR, linewidth=1.8)
    finish(axes[2], t, model_title)
    save(fig, "drift_side_by_side.png")

    fig, ax = plt.subplots(figsize=(14, PANEL_HEIGHT + 0.6), facecolor=SURFACE)
    ax.plot(t, x, color=RAW_COLOR, linewidth=0.6, label="Recording, full raw voltage")
    ax.plot(t, drift, color=DETRENDED_COLOR, linewidth=1.8, label="Drift model (fitted sine)")
    finish(ax, t, "Recording (full raw voltage) vs drift model")
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=8)
    save(fig, "drift_overlay.png")


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


if __name__ == "__main__":
    main()
