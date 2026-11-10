import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spike_model import SEED, WINDOW_S, fit_waveform, recording, simulate, spike_train, window_edges

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Part 5"))
from common import DETRENDED_COLOR, RAW_COLOR, REFRACTORY_MS, SURFACE, TEXT_PRIMARY, THRESHOLD, style_axis  # noqa: E402

HERE = Path(__file__).resolve().parent
PANEL_HEIGHT = 2.8
X_TICK_STEP = 1
OVERLAY_ALPHA = 0.85


def finish(ax, t, ylim, title, edges):
    ax.set_title(title, color=TEXT_PRIMARY, loc="left")
    for edge in edges[1:-1]:
        ax.axvline(edge, color=TEXT_PRIMARY, linewidth=0.8, linestyle=":")
    ax.set_xlim(t[0], t[-1] + (t[1] - t[0]))
    ax.set_xticks(np.arange(0, t[-1] + X_TICK_STEP, X_TICK_STEP))
    ax.set_ylim(*ylim)
    ax.tick_params(labelbottom=True, labelleft=True)
    style_axis(ax, "Time (s)", "Voltage (a.u.)")


def main():
    parser = argparse.ArgumentParser(description="Compare the recording with the windowed spike model (no drift, no noise).")
    parser.add_argument("--window-s", type=float, default=WINDOW_S, help="window width in seconds")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    t, y, peaks, dt, duration = recording()
    if not 0 < args.window_s <= duration:
        parser.error(f"--window-s must be between 0 and {duration:g} s")
    amplitude, tau = fit_waveform(y, peaks, dt)
    edges = window_edges(duration, args.window_s)
    rates, times = simulate(t[peaks], duration, args.window_s, REFRACTORY_MS / 1e3, np.random.default_rng(args.seed))
    sim = spike_train(times, t.size, dt, amplitude, tau)
    name = "constant" if rates.size == 1 else f"{args.window_s:g} s windows"
    model = f"Model, {name} ({', '.join(f'{r:.1f}' for r in rates)} Hz)"
    # Default window keeps the plain file names; any other width gets its own files instead of overwriting them.
    suffix = "" if args.window_s == WINDOW_S else ("_constant" if rates.size == 1 else f"_{args.window_s:g}s")
    # Drift removed, then everything at or below the threshold (the noise) set to 0, so only the spikes are left.
    y = np.where(y > THRESHOLD, y, 0)
    real_label = f"Recording, spikes only (> {THRESHOLD:g} a.u.): {peaks.size} spikes"

    ylim = (min(y.min(), sim.min()) - 0.3, max(y.max(), sim.max()) + 0.3)
    print(f"Seed {args.seed}")
    print(real_label)
    print(f"{model}: {times.size} spikes")

    fig, axes = plt.subplots(2, 1, figsize=(14, PANEL_HEIGHT * 2 + 0.6), facecolor=SURFACE)
    axes[0].plot(t, y, color=RAW_COLOR, linewidth=0.6)
    finish(axes[0], t, ylim, real_label, edges)
    axes[1].plot(t, sim, color=DETRENDED_COLOR, linewidth=0.6)
    finish(axes[1], t, ylim, f"{model}: {times.size} spikes", edges)
    fig.tight_layout()
    fig.savefig(HERE / f"compare_side_by_side{suffix}.png", dpi=150)
    print(f"Saved compare_side_by_side{suffix}.png")

    fig, ax = plt.subplots(figsize=(14, PANEL_HEIGHT + 0.6), facecolor=SURFACE)
    ax.plot(t, y, color=RAW_COLOR, linewidth=0.6, label="Recording, spikes only")
    ax.plot(t, sim, color=DETRENDED_COLOR, linewidth=0.8, alpha=OVERLAY_ALPHA, label=model)
    finish(ax, t, ylim, f"Recording ({peaks.size} spikes) vs {model[0].lower() + model[1:]}: {times.size} spikes", edges)
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / f"compare_overlay{suffix}.png", dpi=150)
    print(f"Saved compare_overlay{suffix}.png")


if __name__ == "__main__":
    main()
