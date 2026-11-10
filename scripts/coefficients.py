import argparse
import sys
from pathlib import Path

import numpy as np

from spike_model import SEED, WINDOW_S, fit_waveform, recording, simulate, spike_train, window_edges, window_rates

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Part 5"))
from common import REFRACTORY_MS  # noqa: E402


def coefficients(times, trace, dt, edges):
    peaks = np.round(times / dt).astype(int)
    isi = np.diff(times) * 1e3
    amplitude, tau = fit_waveform(trace, peaks, dt)
    heights = trace[peaks]
    return [
        f"{times.size}",
        f"{times.size / edges[-1]:.2f}",
        ", ".join(f"{r:.1f}" for r in window_rates(times, edges)),
        f"{isi.mean():.0f}",
        f"{np.median(isi):.0f}",
        f"{isi.min():.0f}",
        f"{isi.max():.0f}",
        f"{isi.std() / isi.mean():.2f}",
        f"{heights.mean():.2f} ± {heights.std():.2f}",
        f"{amplitude:.2f}",
        f"{tau * 1e3:.2f}",
    ]


def rows(window_s):
    return [
        "Spikes",
        "Rate (Hz)",
        f"Rate per {window_s:g} s window (Hz)",
        "ISI mean (ms)",
        "ISI median (ms)",
        "ISI min (ms)",
        "ISI max (ms)",
        "ISI CV",
        "Peak height mean ± SD (a.u.)",
        "Fitted amplitude A (a.u.)",
        "Fitted decay tau (ms)",
    ]


def main():
    parser = argparse.ArgumentParser(description="Spike train coefficients for the recording and the windowed model.")
    parser.add_argument("--window-s", type=float, default=WINDOW_S,
                        help="window width in seconds for the windowed model")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    t, y, peaks, dt, duration = recording()
    if not 0 < args.window_s <= duration:
        parser.error(f"--window-s must be between 0 and {duration:g} s")
    amplitude, tau = fit_waveform(y, peaks, dt)
    columns = [("Recording", t[peaks], y)]
    rates, times = simulate(t[peaks], duration, args.window_s, REFRACTORY_MS / 1e3, np.random.default_rng(args.seed))
    name = "Constant" if rates.size == 1 else f"{args.window_s:g} s windows"
    columns.append((name, times, spike_train(times, t.size, dt, amplitude, tau)))

    table = [coefficients(times, trace, dt, window_edges(duration, args.window_s)) for _, times, trace in columns]
    labels = rows(args.window_s)
    if rates.size == 1:
        labels = [l for l in labels if "window" not in l]
        table = [[v for l, v in zip(rows(args.window_s), c) if "window" not in l] for c in table]
    width = max(len(r) for r in labels)
    col = max(len(v) for c in table for v in c + [n for n, *_ in columns]) + 2
    print(f"Seed {args.seed}. Recording = drift removed, spikes found with threshold 3, refractory 5 ms.")
    print()
    print(f"{'':<{width}}" + "".join(f"{n:>{col}}" for n, *_ in columns))
    for i, row in enumerate(labels):
        print(f"{row:<{width}}" + "".join(f"{c[i]:>{col}}" for c in table))


if __name__ == "__main__":
    main()
