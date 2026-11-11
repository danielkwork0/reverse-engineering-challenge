import argparse

import numpy as np

from common import THRESHOLD, REFRACTORY_MS, detect_spikes, load, refractory_samples_from, sampling_interval

WIDTHS = [0.5, 1, 2, 2.5, 3, 4, 5, 10]
OFFSETS = 20


def bin_cost(spike_times, width, duration):
    # Shimazaki & Shinomoto (2007). Only full bins are used; when the width doesn't divide the recording,
    # the cost is averaged over every placement of the leftover piece.
    n = int(duration // width + 1e-9)
    leftover = duration - n * width
    starts = np.linspace(0, leftover, OFFSETS) if leftover > 1e-9 else [0.0]
    costs, counts = [], None
    for s in starts:
        k = np.histogram(spike_times, s + np.arange(n + 1) * width)[0]
        costs.append((2 * k.mean() - k.var()) / width ** 2)
        counts = k if counts is None else counts
    return float(np.mean(costs)), n, counts


def main():
    parser = argparse.ArgumentParser(description="Bin-width cost for the firing rate histogram.")
    parser.add_argument("--widths", type=float, nargs="+", default=WIDTHS)
    args = parser.parse_args()

    t, x = load()
    idx = detect_spikes(x, THRESHOLD, refractory_samples_from(t, REFRACTORY_MS))
    duration = t.size * sampling_interval(t)

    print(f"{idx.size} spikes in {duration:g} s (threshold {THRESHOLD:g}, refractory {REFRACTORY_MS:g} ms)")
    print()
    print(f"{'Width (s)':>9}  {'Bins':>4}  {'Cost':>6}  Counts per bin (bins from t = 0)")
    for w in args.widths:
        cost, n, counts = bin_cost(t[idx], w, duration)
        print(f"{w:>9g}  {n:>4}  {cost:>6.2f}  {' '.join(str(c) for c in counts)}")


if __name__ == "__main__":
    main()
