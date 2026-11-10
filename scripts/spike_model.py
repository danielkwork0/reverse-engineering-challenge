import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Part 5"))
from common import (REFRACTORY_MS, THRESHOLD, baseline, detect_spikes, load, refractory_samples_from,  # noqa: E402
                    sampling_interval)

PEAK_SEARCH = 5
FIT_SAMPLES = 8
KERNEL_TAUS = 6
WINDOW_S = 5.0
SEED = 0


def detected_spikes():
    t, x = load()
    y = x - baseline(t, x)
    idx = detect_spikes(x, THRESHOLD, refractory_samples_from(t, REFRACTORY_MS))
    peaks = np.array([i + np.argmax(y[i:i + PEAK_SEARCH]) for i in idx], dtype=int)
    return t, y, peaks


def decay(n, amplitude, tau):
    return amplitude * np.exp(-n / tau)


def fit_waveform(y, peaks, dt):
    peaks = peaks[peaks + FIT_SAMPLES <= y.size]
    mean = np.array([y[p:p + FIT_SAMPLES] for p in peaks]).mean(axis=0)
    (amplitude, tau), _ = curve_fit(decay, np.arange(FIT_SAMPLES), mean, p0=(5, 3))
    return amplitude, tau * dt


def window_edges(duration, width):
    # A width that doesn't divide the recording leaves a shorter last window.
    return np.append(np.arange(0, duration - 1e-9, width), duration)


def window_rates(spike_times, edges):
    return np.histogram(spike_times, edges)[0] / np.diff(edges)


def poisson_times(rates, edges, dead_time, rng):
    # Constant rate inside each window; spikes closer than dead_time to the last kept one are dropped,
    # the same rule the detector uses.
    times = np.sort(np.concatenate([rng.uniform(lo, hi, rng.poisson(r * (hi - lo)))
                                    for r, lo, hi in zip(rates, edges[:-1], edges[1:])]))
    kept = []
    for s in times:
        if not kept or s - kept[-1] >= dead_time:
            kept.append(s)
    return np.array(kept)


def simulate(spike_times, duration, window_s, dead_time, rng):
    edges = window_edges(duration, window_s)
    rates = window_rates(spike_times, edges)
    return rates, poisson_times(rates, edges, dead_time, rng)


def spike_train(spike_times, n_samples, dt, amplitude, tau):
    impulses = np.zeros(n_samples)
    np.add.at(impulses, np.round(spike_times / dt).astype(int).clip(0, n_samples - 1), 1)
    kernel = decay(np.arange(int(KERNEL_TAUS * tau / dt)), amplitude, tau / dt)
    return np.convolve(impulses, kernel)[:n_samples]


def recording():
    t, y, peaks = detected_spikes()
    dt = sampling_interval(t)
    return t, y, peaks, dt, t.size * dt
