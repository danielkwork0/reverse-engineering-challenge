import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for part in ("spikes", "drift", "noise", "full_model"):
    sys.path.insert(0, str(HERE.parent / part))
from drift_model import MASK_AFTER, MASK_BEFORE, sine, spike_mask  # noqa: E402
from full_model import fit_all, fit_signal, synthesize  # noqa: E402
from noise_model import fit_noise  # noqa: E402
from spike_model import decay, poisson_times, spike_train, window_edges, window_rates  # noqa: E402

N = 10_000
DT = 0.001
T = np.arange(N) * DT
DURATION = N * DT
DEAD_TIME = 0.005
WINDOW_S = 5.0
# Known values used to build the synthetic signals; close to the recording's own fit.
TRUE = dict(amplitude=5.0, tau=0.003, drift=(0.65, 0.09, 0.3, 0.0), sigma=0.3, rates=(4.6, 3.8))
RECOVERY_SEEDS = range(10)
# Tolerances are about 4 SD of each fitted value's spread over 20 synthetic runs, so a correct fit passes every time.
TOL = dict(amplitude=0.2, tau=0.0003, drift=(0.015, 0.0015, 0.06, 0.02), sigma=0.008)
RATE_RUNS = 2000


def synthetic(spike_seed, noise_seed):
    edges = window_edges(DURATION, WINDOW_S)
    times = poisson_times(np.array(TRUE["rates"]), edges, DEAD_TIME, np.random.default_rng(spike_seed))
    spikes = spike_train(times, N, DT, TRUE["amplitude"], TRUE["tau"])
    noise = np.random.default_rng(noise_seed).normal(0, TRUE["sigma"], N)
    return spikes + sine(T, *TRUE["drift"]) + noise, times


class TestSmallFunctions(unittest.TestCase):
    def test_sine_known_values(self):
        period = 10.0
        values = sine(np.array([0, period / 4, period / 2]), 2.0, 1 / period, 0.0, 0.5)
        np.testing.assert_allclose(values, [0.5, 2.5, 0.5], atol=1e-12)

    def test_spike_mask_covers_window(self):
        keep = spike_mask(100, [50])
        self.assertEqual(np.flatnonzero(~keep).tolist(), list(range(50 - MASK_BEFORE, 50 + MASK_AFTER)))

    def test_spike_mask_at_signal_edges(self):
        keep = spike_mask(100, [0, 99])
        self.assertEqual(np.flatnonzero(~keep).tolist(), [*range(0, MASK_AFTER), *range(99 - MASK_BEFORE, 100)])

    def test_window_edges_divides(self):
        self.assertEqual(window_edges(10.0, 5.0).tolist(), [0, 5, 10])

    def test_window_edges_leftover(self):
        self.assertEqual(window_edges(10.0, 3.0).tolist(), [0, 3, 6, 9, 10])

    def test_window_rates(self):
        times = np.array([0.5, 1.0, 1.5, 6.0])
        self.assertEqual(window_rates(times, np.array([0, 5, 10])).tolist(), [0.6, 0.2])

    def test_spike_train_shape(self):
        x = spike_train(np.array([0.1]), 200, DT, 5.0, 0.003)
        self.assertAlmostEqual(x[100], 5.0)
        self.assertEqual(x[:100].tolist(), [0] * 100)
        np.testing.assert_allclose(x[101:105] / x[100:104], np.exp(-1 / 3))

    def test_spike_train_close_spikes_add(self):
        x = spike_train(np.array([0.1, 0.102]), 200, DT, 5.0, 0.003)
        self.assertAlmostEqual(x[102], 5.0 + float(decay(2, 5.0, 3)))

    def test_poisson_times_respect_dead_time_and_bounds(self):
        times = poisson_times(np.array([50.0, 50.0]), window_edges(DURATION, WINDOW_S), DEAD_TIME,
                              np.random.default_rng(0))
        self.assertGreater(times.size, 100)
        self.assertGreaterEqual(np.diff(times).min(), DEAD_TIME)
        self.assertTrue(((times >= 0) & (times < DURATION)).all())

    def test_fit_noise_ignores_masked_samples(self):
        self.assertAlmostEqual(fit_noise(np.array([1.0, -1.0, np.nan, 1.0, -1.0])), np.std([1, -1, 1, -1], ddof=1))


class TestParameterRecovery(unittest.TestCase):
    def setUp(self):
        self.fits = [fit_signal(T, synthetic(s, 100 + s)[0]) for s in RECOVERY_SEEDS]

    def test_spike_shape_recovered(self):
        for _, _, _, amplitude, tau, _, _ in self.fits:
            self.assertAlmostEqual(amplitude, TRUE["amplitude"], delta=TOL["amplitude"])
            self.assertAlmostEqual(tau, TRUE["tau"], delta=TOL["tau"])

    def test_drift_recovered(self):
        for *_, drift, _ in self.fits:
            for fitted, true, tol in zip(drift, TRUE["drift"], TOL["drift"]):
                self.assertAlmostEqual(fitted, true, delta=tol)

    def test_noise_recovered(self):
        for *_, sigma in self.fits:
            self.assertAlmostEqual(sigma, TRUE["sigma"], delta=TOL["sigma"])

    def test_mean_spike_count_matches_rate(self):
        edges = window_edges(DURATION, WINDOW_S)
        counts = [poisson_times(np.array(TRUE["rates"]), edges, DEAD_TIME, np.random.default_rng(s)).size
                  for s in range(RATE_RUNS)]
        expected = sum(TRUE["rates"]) * WINDOW_S
        # The dead time removes about 1 spike in 42, so the mean sits just below the expected count.
        self.assertAlmostEqual(np.mean(counts), expected, delta=2.0)
        self.assertAlmostEqual(np.std(counts), np.sqrt(expected), delta=1.0)


class TestRoundTrip(unittest.TestCase):
    def test_detector_finds_every_generated_spike(self):
        for s in RECOVERY_SEEDS:
            x, times = synthetic(s, 100 + s)
            peaks = fit_signal(T, x)[0]
            self.assertEqual(peaks.size, times.size)
            np.testing.assert_array_equal(peaks, np.round(times / DT).astype(int))

    def test_refit_of_full_model_returns_recording_coefficients(self):
        t, x, peaks, dt, duration, amplitude, tau, drift, sigma = fit_all()
        synth, _, _ = synthesize(t, peaks, dt, duration, amplitude, tau, drift, sigma, WINDOW_S, 0, 42)
        _, _, _, a2, tau2, drift2, sigma2 = fit_signal(t, synth)
        self.assertAlmostEqual(a2, amplitude, delta=TOL["amplitude"])
        self.assertAlmostEqual(tau2, tau, delta=TOL["tau"])
        for fitted, true, tol in zip(drift2, drift, TOL["drift"]):
            self.assertAlmostEqual(fitted, true, delta=tol)
        self.assertAlmostEqual(sigma2, sigma, delta=TOL["sigma"])


if __name__ == "__main__":
    unittest.main()
