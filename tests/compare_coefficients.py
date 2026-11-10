import sys
from pathlib import Path

import numpy as np

from test_model import DT, N, RECOVERY_SEEDS, T, TOL, TRUE, WINDOW_S, synthetic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "full_model"))
from full_model import fit_all, fit_signal, synthesize  # noqa: E402

NAMES = ["Spike amplitude A (a.u.)", "Spike decay tau (ms)", "Drift amplitude (a.u.)", "Drift frequency (Hz)",
         "Drift phase (rad)", "Drift offset (a.u.)", "Noise SD sigma (a.u.)"]
FORMATS = ["{:.3f}", "{:.3f}", "{:.4f}", "{:.5f}", "{:.3f}", "{:.4f}", "{:.4f}"]


def flat(amplitude, tau, drift, sigma):
    return [amplitude, tau * 1e3, *drift, sigma]


def table(header, rows):
    width = max(len(n) for n in NAMES)
    print(f"{'':<{width}}" + "".join(f"{h:>14}" for h in header))
    for i, name in enumerate(NAMES):
        print(f"{name:<{width}}" + "".join(f"{r[i]:>14}" for r in rows))


def main():
    true = flat(TRUE["amplitude"], TRUE["tau"], TRUE["drift"], TRUE["sigma"])
    tol = flat(TOL["amplitude"], TOL["tau"], TOL["drift"], TOL["sigma"])

    fits = []
    for s in RECOVERY_SEEDS:
        _, _, _, amplitude, tau, drift, sigma = fit_signal(T, synthetic(s, 100 + s)[0])
        fits.append(flat(amplitude, tau, drift, sigma))
    fits = np.array(fits)
    worst = np.abs(fits - true).max(axis=0)

    print(f"1. Parameter recovery: synthetic signals built with known values, then fitted "
          f"({len(RECOVERY_SEEDS)} seeds, {N} samples at {1 / DT:g} Hz)")
    print()
    table(["True", "Fitted mean", "Fitted SD", "Worst error", "Tolerance", "Pass"],
          [[f.format(v) for f, v in zip(FORMATS, true)],
           [f.format(v) for f, v in zip(FORMATS, fits.mean(axis=0))],
           [f.format(v) for f, v in zip(FORMATS, fits.std(axis=0))],
           [f.format(v) for f, v in zip(FORMATS, worst)],
           [f.format(v) for f, v in zip(FORMATS, tol)],
           ["yes" if w <= t else "NO" for w, t in zip(worst, tol)]])
    print()

    t, x, peaks, dt, duration, amplitude, tau, drift, sigma = fit_all()
    synth, _, _ = synthesize(t, peaks, dt, duration, amplitude, tau, drift, sigma, WINDOW_S, 0, 42)
    _, _, _, a2, tau2, drift2, sigma2 = fit_signal(t, synth)
    real = flat(amplitude, tau, drift, sigma)
    refit = flat(a2, tau2, drift2, sigma2)
    error = np.abs(np.array(refit) - real)

    print("2. Round trip: coefficients fitted from the recording, then refitted from the full model built with them "
          "(spike seed 0, noise seed 42)")
    print()
    table(["Recording", "Model refit", "Error", "Tolerance", "Pass"],
          [[f.format(v) for f, v in zip(FORMATS, real)],
           [f.format(v) for f, v in zip(FORMATS, refit)],
           [f.format(v) for f, v in zip(FORMATS, error)],
           [f.format(v) for f, v in zip(FORMATS, tol)],
           ["yes" if e <= t else "NO" for e, t in zip(error, tol)]])


if __name__ == "__main__":
    main()
