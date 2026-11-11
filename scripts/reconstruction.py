from pathlib import Path

import numpy as np

from drift_model import sine
from full_model import fit_all
from spike_model import spike_train

TEAM_ID = 7
NOISE_SEED = 42
OUT_FILE = Path(__file__).resolve().parent.parent / f"team_{TEAM_ID}_reconstruction.npz"


def main():
    t, x, peaks, dt, _, amplitude, tau, drift_params, sigma = fit_all()
    drift = sine(t, *drift_params)
    spike_signal = spike_train(t[peaks], t.size, dt, amplitude, tau)
    noise = np.random.default_rng(NOISE_SEED).normal(0, sigma, t.size)
    reconstruction = drift + spike_signal
    np.savez(OUT_FILE, t=t, drift=drift, spike_signal=spike_signal, noise=noise, reconstruction=reconstruction,
             spike_indices=peaks)

    a_d, f_d, p_d, c_d = drift_params
    print(f"Spikes   {peaks.size}, A {amplitude:.3f} a.u., tau {tau * 1e3:.3f} ms")
    print(f"Drift    {a_d:.4f} sin(2 pi {f_d:.5f} t + {p_d:.3f}) + {c_d:.4f}")
    print(f"Noise    Gaussian, SD {sigma:.4f} (seed {NOISE_SEED})")
    print(f"Residual SD (recording - reconstruction) {(x - reconstruction).std(ddof=1):.4f}")
    print(f"Saved {OUT_FILE.name}")


if __name__ == "__main__":
    main()
