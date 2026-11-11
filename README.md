# Reverse-engineering spike_data.npz — Team 7

Carol Farah, Daniel Keren, Mariana Montoya

`data/spike_data.npz` is 10 s of electrode voltage at 1 kHz (10,000 samples). We split it into spikes, drift and
noise, fitted each part, and saved the structured part (drift + spikes) in `team_7_reconstruction.npz`.
The full reasoning, with plots, is in `Challenge.ipynb`.

## Model and parameters

```
x(t) = drift(t) + spikes(t) + noise(t)

drift(t)  = a sin(2 pi f t + phi) + c
spikes(t) = sum over k of  A exp(-(t - t_k) / tau)  for t >= t_k
noise(t)  ~ N(0, sigma^2), independent per sample
```

| Part | Parameter | Value |
|---|---|---|
| Spikes | count | 42 (4.20 Hz) |
| | amplitude A | 4.999 a.u. |
| | decay tau | 2.94 ms |
| | rate per 5 s window | 4.6 Hz, 3.8 Hz |
| | dead time | 5 ms |
| Drift | amplitude a | 0.648 a.u. |
| | frequency f | 0.0908 Hz (period 11.0 s) |
| | phase phi | 0.290 rad |
| | offset c | 0.005 a.u. |
| Noise | SD sigma | 0.298 a.u. |

Residual SD (recording minus reconstruction) is 0.2985, the same as sigma.

## Assumptions

- Every spike has the same shape: it jumps to A in one sample and decays exponentially.
- Spike times are Poisson with a constant rate inside each 5 s window (a bin-width test, Shimazaki–Shinomoto,
  picked 5 s), and no two spikes are closer than 5 ms.
- The drift is one sine. The recording holds less than one full cycle, so the frequency rests on that assumption.
- The noise is white Gaussian (normaltest p 0.47, KS p 0.99, lag-1 autocorrelation -0.02).

## Detection and reconstruction

1. **Detection.** Every sample above 3 a.u. is a candidate. It counts as a spike if it is the first one or at
   least 5 ms after the last accepted spike. The spike index is the highest sample in the next 5 samples after the
   drift is removed (501 ms running median). This gives 42 spikes.
2. **Spike shape.** The 8 samples from each spike peak are averaged and `A exp(-n / tau)` is fitted with
   `curve_fit`.
3. **Drift.** The sine is fitted with `curve_fit` directly on the raw voltage, with the samples from 2 before to 12
   after each spike removed so the spikes don't pull it up.
4. **Noise.** SD of the raw voltage minus the drift, on the same spike-free samples.
5. **Reconstruction.** `drift` is the fitted sine, `spike_signal` is the fitted spike shape placed at every
   detected spike, `reconstruction = drift + spike_signal`, and `noise` is a new Gaussian realization with the
   fitted SD (seed 42), kept separate.

```
cd scripts
python3 reconstruction.py      # writes team_7_reconstruction.npz
```

## Evaluator

`evaluate_team.py` has `evaluate(reference_file, candidate_file)`. It returns a dictionary with validity checks,
metrics, part scores, an overlay figure and `final_score` (0-100).

The validity checks are: all 6 arrays exist, the 10,000-sample arrays have the right shape, `t` matches the
reference, the values are finite, and the spike indices are increasing integers inside the recording. An invalid
file scores 0.

| Part | Weight | Score |
|---|---|---|
| spikes | 0.35 | F1 of candidate vs reference spikes, one-to-one, within 2 samples. Reference spikes: > 3 a.u. with a 5 ms dead time |
| fit | 0.35 | residual SD vs noise floor, `min(r, 1/r)`. The noise floor comes from first differences of the recording away from spikes, so it doesn't depend on anyone's drift model |
| drift | 0.1 | RMS distance from a 501 ms running median of the spike-free recording; 0 at 0.2 a.u. |
| whiteness | 0.1 | mean absolute residual autocorrelation at lags 1-5, above the 2/sqrt(N) chance level; 0 at 0.1 |
| noise | 0.1 | candidate noise SD vs noise floor; 0 at 20% off |

```
python3 evaluate_team.py data/spike_data.npz team_7_reconstruction.npz other_team.npz ... --out-dir results
```

This saves `<name>_result.json` and `<name>_overlay.png` for each candidate and prints a score table.

## Files

| Path | What |
|---|---|
| `Challenge.ipynb` | write-up and plots |
| `data/spike_data.npz` | the recording |
| `team_7_reconstruction.npz` | Part 1 submission |
| `evaluate_team.py` | Part 2 evaluator |
| `scripts/` | `spike_model.py`, `bin_width.py`, `drift_model.py`, `noise_model.py`, `full_model.py`, `reconstruction.py` |
| `tests/` | `test_model.py` (`python3 -m unittest -v`), `compare_coefficients.py` |
