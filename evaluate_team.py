import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

REQUIRED = ("t", "drift", "spike_signal", "noise", "reconstruction", "spike_indices")
THRESHOLD = 3.0
DEAD_TIME_S = 0.005
MATCH_TOLERANCE = 2
MASK_BEFORE = 2
MASK_AFTER = 12
BASELINE_WINDOW_S = 0.501
MAD_TO_SD = 1.4826
MAX_LAG = 5
WEIGHTS = dict(spikes=0.35, fit=0.35, drift=0.1, whiteness=0.1, noise=0.1)
# Distances at which a part score reaches 0.
DRIFT_RMS_LIMIT = 0.2
WHITENESS_LIMIT = 0.1
NOISE_SD_LIMIT = 0.2
ZOOM = (0.3, 0.5)
COLORS = dict(raw="#2a78d6", model="#eb6834", ref="#1baf7a")


def detect_reference_spikes(x, dead_samples):
    spikes = []
    for i in np.flatnonzero(x > THRESHOLD):
        if not spikes or i - spikes[-1] >= dead_samples:
            spikes.append(i)
    return np.array(spikes, dtype=int)


def spike_mask(n, idx):
    keep = np.ones(n, bool)
    for i in idx:
        keep[max(0, i - MASK_BEFORE):i + MASK_AFTER] = False
    return keep


def running_median(x, window):
    window |= 1
    return np.median(sliding_window_view(np.pad(x, window // 2, mode="edge"), window), axis=1)


def noise_floor(x, keep):
    # From first differences, so the drift doesn't count as noise and no model of ours is assumed.
    d = np.diff(x)[keep[1:] & keep[:-1]]
    return MAD_TO_SD * np.median(np.abs(d - np.median(d))) / np.sqrt(2)


def match_spikes(ref, cand):
    used = np.zeros(ref.size, bool)
    hits = 0
    for c in np.sort(cand):
        dist = np.where(used, MATCH_TOLERANCE + 1, np.abs(ref - c))
        if dist.size and dist.min() <= MATCH_TOLERANCE:
            used[dist.argmin()] = True
            hits += 1
    precision = hits / cand.size if cand.size else 0.0
    recall = hits / ref.size if ref.size else 0.0
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    return precision, recall, f1


def autocorr(y):
    y = y - y.mean()
    return np.array([np.dot(y[:-k], y[k:]) for k in range(1, MAX_LAG + 1)]) / np.dot(y, y)


def check_validity(t_ref, c):
    n = t_ref.size
    checks = {"has_all_arrays": all(k in c for k in REQUIRED)}
    if not checks["has_all_arrays"]:
        checks["missing"] = [k for k in REQUIRED if k not in c]
        checks["valid"] = False
        return checks
    checks["shapes_ok"] = all(c[k].shape == (n,) for k in REQUIRED if k != "spike_indices")
    checks["time_matches"] = bool(checks["shapes_ok"] and np.allclose(c["t"], t_ref))
    checks["all_finite"] = all(bool(np.isfinite(c[k]).all()) for k in REQUIRED)
    idx = c["spike_indices"]
    checks["spike_indices_ok"] = bool(idx.ndim == 1 and np.all(idx == np.round(idx)) and np.all((idx >= 0) & (idx < n))
                                      and np.all(np.diff(idx) > 0))
    if checks["shapes_ok"]:
        structured = c["drift"] + c["spike_signal"]
        checks["reconstruction_is_drift_plus_spikes"] = bool(np.allclose(c["reconstruction"], structured, atol=1e-6))
    checks["valid"] = all(checks[k] for k in ("shapes_ok", "time_matches", "all_finite", "spike_indices_ok"))
    return checks


def overlay(t, x, c, ref_idx):
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    zoom = (t >= ZOOM[0]) & (t < ZOOM[1])
    for ax, sel, title in [(axes[0], slice(None), "Full recording"),
                           (axes[1], zoom, f"Zoom {ZOOM[0]:g}-{ZOOM[1]:g} s")]:
        ax.plot(t[sel], x[sel], color=COLORS["raw"], linewidth=0.5, label="Recording")
        ax.plot(t[sel], c["reconstruction"][sel], color=COLORS["model"], linewidth=0.9, label="Candidate drift + spikes")
        lo, hi = t[sel][0], t[sel][-1]
        r = t[ref_idx][(t[ref_idx] >= lo) & (t[ref_idx] <= hi)]
        k = t[c["spike_indices"].astype(int)]
        k = k[(k >= lo) & (k <= hi)]
        ax.plot(r, np.full(r.size, x.max() + 0.3), "v", color=COLORS["ref"], markersize=5, label="Reference spikes")
        ax.plot(k, np.full(k.size, x.max() + 0.8), "v", color=COLORS["model"], markersize=5, label="Candidate spikes")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (a.u.)")
    axes[0].legend(loc="lower left", ncol=4, fontsize=8)
    fig.tight_layout()
    return fig


def evaluate(reference_file, candidate_file):
    with np.load(reference_file) as ref:
        t, x = np.asarray(ref["t"], float), np.asarray(ref["raw_data"], float).ravel()
    with np.load(candidate_file) as data:
        c = {k: np.asarray(data[k]) for k in data.files}
    checks = check_validity(t, c)
    result = {"candidate": str(candidate_file), "validity": checks, "metrics": {}, "part_scores": {},
              "final_score": 0.0, "overlay_figure": None}
    if not checks["valid"]:
        return result

    dt = float(np.median(np.diff(t)))
    ref_idx = detect_reference_spikes(x, int(round(DEAD_TIME_S / dt)))
    keep = spike_mask(x.size, ref_idx)
    floor = noise_floor(x, keep)
    residual = x - (c["drift"] + c["spike_signal"])
    precision, recall, f1 = match_spikes(ref_idx, c["spike_indices"].astype(int))
    baseline = running_median(np.where(keep, x, np.nan), int(round(BASELINE_WINDOW_S / dt)))
    ok = ~np.isnan(baseline)
    drift_rms = float(np.sqrt(np.mean((c["drift"][ok] - baseline[ok]) ** 2)))
    acf = autocorr(residual)
    excess_acf = max(0.0, float(np.abs(acf).mean()) - 2 / np.sqrt(x.size))
    residual_sd = float(residual.std(ddof=1))
    noise_sd = float(c["noise"].std(ddof=1))

    parts = {
        "spikes": f1,
        "fit": min(floor / residual_sd, residual_sd / floor),
        "drift": max(0.0, 1 - drift_rms / DRIFT_RMS_LIMIT),
        "whiteness": max(0.0, 1 - excess_acf / WHITENESS_LIMIT),
        "noise": max(0.0, 1 - abs(noise_sd / floor - 1) / NOISE_SD_LIMIT),
    }
    result["metrics"] = {
        "n_spikes_reference": int(ref_idx.size),
        "n_spikes_candidate": int(c["spike_indices"].size),
        "spike_precision": round(precision, 4),
        "spike_recall": round(recall, 4),
        "spike_f1": round(f1, 4),
        "noise_floor_sd": round(float(floor), 4),
        "residual_sd": round(residual_sd, 4),
        "drift_rms_vs_running_median": round(drift_rms, 4),
        "residual_autocorr_lags_1_to_5": [round(float(a), 4) for a in acf],
        "candidate_noise_sd": round(noise_sd, 4),
    }
    result["part_scores"] = {k: round(float(v), 4) for k, v in parts.items()}
    result["final_score"] = round(100 * sum(WEIGHTS[k] * parts[k] for k in WEIGHTS), 2)
    result["overlay_figure"] = overlay(t, x, c, ref_idx)
    return result


def main():
    parser = argparse.ArgumentParser(description="Score team reconstructions against spike_data.npz.")
    parser.add_argument("reference_file")
    parser.add_argument("candidate_files", nargs="+")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    rows = []
    for candidate in args.candidate_files:
        result = evaluate(args.reference_file, candidate)
        name = Path(candidate).stem
        fig = result.pop("overlay_figure")
        if fig is not None:
            fig.savefig(out / f"{name}_overlay.png", dpi=120)
            plt.close(fig)
        (out / f"{name}_result.json").write_text(json.dumps(result, indent=2))
        rows.append((name, result))

    width = max(len(n) for n, _ in rows)
    print(f"{'Candidate':<{width}}  " + "".join(f"{k:>11}" for k in WEIGHTS) + f"{'Final':>9}  Valid")
    for name, r in rows:
        parts = "".join(f"{r['part_scores'].get(k, 0):>11.3f}" for k in WEIGHTS)
        print(f"{name:<{width}}  {parts}{r['final_score']:>9.2f}  {r['validity']['valid']}")
    print(f"\nWeights: {', '.join(f'{k} {w:g}' for k, w in WEIGHTS.items())}. Results saved in {out}/")


if __name__ == "__main__":
    main()
