from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE.parent / "data" / "spike_data.npz"
THRESHOLD = 3.0
REFRACTORY_MS = 5.0
BASELINE_WINDOW_MS = 501
MAD_TO_SD = 1.4826

RAW_COLOR = "#2a78d6"
DETRENDED_COLOR = "#eb6834"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
PLATEAU = "#ecebe7"


def detect_spikes(x, threshold=THRESHOLD, refractory_samples=5):
    x = np.asarray(x)
    spikes = []
    for i in np.flatnonzero(x > threshold):
        if not spikes or i - spikes[-1] >= refractory_samples:
            spikes.append(i)
    return np.array(spikes, dtype=int)


def refractory_samples_from(t, refractory_ms):
    dt = float(np.median(np.diff(t)))
    return int(round(refractory_ms / 1e3 / dt))


def load():
    with np.load(DATA_FILE) as data:
        return data["t"], data["raw_data"]


def sampling_interval(t):
    return float(np.median(np.diff(t)))


def baseline(t, x):
    window = int(round(BASELINE_WINDOW_MS / 1e3 / sampling_interval(t))) | 1
    padded = np.pad(x, window // 2, mode="edge")
    return np.median(sliding_window_view(padded, window), axis=1)


def noise_sd(y):
    return MAD_TO_SD * np.median(np.abs(y - np.median(y)))


def style_axis(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
