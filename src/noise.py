"""Synthetic per-mode noise generators.

Each generator produces an N-sample mono signal at FS Hz with characteristics
loosely matched to its target acoustic environment. These are stand-ins for
DEMAND/ESC-50 samples and let the pipeline run with zero external downloads;
swapping in real WAVs later only requires replacing `generate_noise`.

Spectral character (per mode):
  quiet   -> low-level white (HVAC floor)
  babble  -> AM-modulated band-limited noise (speech-like)
  traffic -> pink-tilted broadband (engine + rolling tyre rumble)
  wind    -> low-shelf-boosted broadband + sparse impulsive gusts
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, lfilter

from . import config as cfg


def _white(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n)


def _pinkish(n: int, rng: np.random.Generator) -> np.ndarray:
    """Cheap pink-ish noise via a low-order IIR low-shelf on white noise."""
    b, a = butter(2, 0.15, btype="low")
    pink = lfilter(b, a, _white(n, rng))
    pink += 0.3 * _white(n, rng)
    return pink / (np.std(pink) + 1e-12)


def _bandpass(x: np.ndarray, lo: float, hi: float, fs: int) -> np.ndarray:
    b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return lfilter(b, a, x)


def _quiet(n: int, fs: int, rng: np.random.Generator) -> np.ndarray:
    x = 0.05 * _white(n, rng)
    return x


def _babble(n: int, fs: int, rng: np.random.Generator) -> np.ndarray:
    base = _bandpass(_white(n, rng), 200.0, 3500.0, fs)
    t = np.arange(n) / fs
    # Two slow AM envelopes mimicking overlapping talkers.
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * t + rng.uniform(0, 2 * np.pi))
    env *= 0.7 + 0.3 * np.sin(2 * np.pi * 1.3 * t + rng.uniform(0, 2 * np.pi))
    x = base * env
    return x / (np.std(x) + 1e-12)


def _traffic(n: int, fs: int, rng: np.random.Generator) -> np.ndarray:
    pink = _pinkish(n, rng)
    rumble = _bandpass(_white(n, rng), 30.0, 300.0, fs)
    x = 0.6 * pink + 0.8 * (rumble / (np.std(rumble) + 1e-12))
    return x / (np.std(x) + 1e-12)


def _wind(n: int, fs: int, rng: np.random.Generator) -> np.ndarray:
    # Low-shelf boosted broadband.
    b, a = butter(2, 0.06, btype="low")
    base = lfilter(b, a, _white(n, rng)) * 2.5 + 0.4 * _white(n, rng)
    base = base / (np.std(base) + 1e-12)
    # Impulsive gusts (~ 2/sec, heavy-tailed) — makes wind distinguishable
    # from traffic broadband under the IMM's innovation likelihood.
    n_gusts = max(1, int(2.0 * n / fs))
    locs = rng.integers(0, n, size=n_gusts)
    widths = rng.integers(int(0.05 * fs), int(0.3 * fs), size=n_gusts)
    amps = rng.standard_t(df=3, size=n_gusts) * 1.5
    gust = np.zeros(n)
    for loc, w, a_ in zip(locs, widths, amps):
        end = min(n, loc + w)
        env = np.hanning(end - loc) if end > loc else np.array([])
        gust[loc:end] += a_ * env * rng.standard_normal(end - loc)
    return base + gust


_MODE_GENERATORS = {
    "quiet":   _quiet,
    "babble":  _babble,
    "traffic": _traffic,
    "wind":    _wind,
}


def generate_noise(mode: str, n_samples: int,
                   rng: np.random.Generator,
                   fs: int = cfg.FS) -> np.ndarray:
    """Generate `n_samples` of mode-conditioned noise at unit RMS.

    Uses the synthetic generators by default. If `cfg.NOISE_SOURCE == "recorded"`
    and real clips exist for this mode (noise_samples/<mode>/), draws from those
    instead; modes with no clips silently fall back to synthetic. Same interface
    either way, so the rest of the pipeline is unchanged.
    """
    if mode not in _MODE_GENERATORS:
        raise ValueError(f"Unknown mode '{mode}'. Expected one of {cfg.MODE_NAMES}.")
    if cfg.NOISE_SOURCE == "recorded":
        from . import noise_recorded as nr
        if nr.available(mode):
            return nr.generate_recorded(mode, n_samples, rng, fs)
    x = _MODE_GENERATORS[mode](n_samples, fs, rng)
    return x.astype(np.float64)
