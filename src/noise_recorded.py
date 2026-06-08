"""Drop-in recorded-noise source: read real ambient clips instead of the
synthetic generators, with the *same* interface as `noise.generate_noise`.

Layout (you place the files; nothing is downloaded automatically):

    noise_samples/
        quiet/    *.wav | *.flac
        babble/   ...
        traffic/  ...
        wind/     ...

A suggested mapping from public datasets (DEMAND, ESC-50) is in
noise_samples/README.md. Anything mono/stereo at any sample rate works; clips
are mixed to mono, resampled to FS, and normalized to unit RMS so they sit at
the same scale as the synthetic generators (the rest of the pipeline is
unchanged).

`generate_recorded(mode, n, rng, fs)` returns an `n`-sample unit-RMS clip: it
picks a random file for the mode, then a random start offset, looping if the
file is shorter than `n`. Loaded+resampled files are cached per (mode, fs).
"""
from __future__ import annotations
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
NOISE_SAMPLES_DIR = ROOT / "noise_samples"

_AUDIO_EXTS = (".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3")

# Cache: (mode, fs) -> list of unit-RMS mono arrays at fs.
_cache: dict[tuple[str, int], list[np.ndarray]] = {}


def mode_dir(mode: str) -> Path:
    return NOISE_SAMPLES_DIR / mode


def _list_clips(mode: str) -> list[Path]:
    d = mode_dir(mode)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir()
                  if p.suffix.lower() in _AUDIO_EXTS and p.is_file())


def available(mode: str) -> bool:
    """True if at least one usable clip exists for `mode`."""
    return len(_list_clips(mode)) > 0


def _load_clips(mode: str, fs: int) -> list[np.ndarray]:
    key = (mode, fs)
    if key in _cache:
        return _cache[key]
    import soundfile as sf

    clips: list[np.ndarray] = []
    for path in _list_clips(mode):
        audio, src_fs = sf.read(path, dtype="float64", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if src_fs != fs:
            g = gcd(int(src_fs), int(fs))
            audio = resample_poly(audio, fs // g, int(src_fs) // g)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms > 0 and np.isfinite(rms):
            clips.append((audio / rms).astype(np.float64))
    _cache[key] = clips
    return clips


def generate_recorded(mode: str, n_samples: int,
                      rng: np.random.Generator, fs: int) -> np.ndarray:
    """Unit-RMS `n_samples` clip drawn from a random recording for `mode`.

    Raises FileNotFoundError if no clips are available (callers should gate on
    `available(mode)` and fall back to synthetic).
    """
    clips = _load_clips(mode, fs)
    if not clips:
        raise FileNotFoundError(
            f"No recorded noise clips for mode '{mode}' in {mode_dir(mode)}")
    clip = clips[int(rng.integers(0, len(clips)))]
    if len(clip) >= n_samples:
        start = int(rng.integers(0, len(clip) - n_samples + 1))
        out = clip[start:start + n_samples]
    else:
        reps = (n_samples // len(clip)) + 1
        out = np.tile(clip, reps)[:n_samples]
    # Re-normalize the slice itself so per-segment RMS is exactly unit.
    rms = float(np.sqrt(np.mean(out ** 2)) + 1e-12)
    return (out / rms).astype(np.float64)
