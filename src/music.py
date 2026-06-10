"""Music-clip loading shared by the music-aware tests, the test bench and the app.

`load_music_clip` returns a mono clip at `target_fs`, exactly `duration_sec`
long (center-cropped if the file is longer, looped if shorter), scaled to
`target_rms` so it sits at a known level against the unit-RMS noise.
"""
from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_music_clip(path: Path, target_fs: int, duration_sec: float,
                    target_rms: float = 1.0) -> np.ndarray:
    audio, fs = sf.read(path, dtype="float64", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if fs != target_fs:
        g = gcd(int(fs), int(target_fs))
        audio = resample_poly(audio, target_fs // g, int(fs) // g)
    n_target = int(duration_sec * target_fs)
    if len(audio) >= n_target:
        start = (len(audio) - n_target) // 2
        audio = audio[start:start + n_target]
    else:
        reps = (n_target // len(audio)) + 1
        audio = np.tile(audio, reps)[:n_target]
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms > 0:
        audio = audio * (target_rms / rms)
    return audio.astype(np.float64)
