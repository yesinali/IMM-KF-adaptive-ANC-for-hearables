"""Phase A sanity check #2: build a full scenario and visualize it.

Plots the mode trajectory, source noise waveform, primary signal,
and per-mode RMS levels. Also dumps a WAV of the source noise so the
user can listen to it.

Run from project root:
    python -m scripts.02_inspect_scenario
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src import scenario as sc
from src import utils as ut


def main() -> None:
    rng = np.random.default_rng(seed=7)
    s = sc.build_scenario(rng=rng)
    t = np.arange(len(s.n_source)) / s.fs

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t, s.n_source, lw=0.5)
    axes[0].set_ylabel("n(k)")
    axes[0].set_title("Source noise (mode-conditioned)")

    axes[1].plot(t, s.d, lw=0.5, color="C1")
    axes[1].set_ylabel("d(k)")
    axes[1].set_title("Primary signal d(k) = P(z) * n(k)")

    axes[2].plot(t, s.x_filt, lw=0.5, color="C2")
    axes[2].set_ylabel("xf(k)")
    axes[2].set_title("Filtered reference xf(k) = S(z) * x(k)")

    axes[3].step(t, s.mode_labels, where="post", color="k")
    axes[3].set_yticks(range(cfg.N_MODES))
    axes[3].set_yticklabels(cfg.MODE_NAMES)
    axes[3].set_ylabel("active mode")
    axes[3].set_xlabel("time [s]")
    axes[3].set_title("Ground-truth mode trajectory")

    fig.tight_layout()
    out_png = ROOT / "figures" / "02_scenario.png"
    out_png.parent.mkdir(exist_ok=True)
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_png}")

    out_wav = ROOT / "figures" / "02_source_noise.wav"
    norm = s.n_source / (np.max(np.abs(s.n_source)) + 1e-12) * 0.9
    sf.write(out_wav, norm.astype(np.float32), s.fs)
    print(f"saved {out_wav}")

    # Per-segment RMS sanity print.
    print("\nPer-segment RMS / dBFS:")
    idx = 0
    for seg in s.segments:
        n = int(seg.duration_sec * s.fs)
        chunk = s.n_source[idx: idx + n]
        print(f"  {seg.mode:8s}  {seg.duration_sec:5.1f}s   "
              f"RMS={ut.rms(chunk):.3f}  dBFS={ut.db20(ut.rms(chunk)):+6.2f}")
        idx += n


if __name__ == "__main__":
    main()
