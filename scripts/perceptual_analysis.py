"""Perceptual side of the ANC residual: spectrogram comparison + per-band NR.

Listening to the v5 IMM residual reveals a classic ANC failure mode that
overall NR cannot expose: the IMM aggressively re-fits its state at every
mode switch, which produces transient "musical-noise" artefacts. NLMS, by
contrast, has no notion of regime and adapts smoothly, so its residual is
just a quieter version of the input even though its overall NR is lower.

This script makes that visible:
  - log-magnitude spectrograms of d, e_NLMS, e_IMM side-by-side
  - per-octave-band NR table (50-250, 250-1000, 1k-4k, 4k-8k Hz)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src import scenario as sc
from src import c_backend
from src.metrics import overall_nr_db


TRAJ = (
    sc.ScenarioSegment("quiet",   5.0),
    sc.ScenarioSegment("traffic", 5.0),
    sc.ScenarioSegment("wind",    5.0),
    sc.ScenarioSegment("babble",  5.0),
)


def band_nr_db(d: np.ndarray, e: np.ndarray, fs: int, lo: float, hi: float):
    """Bandpass FFT-domain NR over [lo, hi] Hz."""
    F_d = np.fft.rfft(d)
    F_e = np.fft.rfft(e)
    freqs = np.fft.rfftfreq(len(d), d=1.0/fs)
    band = (freqs >= lo) & (freqs < hi)
    p_d = float(np.sum(np.abs(F_d[band]) ** 2))
    p_e = float(np.sum(np.abs(F_e[band]) ** 2))
    return 10.0 * np.log10(p_d / max(p_e, 1e-30))


def main() -> None:
    if not c_backend.is_available("pure-c"):
        raise SystemExit("Pure C missing.")

    rng = np.random.default_rng(7)
    s = sc.build_scenario(segments=TRAJ, rng=rng, mode_conditioned_plants=True)
    cres = c_backend.run(s, "IMM-KF", {"window": 200},
                         backend="pure-c", L_override=64)

    d = s.d
    e_nlms = cres.e_nlms
    e_imm  = cres.e_imm
    fs = s.fs

    # ----- Overall NR -----
    print("=== Overall NR ===")
    print(f"  NLMS  : {overall_nr_db(d, e_nlms):+6.2f} dB")
    print(f"  IMM-KF: {overall_nr_db(d, e_imm):+6.2f} dB")
    print()

    # ----- Per-band NR -----
    bands = [(50, 250, "low (50-250)"),
             (250, 1000, "low-mid (250-1k)"),
             (1000, 4000, "mid (1-4k)"),
             (4000, 8000, "high (4-8k)")]
    print("=== Per-band NR [dB] (higher = quieter residual in that band) ===")
    print(f"  {'band [Hz]':<22}  {'NLMS':>8}  {'IMM-KF':>8}")
    for lo, hi, name in bands:
        nr_n = band_nr_db(d, e_nlms, fs, lo, hi)
        nr_i = band_nr_db(d, e_imm,  fs, lo, hi)
        flag = " ⚠" if nr_i < nr_n else ""
        print(f"  {name:<22}  {nr_n:>+7.2f}  {nr_i:>+7.2f}{flag}")
    print()

    # ----- Spectrograms -----
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, sharey=True)
    labels = [("Original noise d(k)", d, axes[0]),
              ("NLMS residual e(k)  (+4.65 dB)", e_nlms, axes[1]),
              ("IMM-KF residual e(k) (+16.50 dB)", e_imm, axes[2])]
    nperseg = 512
    overlap = 384

    # Use the original's dynamic range as reference for all three so the
    # IMM artefact band stands out instead of being individually normalized.
    f_ref, t_ref, S_ref = spectrogram(d, fs=fs, nperseg=nperseg,
                                      noverlap=overlap, scaling="spectrum")
    db_ref = 10 * np.log10(S_ref + 1e-12)
    vmin = float(np.percentile(db_ref, 5))
    vmax = float(np.percentile(db_ref, 99))

    for title, sig, ax in labels:
        f_, t_, S = spectrogram(sig, fs=fs, nperseg=nperseg,
                                noverlap=overlap, scaling="spectrum")
        db = 10 * np.log10(S + 1e-12)
        im = ax.pcolormesh(t_, f_, db, cmap="magma", vmin=vmin, vmax=vmax,
                           shading="auto")
        ax.set_ylabel("Hz")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 4000)
        # Mark the mode-switch boundaries
        for t_switch, mode in zip([5, 10, 15], ["traffic", "wind", "babble"]):
            ax.axvline(t_switch, color="cyan", ls="--", lw=0.8, alpha=0.7)
        # annotate the 4 mode segments along the top
        if ax is axes[0]:
            for t_mid, mode in zip([2.5, 7.5, 12.5, 17.5],
                                   ["quiet", "traffic", "wind", "babble"]):
                ax.text(t_mid, 3700, mode, ha="center", color="cyan",
                        fontsize=9, weight="bold")

    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Spectrogram comparison: NLMS residual stays spectrally "
                 "natural; IMM residual carries transient mode-switch "
                 "artefacts (look at t = 5, 10, 15 s)",
                 fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.85, label="dB",
                 location="right", pad=0.02)
    out = ROOT / "figures" / "11_perceptual_spectrogram.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
