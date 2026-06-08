"""Phase A sanity check #1: visualize random primary and secondary paths.

Run from project root:
    python -m scripts.01_inspect_paths
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src import paths as pth


def main() -> None:
    rng = np.random.default_rng(seed=42)
    p_ir = pth.primary_path(rng)
    s_ir = pth.secondary_path(rng)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6))

    axes[0, 0].stem(p_ir, basefmt=" ")
    axes[0, 0].set_title(f"Primary path P(z), {cfg.P_LEN} taps")
    axes[0, 0].set_xlabel("tap"); axes[0, 0].set_ylabel("amplitude")

    axes[0, 1].stem(s_ir, basefmt=" ")
    axes[0, 1].set_title(f"Secondary path S(z), {cfg.S_LEN} taps")
    axes[0, 1].set_xlabel("tap"); axes[0, 1].set_ylabel("amplitude")

    for ir, ax, name in [(p_ir, axes[1, 0], "P(z)"), (s_ir, axes[1, 1], "S(z)")]:
        w, h = freqz(ir, worN=2048, fs=cfg.FS)
        ax.semilogx(w, 20 * np.log10(np.abs(h) + 1e-12))
        ax.set_title(f"{name} magnitude response")
        ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("|H| [dB]")
        ax.grid(True, which="both", ls=":", alpha=0.5)

    fig.tight_layout()
    out = ROOT / "figures" / "01_paths.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
