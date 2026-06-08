"""Re-render the NEES/NIS consistency bar charts on a log scale.

The linear-scale plot saved by `07_consistency_test.py` is dominated by the
fixed-Q KF NEES values (10^7..10^9), which collapses every interesting bar
to invisible. A log y-axis is the only honest way to display 5 orders of
magnitude side-by-side.

Loads `figures/07_consistency.npz` (saved by the consistency test) and
re-renders to `figures/07_consistency_log.png`.

Run:
    python -m scripts.07b_consistency_replot
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    npz = np.load(ROOT / "figures" / "07_consistency.npz")
    names = list(npz["method_names"])
    nees = npz["nees_means"]
    nis = npz["nis_means"]
    nees_band = npz["nees_band"]
    nis_band = npz["nis_band"]
    L = int(npz["L"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    xs = np.arange(len(names))
    colors = ["#888"] * (len(names) - 1) + ["#1f77b4"]

    # NEES (log scale)
    bars1 = ax1.bar(xs, nees, color=colors)
    ax1.set_yscale("log")
    ax1.axhline(L, color="green", ls="-", lw=2, label=f"expected = L = {L}")
    ax1.axhspan(nees_band[0], nees_band[1], color="green", alpha=0.15,
                label="95% χ² band")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.set_ylabel("time-averaged NEES (log scale)")
    ax1.set_title(f"NEES consistency, dof = L = {L}")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, axis="y", which="both", ls=":", alpha=0.4)
    for x, v in zip(xs, nees):
        ax1.text(x, v * 1.5, f"{v:.2g}", ha="center", fontsize=8)
    # Headroom so the labels above the tallest bar do not get clipped
    top = max(nees.max() * 6.0, L * 2)
    ax1.set_ylim(top=top)

    # NIS (log scale — much narrower range but log still helps)
    bars2 = ax2.bar(xs, nis, color=colors)
    ax2.set_yscale("log")
    ax2.axhline(1.0, color="green", ls="-", lw=2, label="expected = 1")
    ax2.axhspan(nis_band[0], nis_band[1], color="green", alpha=0.15,
                label="95% χ² band")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.set_ylabel("time-averaged NIS (log scale)")
    ax2.set_title("NIS consistency, dof = 1")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, axis="y", which="both", ls=":", alpha=0.4)
    for x, v in zip(xs, nis):
        ax2.text(x, v * 1.3, f"{v:.3g}", ha="center", fontsize=8)
    ax2.set_ylim(top=max(nis.max() * 4.0, 4.0))

    fig.tight_layout()
    out = ROOT / "figures" / "07_consistency_log.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
