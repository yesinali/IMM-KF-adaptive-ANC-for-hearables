"""Phase C: run IMM-KF and compare against NLMS / fixed-Q KF baselines.

Plots:
  - mode trajectory (ground truth)
  - mode posteriors mu_j(k) from IMM
  - noise-reduction over time for all methods
  - misalignment over time for all methods
Prints overall NR, final misalignment, and mode-tracking accuracy.

Run:
    python -m scripts.04_imm_run
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src import scenario as sc
from src.filters import NLMSFilter, KalmanANCFilter
from src.imm import IMMKalmanANC
from src.anc import simulate_anc
from src.metrics import (
    noise_reduction_db_sliding, overall_nr_db, misalignment_db,
)


def main() -> None:
    rng = np.random.default_rng(seed=7)
    traj = (
        sc.ScenarioSegment("quiet",   8.0),
        sc.ScenarioSegment("traffic", 8.0),
        sc.ScenarioSegment("wind",    8.0),
        sc.ScenarioSegment("babble",  8.0),
    )
    s = sc.build_scenario(segments=traj, rng=rng)
    w_opt = sc.wiener_weights(s, L=cfg.L)
    print(f"scenario: {s.duration_sec:.1f}s, L={cfg.L}")

    quiet = next(p for p in cfg.MODE_PARAMS if p.name == "quiet")
    traffic = next(p for p in cfg.MODE_PARAMS if p.name == "traffic")
    wind = next(p for p in cfg.MODE_PARAMS if p.name == "wind")

    methods: dict[str, object] = {
        "NLMS µ=0.01": NLMSFilter(L=cfg.L, mu=0.01),
        "NLMS µ=0.10": NLMSFilter(L=cfg.L, mu=0.10),
        "KF (quiet-tuned)":
            KalmanANCFilter(L=cfg.L, sigma_q2=quiet.sigma_q2, sigma_r2=quiet.sigma_r2),
        "KF (traffic-tuned)":
            KalmanANCFilter(L=cfg.L, sigma_q2=traffic.sigma_q2, sigma_r2=traffic.sigma_r2),
        "KF (wind-tuned)":
            KalmanANCFilter(L=cfg.L, sigma_q2=wind.sigma_q2, sigma_r2=wind.sigma_r2),
        "IMM-KF (4 modes)": IMMKalmanANC(L=cfg.L),
    }

    results = {}
    for name, filt in methods.items():
        t0 = time.perf_counter()
        results[name] = simulate_anc(s, filt, w_opt=w_opt, log_mu=True)
        print(f"  ran {name:25s} in {time.perf_counter()-t0:.2f}s")

    # ---- Plots ----
    win = int(0.5 * s.fs)
    t_full = np.arange(len(s.d)) / s.fs
    t_nr = t_full[win - 1:]

    fig, axes = plt.subplots(4, 1, figsize=(13, 10.5), sharex=True)

    # (1) ground-truth modes
    axes[0].step(t_full, s.mode_labels, where="post", color="k")
    axes[0].set_yticks(range(cfg.N_MODES))
    axes[0].set_yticklabels(cfg.MODE_NAMES)
    axes[0].set_ylabel("true mode")
    axes[0].set_title("(a) Ground-truth mode trajectory")

    # (2) IMM mode posteriors
    imm_run = results["IMM-KF (4 modes)"]
    mu_hist = imm_run["mu_history"]
    for j, name in enumerate(cfg.MODE_NAMES):
        axes[1].plot(t_full, mu_hist[:, j], label=name, lw=0.8)
    axes[1].set_ylabel(r"$\mu_j(k)$")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("(b) IMM mode posteriors")
    axes[1].legend(loc="center right", ncol=2)
    axes[1].grid(True, ls=":", alpha=0.4)

    # (3) NR over time
    for name, run in results.items():
        nr = noise_reduction_db_sliding(s.d, run["e"], win)
        axes[2].plot(t_nr, nr, label=name, lw=0.8)
    axes[2].axhline(0, color="gray", lw=0.5)
    axes[2].set_ylabel("NR [dB]")
    axes[2].set_title("(c) Noise reduction (0.5 s sliding window)")
    axes[2].legend(loc="lower right", ncol=2, fontsize=8)
    axes[2].grid(True, ls=":", alpha=0.4)

    # (4) misalignment
    for name, run in results.items():
        axes[3].plot(t_full, misalignment_db(run["misalignment"]), label=name, lw=0.8)
    axes[3].set_ylabel("misalignment [dB]")
    axes[3].set_xlabel("time [s]")
    axes[3].set_title("(d) Normalized misalignment vs Wiener weights")
    axes[3].legend(loc="upper right", ncol=2, fontsize=8)
    axes[3].grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    out = ROOT / "figures" / "04_imm.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\nsaved {out}")

    # ---- Summary ----
    print("\nOverall noise reduction (full run):")
    for name, run in results.items():
        print(f"  {name:25s} : {overall_nr_db(s.d, run['e']):+6.2f} dB")

    print("\nFinal misalignment [dB] (last 10% of run, mean):")
    tail = int(0.9 * len(s.d))
    for name, run in results.items():
        m = run["misalignment"][tail:].mean()
        print(f"  {name:25s} : {10*np.log10(m + 1e-12):+6.2f} dB")

    # Mode-tracking accuracy (IMM argmax vs ground truth)
    pred = mu_hist.argmax(axis=1)
    acc = float((pred == s.mode_labels).mean())
    print(f"\nIMM mode-tracking accuracy (argmax mu_j == true mode): {acc*100:.1f}%")


if __name__ == "__main__":
    main()
