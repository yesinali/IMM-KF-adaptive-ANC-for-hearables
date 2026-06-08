"""Phase B: run NLMS (two step-sizes) and a single-mode KF on the scenario,
compare against the Wiener solution.

Uses a shortened scenario (8 s per segment x 4 segments = 32 s) so the
Python-level KF loop finishes in well under a minute. Phase D will move
heavy Monte Carlo to a larger budget.

Run:
    python -m scripts.03_baseline_run
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
from src.anc import simulate_anc
from src.metrics import (
    noise_reduction_db_sliding, overall_nr_db, misalignment_db,
)


def main() -> None:
    rng = np.random.default_rng(seed=7)
    short_traj = (
        sc.ScenarioSegment("quiet",   8.0),
        sc.ScenarioSegment("traffic", 8.0),
        sc.ScenarioSegment("wind",    8.0),
        sc.ScenarioSegment("babble",  8.0),
    )
    s = sc.build_scenario(segments=short_traj, rng=rng)
    print(f"scenario: {s.duration_sec:.1f}s, {len(s.d)} samples @ {s.fs} Hz, L={cfg.L}")

    t0 = time.perf_counter()
    w_opt = sc.wiener_weights(s, L=cfg.L)
    print(f"  Wiener solve: {time.perf_counter() - t0:.2f}s")

    # Use the calibrated traffic-mode (Q, R) pair from config as the
    # representative fixed-Q KF baseline.
    traffic = next(p for p in cfg.MODE_PARAMS if p.name == "traffic")
    runs = {
        "NLMS µ=0.01": NLMSFilter(L=cfg.L, mu=0.01),
        "NLMS µ=0.10": NLMSFilter(L=cfg.L, mu=0.10),
        "KF fixed-Q (traffic-tuned)":
            KalmanANCFilter(L=cfg.L,
                            sigma_q2=traffic.sigma_q2,
                            sigma_r2=traffic.sigma_r2),
    }

    results = {}
    for name, filt in runs.items():
        t0 = time.perf_counter()
        results[name] = simulate_anc(s, filt, w_opt=w_opt)
        print(f"  ran {name:30s} in {time.perf_counter() - t0:.2f}s")

    # ---- Plots ----
    win = int(0.5 * s.fs)                           # 0.5 s NR window
    t_full = np.arange(len(s.d)) / s.fs
    t_nr = t_full[win - 1:]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)

    axes[0].step(t_full, s.mode_labels, where="post", color="k")
    axes[0].set_yticks(range(cfg.N_MODES))
    axes[0].set_yticklabels(cfg.MODE_NAMES)
    axes[0].set_ylabel("mode")
    axes[0].set_title("Ground-truth mode trajectory")

    for name, run in results.items():
        nr = noise_reduction_db_sliding(s.d, run["e"], win)
        axes[1].plot(t_nr, nr, label=name, lw=0.8)
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_ylabel("NR [dB]")
    axes[1].set_title("Noise reduction (0.5 s sliding window) — higher is better")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, ls=":", alpha=0.5)

    for name, run in results.items():
        axes[2].plot(t_full, misalignment_db(run["misalignment"]),
                     label=name, lw=0.8)
    axes[2].set_ylabel("misalignment [dB]")
    axes[2].set_xlabel("time [s]")
    axes[2].set_title("Normalized misalignment vs Wiener weights — lower is better")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, ls=":", alpha=0.5)

    fig.tight_layout()
    out = ROOT / "figures" / "03_baselines.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\nsaved {out}")

    # ---- Summary table ----
    print("\nOverall noise reduction (full run):")
    for name, run in results.items():
        print(f"  {name:30s} : {overall_nr_db(s.d, run['e']):+6.2f} dB")

    print("\nFinal misalignment [dB] (last 10% of run, mean):")
    tail_start = int(0.9 * len(s.d))
    for name, run in results.items():
        m = run["misalignment"][tail_start:].mean()
        print(f"  {name:30s} : {10 * np.log10(m + 1e-12):+6.2f} dB")


if __name__ == "__main__":
    main()
