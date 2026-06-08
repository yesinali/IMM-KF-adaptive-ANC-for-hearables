"""Phase C-dynamic: run IMM-KF and baselines on a *mode-conditioned plant*
scenario, where every acoustic mode has its own random (P, S) IRs. This
exposes the regime-switching benefit IMM is supposed to capture --- each
mode's optimal Wiener controller is now different, and the filter must
re-adapt at every segment boundary.

Misalignment is measured against the *current* mode's Wiener reference
(per-sample target), not a global one.

Run:
    python -m scripts.05_dynamic_imm
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
    rng = np.random.default_rng(seed=11)
    traj = (
        sc.ScenarioSegment("quiet",   6.0),
        sc.ScenarioSegment("traffic", 6.0),
        sc.ScenarioSegment("wind",    6.0),
        sc.ScenarioSegment("babble",  6.0),
        sc.ScenarioSegment("quiet",   6.0),
    )
    s = sc.build_scenario(segments=traj, rng=rng, mode_conditioned_plants=True)
    print(f"scenario: {s.duration_sec:.1f}s, L={cfg.L}, "
          f"mode-conditioned plants={s.mode_conditioned}")

    t0 = time.perf_counter()
    w_opt_per_mode = sc.wiener_weights_per_mode(s, L=cfg.L)
    w_opt_arr = sc.per_sample_wiener_array(s, w_opt_per_mode)
    print(f"  Wiener (per-mode) solve: {time.perf_counter()-t0:.2f}s")
    for m, w in w_opt_per_mode.items():
        print(f"    {m:8s} ||w_opt||={np.linalg.norm(w):.4f}")

    quiet_p = next(p for p in cfg.MODE_PARAMS if p.name == "quiet")
    traffic_p = next(p for p in cfg.MODE_PARAMS if p.name == "traffic")
    wind_p = next(p for p in cfg.MODE_PARAMS if p.name == "wind")

    methods: dict[str, object] = {
        "NLMS µ=0.01": NLMSFilter(L=cfg.L, mu=0.01),
        "NLMS µ=0.10": NLMSFilter(L=cfg.L, mu=0.10),
        "KF quiet-tuned":
            KalmanANCFilter(L=cfg.L, sigma_q2=quiet_p.sigma_q2, sigma_r2=quiet_p.sigma_r2),
        "KF traffic-tuned":
            KalmanANCFilter(L=cfg.L, sigma_q2=traffic_p.sigma_q2, sigma_r2=traffic_p.sigma_r2),
        "KF wind-tuned":
            KalmanANCFilter(L=cfg.L, sigma_q2=wind_p.sigma_q2, sigma_r2=wind_p.sigma_r2),
        # 200-sample (~12 ms) likelihood smoothing helps the posterior commit.
        "IMM-KF (4 modes)": IMMKalmanANC(L=cfg.L, likelihood_window=200),
    }

    results = {}
    for name, filt in methods.items():
        t0 = time.perf_counter()
        results[name] = simulate_anc(s, filt, w_opt=w_opt_arr, log_mu=True)
        print(f"  ran {name:20s} in {time.perf_counter()-t0:.2f}s")

    # ---- Plots ----
    win = int(0.5 * s.fs)
    t_full = np.arange(len(s.d)) / s.fs
    t_nr = t_full[win - 1:]

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

    axes[0].step(t_full, s.mode_labels, where="post", color="k")
    axes[0].set_yticks(range(cfg.N_MODES))
    axes[0].set_yticklabels(cfg.MODE_NAMES)
    axes[0].set_ylabel("true mode")
    axes[0].set_title("(a) Ground-truth mode trajectory — each segment also switches the plant")

    imm_mu = results["IMM-KF (4 modes)"]["mu_history"]
    for j, name in enumerate(cfg.MODE_NAMES):
        axes[1].plot(t_full, imm_mu[:, j], label=name, lw=0.8)
    axes[1].set_ylabel(r"$\mu_j(k)$")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("(b) IMM mode posteriors")
    axes[1].legend(loc="center right", ncol=2, fontsize=8)
    axes[1].grid(True, ls=":", alpha=0.4)

    for name, run in results.items():
        nr = noise_reduction_db_sliding(s.d, run["e"], win)
        axes[2].plot(t_nr, nr, label=name, lw=0.8)
    axes[2].axhline(0, color="gray", lw=0.5)
    axes[2].set_ylabel("NR [dB]")
    axes[2].set_title("(c) Noise reduction (0.5 s sliding window)")
    axes[2].legend(loc="lower right", ncol=2, fontsize=8)
    axes[2].grid(True, ls=":", alpha=0.4)

    for name, run in results.items():
        axes[3].plot(t_full, misalignment_db(run["misalignment"]), label=name, lw=0.8)
    axes[3].set_ylabel("misalignment [dB]")
    axes[3].set_xlabel("time [s]")
    axes[3].set_title("(d) Misalignment vs current mode's Wiener — segment boundaries shaded")
    # Shade segment boundaries
    idx = 0
    for seg in s.segments[:-1]:
        idx += int(seg.duration_sec * s.fs)
        axes[3].axvline(idx / s.fs, color="gray", lw=0.5, alpha=0.5)
    axes[3].legend(loc="upper right", ncol=2, fontsize=8)
    axes[3].grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    out = ROOT / "figures" / "05_dynamic_imm.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\nsaved {out}")

    # ---- Summary ----
    print("\nOverall noise reduction (full run):")
    for name, run in results.items():
        print(f"  {name:20s} : {overall_nr_db(s.d, run['e']):+6.2f} dB")

    print("\nPer-mode mean misalignment [dB] (only that mode's samples, "
          "skipping first 1 s after entry):")
    skip = int(1.0 * s.fs)
    for name, run in results.items():
        line = [f"  {name:20s}"]
        for mode_idx, mode_name in enumerate(cfg.MODE_NAMES):
            mask = s.mode_labels == mode_idx
            # Find each contiguous segment of this mode and drop its first `skip` samples.
            keep = np.zeros_like(mask)
            in_seg = False
            seg_start = 0
            for k in range(len(mask)):
                if mask[k] and not in_seg:
                    in_seg = True
                    seg_start = k
                elif not mask[k] and in_seg:
                    in_seg = False
                    keep[seg_start + skip: k] = True
            if in_seg:
                keep[seg_start + skip:] = True
            if keep.any():
                m = run["misalignment"][keep].mean()
                line.append(f"{mode_name}={10*np.log10(m+1e-12):+6.1f}")
            else:
                line.append(f"{mode_name}=  n/a ")
        print("  ".join(line))

    pred = imm_mu.argmax(axis=1)
    acc = float((pred == s.mode_labels).mean())
    print(f"\nIMM mode-tracking accuracy: {acc*100:.1f}%")
    # Confusion summary
    print("Confusion (rows=truth, cols=pred):")
    cm = np.zeros((cfg.N_MODES, cfg.N_MODES), dtype=int)
    for t_idx, p_idx in zip(s.mode_labels, pred):
        cm[t_idx, p_idx] += 1
    header = "        " + " ".join(f"{n:>8s}" for n in cfg.MODE_NAMES)
    print(header)
    for i, name in enumerate(cfg.MODE_NAMES):
        row = " ".join(f"{cm[i, j]:8d}" for j in range(cfg.N_MODES))
        print(f"  {name:>6s} {row}")


if __name__ == "__main__":
    main()
