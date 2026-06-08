"""Phase D++: NEES / NIS consistency evaluation.

Tests whether each filter's reported covariance is calibrated against the
true estimation error and innovation residual. Under filter consistency:
    average NEES (per sample, averaged over time and over N runs) ≈ L
    average NIS (per sample) ≈ 1
both should lie inside the 95% chi-square confidence band.

Runs a small Monte Carlo (default N=10), each on a 16-second dynamic
mode-conditioned-plant scenario, and reports / plots the per-method
time-averaged NEES and NIS together with the chi^2 bounds.

Run:
    python -m scripts.07_consistency_test --runs 10
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src import scenario as sc
from src.filters import KalmanANCFilter
from src.imm import IMMKalmanANC
from src.anc import simulate_anc
from src.metrics import chi2_bounds, overall_nr_db


TRAJ = (
    sc.ScenarioSegment("quiet",   4.0),
    sc.ScenarioSegment("traffic", 4.0),
    sc.ScenarioSegment("wind",    4.0),
    sc.ScenarioSegment("babble",  4.0),
)


def _steady_state_mask(segments, fs, n_pts, decimate=1, settle_sec=1.0):
    """Boolean mask over a (possibly decimated) per-sample grid that KEEPS only
    samples at least `settle_sec` after each segment start. This excludes the
    post-switch transient — when the synthetic plant teleports to a new random
    IR the estimation error jumps while P has not yet re-inflated, producing a
    NEES spike that is an artifact of the abrupt switch, not steady-state
    miscalibration. The kept samples are the fair consistency test.
    """
    mask = np.zeros(n_pts, dtype=bool)
    idx = 0
    settle = int(settle_sec * fs)
    for seg in segments:
        n = int(round(seg.duration_sec * fs))
        a = (idx + settle) // decimate
        b = (idx + n + decimate - 1) // decimate
        mask[a:b] = True
        idx += n
    return mask


def _build_methods():
    quiet = next(p for p in cfg.MODE_PARAMS if p.name == "quiet")
    traffic = next(p for p in cfg.MODE_PARAMS if p.name == "traffic")
    wind = next(p for p in cfg.MODE_PARAMS if p.name == "wind")
    return {
        "KF quiet":   lambda: KalmanANCFilter(L=cfg.L, sigma_q2=quiet.sigma_q2,   sigma_r2=quiet.sigma_r2),
        "KF traffic": lambda: KalmanANCFilter(L=cfg.L, sigma_q2=traffic.sigma_q2, sigma_r2=traffic.sigma_r2),
        "KF wind":    lambda: KalmanANCFilter(L=cfg.L, sigma_q2=wind.sigma_q2,    sigma_r2=wind.sigma_r2),
        "IMM-KF":     lambda: IMMKalmanANC(L=cfg.L, likelihood_window=200),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    methods = _build_methods()
    nees_log = {name: [] for name in methods}     # list of per-run time means (full)
    nees_ss_log = {name: [] for name in methods}  # steady-state only (transients excluded)
    nis_log = {name: [] for name in methods}
    nis_ss_log = {name: [] for name in methods}
    nr_log = {name: [] for name in methods}

    master = np.random.default_rng(args.seed)
    seeds = master.integers(0, 2**31, size=args.runs)
    t_total = time.perf_counter()

    nees_decimate = 100
    for run_idx, seed in enumerate(seeds, 1):
        t_run = time.perf_counter()
        rng = np.random.default_rng(int(seed))
        s = sc.build_scenario(segments=TRAJ, rng=rng, mode_conditioned_plants=True)
        w_opt_per_mode = sc.wiener_weights_per_mode(s, L=cfg.L)
        w_opt_arr = sc.per_sample_wiener_array(s, w_opt_per_mode)

        for name, factory in methods.items():
            filt = factory()
            r = simulate_anc(s, filt, w_opt=w_opt_arr,
                             log_nees=True, log_nis=True,
                             nees_decimate=nees_decimate)
            # Full (raw) means: skip the first 10% to discard the initial transient.
            skip = int(0.10 * len(r["nees"]))
            nees_log[name].append(float(np.mean(r["nees"][skip:])))
            skip_nis = int(0.10 * len(r["nis"]))
            nis_log[name].append(float(np.mean(r["nis"][skip_nis:])))
            # Steady-state means: drop 1 s after EVERY segment switch (fair test).
            m_nees = _steady_state_mask(TRAJ, s.fs, len(r["nees"]), nees_decimate)
            m_nis = _steady_state_mask(TRAJ, s.fs, len(r["nis"]), 1)
            nees_ss_log[name].append(float(np.mean(r["nees"][m_nees])))
            nis_ss_log[name].append(float(np.mean(r["nis"][m_nis])))
            nr_log[name].append(overall_nr_db(s.d, r["e"]))

        print(f"  run {run_idx:2d}/{args.runs}  "
              f"({time.perf_counter()-t_run:.1f}s, "
              f"total {time.perf_counter()-t_total:.0f}s)")

    # --- Aggregate ---
    # NEES is decimated by 100, NIS is per-sample.
    n_per_run = int(cfg.FS * sum(seg.duration_sec for seg in TRAJ))
    n_kept_nees = int(0.90 * (n_per_run // 100))
    n_kept_nis = int(0.90 * n_per_run)
    nees_lower, nees_upper = chi2_bounds(n_kept_nees * args.runs, dof=cfg.L, confidence=0.95)
    nis_lower, nis_upper = chi2_bounds(n_kept_nis * args.runs, dof=1, confidence=0.95)

    print(f"\n=== Consistency summary (N={args.runs}, dof_NEES={cfg.L}, dof_NIS=1) ===")
    print(f"NEES 95% chi-square band: [{nees_lower:.3f}, {nees_upper:.3f}] "
          f"(expected mean = {cfg.L})")
    print(f"NIS  95% chi-square band: [{nis_lower:.4f}, {nis_upper:.4f}] "
          f"(expected mean = 1)")
    print()

    def _verdict(mean, lo, hi):
        return "UNDERCONF" if mean < lo else "OVERCONF" if mean > hi else "CONSISTENT"

    print("Two NEES columns: 'full' includes post-switch transients; 'steady'")
    print("drops 1 s after each plant switch. 'steady' is the fair consistency test.")
    print()
    hdr = (f"{'Method':<14s}  {'NEES full':>10s}  {'NEES steady':>11s}  "
           f"{'steady verdict':<14s}  {'NIS steady':>10s}  {'NIS verdict':<12s}  "
           f"{'NR [dB]':>8s}")
    print(hdr)
    print("-" * len(hdr))

    summary = {}
    for name in methods:
        nees_full = float(np.array(nees_log[name]).mean())
        nees_ss = float(np.array(nees_ss_log[name]).mean())
        nis_ss = float(np.array(nis_ss_log[name]).mean())
        nr_mean = float(np.array(nr_log[name]).mean())

        nees_v = _verdict(nees_ss, nees_lower, nees_upper)
        nis_v = _verdict(nis_ss, nis_lower, nis_upper)

        summary[name] = dict(nees_full=nees_full, nees_ss=nees_ss, nees_v=nees_v,
                             nis_ss=nis_ss, nis_v=nis_v, nr=nr_mean)
        print(f"  {name:<12s}  {nees_full:>10.2f}  {nees_ss:>11.2f}  "
              f"{nees_v:<14s}  {nis_ss:>10.4f}  {nis_v:<12s}  {nr_mean:>+8.2f}")

    # --- Plots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    names = list(methods)
    xs = np.arange(len(names))

    # Plot the steady-state (fair) consistency metrics.
    nees_means = [summary[n]["nees_ss"] for n in names]
    ax1.bar(xs, nees_means, color=["#888"]*3 + ["#1f77b4"])
    ax1.axhline(cfg.L, color="green", ls="-", lw=2, label=f"expected = L = {cfg.L}")
    ax1.axhspan(nees_lower, nees_upper, color="green", alpha=0.15,
                label=f"95% χ² band")
    ax1.set_xticks(xs); ax1.set_xticklabels(names, rotation=15, ha="right")
    ax1.set_ylabel("steady-state NEES (mean over runs)")
    ax1.set_title(f"NEES consistency (transients excluded), dof = L = {cfg.L}")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, axis="y", ls=":", alpha=0.4)
    for x, v in zip(xs, nees_means):
        ax1.text(x, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)

    nis_means = [summary[n]["nis_ss"] for n in names]
    ax2.bar(xs, nis_means, color=["#888"]*3 + ["#1f77b4"])
    ax2.axhline(1.0, color="green", ls="-", lw=2, label="expected = 1")
    ax2.axhspan(nis_lower, nis_upper, color="green", alpha=0.15,
                label="95% χ² band")
    ax2.set_xticks(xs); ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.set_ylabel("steady-state NIS (mean over runs)")
    ax2.set_title("NIS consistency (transients excluded), dof = 1")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, axis="y", ls=":", alpha=0.4)
    for x, v in zip(xs, nis_means):
        ax2.text(x, v + 0.1, f"{v:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    out = ROOT / "figures" / "07_consistency.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\nsaved {out}")

    # Persist raw arrays
    npz = ROOT / "figures" / "07_consistency.npz"
    np.savez(npz,
             method_names=np.array(names),
             nees_full=np.array([summary[n]["nees_full"] for n in names]),
             nees_means=np.array(nees_means),     # steady-state (fair)
             nis_means=np.array(nis_means),       # steady-state (fair)
             nr_means=np.array([summary[n]["nr"] for n in names]),
             nees_band=np.array([nees_lower, nees_upper]),
             nis_band=np.array([nis_lower, nis_upper]),
             L=cfg.L, n_runs=args.runs)
    print(f"saved raw numbers to {npz}")


if __name__ == "__main__":
    main()
