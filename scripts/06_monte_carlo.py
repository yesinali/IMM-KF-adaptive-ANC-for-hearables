"""Phase D: Monte Carlo evaluation across N independent runs.

Each run draws fresh (P, S) IRs per mode and fresh noise realizations.
We log overall NR and per-mode mean misalignment for every method, then
report mean ± std and produce summary bar charts.

Run:
    python -m scripts.06_monte_carlo --runs 20
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
from src.filters import NLMSFilter, KalmanANCFilter
from src.imm import IMMKalmanANC
from src.anc import simulate_anc
from src.metrics import overall_nr_db


SHORT_TRAJ = (
    sc.ScenarioSegment("quiet",   4.0),
    sc.ScenarioSegment("traffic", 4.0),
    sc.ScenarioSegment("wind",    4.0),
    sc.ScenarioSegment("babble",  4.0),
    sc.ScenarioSegment("quiet",   4.0),
)


def _build_methods() -> dict:
    quiet = next(p for p in cfg.MODE_PARAMS if p.name == "quiet")
    traffic = next(p for p in cfg.MODE_PARAMS if p.name == "traffic")
    wind = next(p for p in cfg.MODE_PARAMS if p.name == "wind")
    return {
        "NLMS µ=0.01": lambda: NLMSFilter(L=cfg.L, mu=0.01),
        "NLMS µ=0.10": lambda: NLMSFilter(L=cfg.L, mu=0.10),
        "KF quiet":   lambda: KalmanANCFilter(L=cfg.L, sigma_q2=quiet.sigma_q2,   sigma_r2=quiet.sigma_r2),
        "KF traffic": lambda: KalmanANCFilter(L=cfg.L, sigma_q2=traffic.sigma_q2, sigma_r2=traffic.sigma_r2),
        "KF wind":    lambda: KalmanANCFilter(L=cfg.L, sigma_q2=wind.sigma_q2,    sigma_r2=wind.sigma_r2),
        "IMM-KF":     lambda: IMMKalmanANC(L=cfg.L, likelihood_window=200),
    }


def _per_mode_mean_misalign(misalign: np.ndarray,
                            mode_labels: np.ndarray,
                            fs: int,
                            skip_sec: float = 1.0) -> dict[str, float]:
    """Mean misalignment per mode, ignoring the first `skip_sec` after each entry."""
    skip = int(skip_sec * fs)
    out = {}
    for mode_idx, mode_name in enumerate(cfg.MODE_NAMES):
        mask = mode_labels == mode_idx
        keep = np.zeros_like(mask)
        in_seg = False
        start = 0
        for k in range(len(mask)):
            if mask[k] and not in_seg:
                in_seg, start = True, k
            elif not mask[k] and in_seg:
                in_seg = False
                keep[start + skip: k] = True
        if in_seg:
            keep[start + skip:] = True
        out[mode_name] = float(misalign[keep].mean()) if keep.any() else np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20, help="number of Monte Carlo runs")
    ap.add_argument("--seed", type=int, default=2026, help="master seed")
    args = ap.parse_args()

    methods = _build_methods()
    method_names = list(methods)
    # Storage: per-method NR list, per-method per-mode misalign list
    nr_log = {m: [] for m in method_names}
    misalign_log = {m: {mode: [] for mode in cfg.MODE_NAMES} for m in method_names}
    imm_acc_log = []

    master = np.random.default_rng(args.seed)
    seeds = master.integers(0, 2**31, size=args.runs)
    total_t0 = time.perf_counter()

    for run_idx, seed in enumerate(seeds, 1):
        run_t0 = time.perf_counter()
        rng = np.random.default_rng(int(seed))
        s = sc.build_scenario(segments=SHORT_TRAJ, rng=rng, mode_conditioned_plants=True)
        w_opt_per_mode = sc.wiener_weights_per_mode(s, L=cfg.L)
        w_opt_arr = sc.per_sample_wiener_array(s, w_opt_per_mode)

        for name, factory in methods.items():
            filt = factory()
            r = simulate_anc(s, filt, w_opt=w_opt_arr,
                             log_mu=(name == "IMM-KF"))
            nr_log[name].append(overall_nr_db(s.d, r["e"]))
            per_mode = _per_mode_mean_misalign(r["misalignment"], s.mode_labels, s.fs)
            for mode_name, val in per_mode.items():
                misalign_log[name][mode_name].append(val)
            if name == "IMM-KF":
                pred = r["mu_history"].argmax(axis=1)
                imm_acc_log.append(float((pred == s.mode_labels).mean()))

        elapsed = time.perf_counter() - run_t0
        cum = time.perf_counter() - total_t0
        eta = cum / run_idx * (args.runs - run_idx)
        print(f"  run {run_idx:2d}/{args.runs}  ({elapsed:5.1f}s, total {cum:.0f}s, eta {eta:.0f}s)")

    # ---- Aggregate ----
    print(f"\n=== Monte Carlo summary over N={args.runs} runs ===\n")
    print(f"{'Method':<14s}  {'NR mean ± std':<18s}")
    nr_means, nr_stds = {}, {}
    for name in method_names:
        arr = np.array(nr_log[name])
        nr_means[name] = arr.mean()
        nr_stds[name] = arr.std()
        print(f"  {name:<12s}  {arr.mean():+6.2f} ± {arr.std():4.2f} dB")

    print(f"\nIMM mode-tracking accuracy: "
          f"{100*np.mean(imm_acc_log):5.1f} ± {100*np.std(imm_acc_log):4.1f}%\n")

    print("Per-mode mean misalignment [dB]:")
    print(f"{'Method':<14s}  " + "  ".join(f"{m:<10s}" for m in cfg.MODE_NAMES))
    misalign_means = {m: {} for m in method_names}
    for name in method_names:
        cells = []
        for mode in cfg.MODE_NAMES:
            arr = np.array(misalign_log[name][mode])
            mean_db = 10 * np.log10(arr.mean() + 1e-12)
            misalign_means[name][mode] = mean_db
            cells.append(f"{mean_db:+6.2f}    ")
        print(f"  {name:<12s}  " + "  ".join(cells))

    # ---- Plot: NR bar chart with error bars + per-mode misalign heatmap ----
    fig, (ax_nr, ax_heat) = plt.subplots(1, 2, figsize=(14, 4.8))
    xs = np.arange(len(method_names))
    means = [nr_means[n] for n in method_names]
    stds = [nr_stds[n] for n in method_names]
    bars = ax_nr.bar(xs, means, yerr=stds, capsize=4,
                     color=["#888"]*5 + ["#1f77b4"])
    ax_nr.set_xticks(xs)
    ax_nr.set_xticklabels(method_names, rotation=20, ha="right")
    ax_nr.set_ylabel("NR [dB]")
    ax_nr.set_title(f"Overall noise reduction (N={args.runs}, mean ± std)")
    ax_nr.grid(True, axis="y", ls=":", alpha=0.5)
    for b, m in zip(bars, means):
        ax_nr.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                   f"{m:+.2f}", ha="center", fontsize=8)

    heat = np.array([[misalign_means[n][m] for m in cfg.MODE_NAMES]
                     for n in method_names])
    im = ax_heat.imshow(heat, aspect="auto", cmap="RdYlGn_r", vmin=-25, vmax=5)
    ax_heat.set_xticks(range(cfg.N_MODES))
    ax_heat.set_xticklabels(cfg.MODE_NAMES)
    ax_heat.set_yticks(range(len(method_names)))
    ax_heat.set_yticklabels(method_names)
    ax_heat.set_title("Per-mode mean misalignment [dB] (lower better)")
    for i in range(len(method_names)):
        for j in range(cfg.N_MODES):
            ax_heat.text(j, i, f"{heat[i,j]:+.1f}",
                         ha="center", va="center", fontsize=8,
                         color="white" if abs(heat[i,j]) > 10 else "black")
    fig.colorbar(im, ax=ax_heat, shrink=0.85, label="dB")

    fig.tight_layout()
    out = ROOT / "figures" / "06_monte_carlo.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\nsaved {out}")

    # Persist raw numbers too.
    npz = ROOT / "figures" / "06_monte_carlo.npz"
    np.savez(npz,
             method_names=np.array(method_names),
             mode_names=np.array(cfg.MODE_NAMES),
             nr_means=np.array([nr_means[n] for n in method_names]),
             nr_stds=np.array([nr_stds[n] for n in method_names]),
             misalign_heatmap_db=heat,
             imm_accuracy=np.array(imm_acc_log))
    print(f"saved raw numbers to {npz}")


if __name__ == "__main__":
    main()
