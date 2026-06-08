"""Run the C binaries (pure + BLAS), reread their residuals, time NumPy on
the same scenario, then report:
  - numeric equivalence (max-abs error against NumPy reference)
  - per-filter wall time and per-sample microseconds
  - overall NR [dB] (all three implementations should match to machine eps).

Generates a side-by-side speed table and a comparison figure.

Run after `python -m scripts.c_port_dump`:
    python -m scripts.c_port_compare
"""
from __future__ import annotations
import argparse
import json
import subprocess
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


def run_c_binary(binary: Path, in_bin: Path, out_bin: Path) -> dict:
    if not binary.exists():
        raise FileNotFoundError(f"missing C binary: {binary}")
    proc = subprocess.run(
        [str(binary), str(in_bin), str(out_bin)],
        capture_output=True, text=True, check=True,
    )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"could not parse JSON from {binary}: {exc}\nstdout was:\n{proc.stdout}"
        )


def read_c_residuals(out_bin: Path, N_expected: int) -> dict[str, np.ndarray]:
    with open(out_bin, "rb") as fp:
        N = int(np.frombuffer(fp.read(4), dtype=np.int32)[0])
        if N != N_expected:
            raise ValueError(f"N mismatch: file says {N}, expected {N_expected}")
        e_nlms = np.frombuffer(fp.read(N * 8), dtype=np.float64).copy()
        e_kf   = np.frombuffer(fp.read(N * 8), dtype=np.float64).copy()
        e_imm  = np.frombuffer(fp.read(N * 8), dtype=np.float64).copy()
    return {"NLMS": e_nlms, "KF": e_kf, "IMM": e_imm}


def run_python_reference(scenario, mu_nlms: float, kf_params) -> dict:
    """Time the three NumPy filters on the same scenario."""
    out = {}

    filt = NLMSFilter(L=cfg.L, mu=mu_nlms)
    t0 = time.perf_counter()
    r = simulate_anc(scenario, filt)
    out["NLMS"] = {"e": r["e"], "wall_sec": time.perf_counter() - t0}

    filt = KalmanANCFilter(L=cfg.L,
                           sigma_q2=kf_params.sigma_q2,
                           sigma_r2=kf_params.sigma_r2)
    t0 = time.perf_counter()
    r = simulate_anc(scenario, filt)
    out["KF"] = {"e": r["e"], "wall_sec": time.perf_counter() - t0}

    filt = IMMKalmanANC(L=cfg.L, likelihood_window=200)
    t0 = time.perf_counter()
    r = simulate_anc(scenario, filt)
    out["IMM"] = {"e": r["e"], "wall_sec": time.perf_counter() - t0}

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cdir", type=str, default=str(ROOT / "c_imm"))
    ap.add_argument("--bin",  type=str, default=str(ROOT / "c_imm" / "scenario.bin"))
    ap.add_argument("--seed", type=int, default=2026,
                    help="must match the seed used in c_port_dump for the side"
                         "car npz to line up")
    ap.add_argument("--skip-blas", action="store_true",
                    help="skip the OpenBLAS binary (use if not yet built)")
    args = ap.parse_args()

    cdir = Path(args.cdir)
    bin_path = Path(args.bin)
    npz_path = bin_path.with_suffix(".npz")
    if not bin_path.exists() or not npz_path.exists():
        raise SystemExit("Run scripts/c_port_dump.py first.")

    sidecar = np.load(npz_path)
    xf = sidecar["xf"]; d = sidecar["d"]
    N = len(d)
    FS = int(sidecar["fs"])
    print(f"scenario: N={N} samples ({N/FS:.2f} s)")

    # --- Run pure-C binary ---
    out_pure = cdir / "out_pure.bin"
    timing_pure = run_c_binary(cdir / "imm_pure.exe", bin_path, out_pure)
    e_pure = read_c_residuals(out_pure, N)

    # --- Run BLAS binary (optional) ---
    timing_blas, e_blas = None, None
    if not args.skip_blas:
        blas_bin = cdir / "imm_blas.exe"
        if blas_bin.exists():
            out_blas = cdir / "out_blas.bin"
            timing_blas = run_c_binary(blas_bin, bin_path, out_blas)
            e_blas = read_c_residuals(out_blas, N)
        else:
            print(f"note: {blas_bin} not built yet -- skipping BLAS variant")

    # --- NumPy reference: rebuild the scenario from the same seed ---
    seg_sec = (N / FS) / 4.0
    traj = (
        sc.ScenarioSegment("quiet",   seg_sec),
        sc.ScenarioSegment("traffic", seg_sec),
        sc.ScenarioSegment("wind",    seg_sec),
        sc.ScenarioSegment("babble",  seg_sec),
    )
    rng = np.random.default_rng(args.seed)
    scen = sc.build_scenario(segments=traj, rng=rng,
                             mode_conditioned_plants=True)
    quiet = next(p for p in cfg.MODE_PARAMS if p.name == "quiet")
    py = run_python_reference(scen, mu_nlms=0.10, kf_params=quiet)

    # Sanity: the dumped xf/d must equal the scenario rebuilt here
    assert np.allclose(scen.x_filt, xf), \
        "xf mismatch: dump and rebuild diverge -- check seed/config"
    assert np.allclose(scen.d, d), \
        "d mismatch: dump and rebuild diverge -- check seed/config"

    # --- Numeric equivalence ---
    print("\n--- numeric equivalence vs NumPy reference (max |delta|) ---")
    print(f"{'filter':<8}  {'pure-C':>11}  {'OpenBLAS':>11}")
    diff_table = {}
    for name in ("NLMS", "KF", "IMM"):
        ref = py[name]["e"]
        d_pure = float(np.max(np.abs(e_pure[name] - ref)))
        d_blas = float(np.max(np.abs(e_blas[name] - ref))) if e_blas else float("nan")
        diff_table[name] = (d_pure, d_blas)
        print(f"  {name:<6}  {d_pure:>11.3e}  {d_blas:>11.3e}")

    # --- NR (should be identical across implementations modulo rounding) ---
    print("\n--- overall NR [dB] (sanity, identical math) ---")
    print(f"{'filter':<8}  {'Python':>9}  {'pure-C':>9}  {'OpenBLAS':>9}")
    for name in ("NLMS", "KF", "IMM"):
        nr_py = overall_nr_db(d, py[name]["e"])
        nr_pure = overall_nr_db(d, e_pure[name])
        nr_blas = overall_nr_db(d, e_blas[name]) if e_blas else float("nan")
        print(f"  {name:<6}  {nr_py:>+9.3f}  {nr_pure:>+9.3f}  {nr_blas:>+9.3f}")

    # --- Speed table ---
    print("\n--- per-sample latency [microseconds] ---")
    rows = [("Python", {
        "NLMS": 1e6 * py["NLMS"]["wall_sec"] / N,
        "KF":   1e6 * py["KF"]  ["wall_sec"] / N,
        "IMM":  1e6 * py["IMM"] ["wall_sec"] / N,
    })]
    rows.append(("pure-C", {
        "NLMS": timing_pure["per_sample_us_nlms"],
        "KF":   timing_pure["per_sample_us_kf"],
        "IMM":  timing_pure["per_sample_us_imm"],
    }))
    if timing_blas:
        rows.append(("OpenBLAS", {
            "NLMS": timing_blas["per_sample_us_nlms"],
            "KF":   timing_blas["per_sample_us_kf"],
            "IMM":  timing_blas["per_sample_us_imm"],
        }))
    print(f"{'impl':<10}  {'NLMS':>9}  {'KF':>9}  {'IMM':>9}")
    for label, row in rows:
        print(f"  {label:<8}  {row['NLMS']:>9.3f}  {row['KF']:>9.3f}  {row['IMM']:>9.3f}")

    # --- Speedup vs Python ---
    print("\n--- speedup vs Python NumPy ---")
    py_us = rows[0][1]
    for label, row in rows[1:]:
        print(f"  {label:<8}  NLMS {py_us['NLMS']/row['NLMS']:>5.1f}x   "
              f"KF {py_us['KF']/row['KF']:>5.1f}x   "
              f"IMM {py_us['IMM']/row['IMM']:>5.1f}x")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 4.6))
    impls = [r[0] for r in rows]
    NLMS_us = [r[1]["NLMS"] for r in rows]
    KF_us   = [r[1]["KF"]   for r in rows]
    IMM_us  = [r[1]["IMM"]  for r in rows]
    x = np.arange(len(impls))
    width = 0.27
    ax.bar(x - width, NLMS_us, width, label="NLMS")
    ax.bar(x,         KF_us,   width, label="KF (quiet)")
    ax.bar(x + width, IMM_us,  width, label="IMM-KF")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(impls)
    ax.set_ylabel("per-sample latency  [us, log scale]")
    ax.set_title("ANC filter step latency, single thread")
    ax.grid(True, axis="y", which="both", ls=":", alpha=0.5)
    ax.legend()
    for xi, val in zip(x - width, NLMS_us): ax.text(xi, val*1.1, f"{val:.2f}", ha="center", fontsize=8)
    for xi, val in zip(x,         KF_us):   ax.text(xi, val*1.1, f"{val:.2f}", ha="center", fontsize=8)
    for xi, val in zip(x + width, IMM_us):  ax.text(xi, val*1.1, f"{val:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    out_png = ROOT / "figures" / "08_c_port_speed.png"
    out_png.parent.mkdir(exist_ok=True)
    fig.savefig(out_png, dpi=120)
    print(f"\nsaved {out_png}")

    out_npz = ROOT / "figures" / "08_c_port_speed.npz"
    np.savez(out_npz,
             impls=np.array(impls),
             nlms_us=np.array(NLMS_us),
             kf_us=np.array(KF_us),
             imm_us=np.array(IMM_us),
             diff_pure=np.array([diff_table[f][0] for f in ("NLMS","KF","IMM")]),
             diff_blas=np.array([diff_table[f][1] for f in ("NLMS","KF","IMM")]))
    print(f"saved raw numbers to {out_npz}")


if __name__ == "__main__":
    main()
