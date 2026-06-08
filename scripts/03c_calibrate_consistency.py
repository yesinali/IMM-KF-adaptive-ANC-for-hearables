"""Calibrate per-mode (sigma_q2, sigma_r2) for filter consistency.

For each mode, run that mode's noise on its own plant (so the filter has a
single Wiener fixed point and consistency is a fair test), sweep sigma_q2
over a log-grid for a few choices of sigma_r2, and report time-averaged NEES,
NIS, and overall NR. The objective is the (Q, R) pair that places NEES
inside the chi^2 band while still delivering competitive NR.

Run:
    python -m scripts.03c_calibrate_consistency
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src import scenario as sc
from src.filters import KalmanANCFilter
from src.anc import simulate_anc
from src.metrics import chi2_bounds, overall_nr_db


# Per-mode single-segment scenario (8 s of pure mode noise on its own plant).
def make_single_mode_scenario(mode: str, rng_seed: int, duration_sec: float = 8.0):
    rng = np.random.default_rng(rng_seed)
    seg = (sc.ScenarioSegment(mode, float(duration_sec)),)
    return sc.build_scenario(segments=seg, rng=rng, mode_conditioned_plants=False)


def main() -> None:
    nees_decimate = 100         # one NEES sample every 100 audio samples
    duration_sec = 8.0
    skip_frac = 0.20            # discard first 20% transient

    print(f"Calibration sweep: L={cfg.L}, FS={cfg.FS}, duration={duration_sec}s, "
          f"nees_decimate={nees_decimate}")
    print()

    Q_grid = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
    R_grid = [1.0, 10.0, 100.0, 1000.0]

    results = {}  # mode -> list of dicts

    for mode_name in cfg.MODE_NAMES:
        print(f"=== Mode: {mode_name} ===")
        s = make_single_mode_scenario(mode_name, rng_seed=hash(mode_name) & 0xFFFF)
        w_opt = sc.wiener_weights(s, L=cfg.L)
        N = len(s.d)
        n_kept = int((1 - skip_frac) * N)
        nees_band = chi2_bounds(n_kept // nees_decimate, dof=cfg.L)
        nis_band = chi2_bounds(n_kept, dof=1)

        print(f"  NEES band: [{nees_band[0]:.2f}, {nees_band[1]:.2f}] "
              f"(target = {cfg.L})")
        print(f"  NIS  band: [{nis_band[0]:.4f}, {nis_band[1]:.4f}] (target = 1)")
        print(f"  {'Q':>10s}  {'R':>8s}  {'NR [dB]':>9s}  {'NEES':>10s}  "
              f"{'NIS':>9s}  {'verdict':<14s}")
        print("  " + "-" * 70)

        results[mode_name] = []
        skip_samples = N - n_kept

        for Q in Q_grid:
            for R in R_grid:
                t0 = time.perf_counter()
                filt = KalmanANCFilter(L=cfg.L, sigma_q2=Q, sigma_r2=R)
                r = simulate_anc(s, filt, w_opt=w_opt,
                                 log_nees=True, log_nis=True,
                                 nees_decimate=nees_decimate)
                # Trim transient
                nees_arr = r["nees"][skip_samples // nees_decimate:]
                nis_arr = r["nis"][skip_samples:]
                nr_db = overall_nr_db(s.d, r["e"][skip_samples:])
                nees_mean = float(nees_arr.mean())
                nis_mean = float(nis_arr.mean())

                # Score: consistency proximity * NR
                nees_in_band = nees_band[0] <= nees_mean <= nees_band[1]
                nis_in_band = nis_band[0] <= nis_mean <= nis_band[1]
                if nees_in_band and nis_in_band:
                    verdict = "BOTH CONSIST"
                elif nees_in_band:
                    verdict = "NEES only"
                elif nis_in_band:
                    verdict = "NIS only"
                else:
                    verdict = ""

                results[mode_name].append({
                    "Q": Q, "R": R, "NR": nr_db,
                    "NEES": nees_mean, "NIS": nis_mean,
                    "nees_ok": nees_in_band, "nis_ok": nis_in_band,
                })
                tag = " <-" if (nees_in_band and nis_in_band) else \
                      "  *" if nees_in_band or nis_in_band else "   "
                print(f"  {Q:>10.0e}  {R:>8.0f}  {nr_db:>+9.2f}  "
                      f"{nees_mean:>10.2f}  {nis_mean:>9.4f}  "
                      f"{verdict:<14s}{tag}")
        print()

    # Best-of-mode summary
    print("=" * 78)
    print("RECOMMENDED PARAMETERS PER MODE")
    print("=" * 78)
    print(f"  Selection: highest NR among (Q,R) with NEES in band; "
          f"else closest to target.")
    print()
    print(f"  {'mode':<8s}  {'sigma_q2':>10s}  {'sigma_r2':>8s}  "
          f"{'NR [dB]':>9s}  {'NEES':>10s}  {'NIS':>9s}")
    print("  " + "-" * 60)

    recommendations = {}
    for mode_name in cfg.MODE_NAMES:
        cand = results[mode_name]
        # Prefer NEES-in-band; among those, prefer highest NR.
        in_band = [c for c in cand if c["nees_ok"]]
        if in_band:
            best = max(in_band, key=lambda c: c["NR"])
        else:
            # NEES closest to target L, breaking ties by NR
            target = cfg.L
            best = min(cand, key=lambda c: (abs(np.log(c["NEES"]/target)), -c["NR"]))
        recommendations[mode_name] = best
        print(f"  {mode_name:<8s}  {best['Q']:>10.0e}  {best['R']:>8.0f}  "
              f"{best['NR']:>+9.2f}  {best['NEES']:>10.2f}  {best['NIS']:>9.4f}")

    print()
    print("Suggested config.py update:")
    print("MODE_PARAMS = (")
    for mode_name in cfg.MODE_NAMES:
        b = recommendations[mode_name]
        print(f'    ModeParams("{mode_name:<7s}", '
              f'sigma_q2={b["Q"]:.0e}, sigma_r2={b["R"]:.1f}),')
    print(")")


if __name__ == "__main__":
    main()
