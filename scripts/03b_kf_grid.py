"""Phase B diagnostic: grid-search sigma_q2 / sigma_r2 for the single-mode KF.

The first Phase B run showed the KF diverging because the proposal-table
(Q, R) values were per-second scale, not per-sample. This script sweeps
a small grid and reports which combination matches or beats NLMS µ=0.1
on the same scenario.

Run:
    python -m scripts.03b_kf_grid
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
from src.filters import NLMSFilter, KalmanANCFilter
from src.anc import simulate_anc
from src.metrics import overall_nr_db


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
    tail = int(0.9 * len(s.d))

    print(f"scenario: {s.duration_sec:.1f}s, L={cfg.L}")
    print("\nNLMS reference:")
    for mu in (0.01, 0.1):
        t0 = time.perf_counter()
        r = simulate_anc(s, NLMSFilter(L=cfg.L, mu=mu), w_opt=w_opt)
        m_tail = r["misalignment"][tail:].mean()
        print(f"  µ={mu:5.3f}  NR={overall_nr_db(s.d, r['e']):+6.2f}dB  "
              f"misalign_tail={10*np.log10(m_tail):+6.2f}dB  "
              f"({time.perf_counter()-t0:.1f}s)")

    print("\nKF grid (sigma_q2, sigma_r2):")
    grid_q = [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
    grid_r = [1e-2, 1.0, 1e2]
    for q in grid_q:
        for r in grid_r:
            t0 = time.perf_counter()
            run = simulate_anc(s, KalmanANCFilter(L=cfg.L, sigma_q2=q, sigma_r2=r),
                               w_opt=w_opt)
            m_tail = run["misalignment"][tail:].mean()
            print(f"  Q={q:.0e}  R={r:.0e}  NR={overall_nr_db(s.d, run['e']):+6.2f}dB"
                  f"  misalign_tail={10*np.log10(m_tail+1e-12):+6.2f}dB"
                  f"  ({time.perf_counter()-t0:.1f}s)")


if __name__ == "__main__":
    main()
