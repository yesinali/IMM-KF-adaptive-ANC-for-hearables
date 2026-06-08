"""Dump a deterministic ANC scenario to a binary file consumed by c_imm/main.c.

The C binary will read this file, run NLMS / KF / IMM, and write the residuals
back to its own output binary; `scripts/c_port_compare.py` then re-reads both
and validates numerics + speed against the NumPy reference.

Run:
    python -m scripts.c_port_dump  [--out scenario.bin] [--seed 2026] [--duration 20]
"""
from __future__ import annotations
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src import scenario as sc
from src.utils import transition_matrix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str,
                    default=str(ROOT / "c_imm" / "scenario.bin"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--duration", type=float, default=20.0,
                    help="total scenario length in seconds (split across 4 modes)")
    args = ap.parse_args()

    seg_sec = args.duration / 4.0
    traj = (
        sc.ScenarioSegment("quiet",   seg_sec),
        sc.ScenarioSegment("traffic", seg_sec),
        sc.ScenarioSegment("wind",    seg_sec),
        sc.ScenarioSegment("babble",  seg_sec),
    )
    rng = np.random.default_rng(args.seed)
    s = sc.build_scenario(segments=traj, rng=rng, mode_conditioned_plants=True)

    quiet = next(p for p in cfg.MODE_PARAMS if p.name == "quiet")
    Pi = transition_matrix()
    Q_imm = np.array([p.sigma_q2 for p in cfg.MODE_PARAMS], dtype=np.float64)
    R_imm = np.array([p.sigma_r2 for p in cfg.MODE_PARAMS], dtype=np.float64)

    N = int(len(s.d))
    L = int(cfg.L)
    M = int(len(cfg.MODE_PARAMS))
    FS = int(cfg.FS)
    W_lik = 200
    mu_nlms = 0.10
    Q_kf = float(quiet.sigma_q2)
    R_kf = float(quiet.sigma_r2)

    xf = np.ascontiguousarray(s.x_filt, dtype=np.float64)
    d = np.ascontiguousarray(s.d, dtype=np.float64)
    assert xf.shape == (N,) and d.shape == (N,)

    Pi_flat = np.ascontiguousarray(Pi.astype(np.float64))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fp:
        # Header (5x int32)
        fp.write(struct.pack("<iiiii", N, L, M, FS, W_lik))
        # Scalars
        fp.write(struct.pack("<ddd", mu_nlms, Q_kf, R_kf))
        # Vectors
        fp.write(Pi_flat.tobytes(order="C"))
        fp.write(Q_imm.tobytes())
        fp.write(R_imm.tobytes())
        fp.write(xf.tobytes())
        fp.write(d.tobytes())

    bytes_total = out_path.stat().st_size
    print(f"wrote {out_path}  ({bytes_total/1e6:.2f} MB)")
    print(f"  N={N}  L={L}  M={M}  FS={FS}  W_lik={W_lik}")
    print(f"  audio length = {N/FS:.2f} s")
    print(f"  mu_NLMS={mu_nlms}, KF (Q={Q_kf:.0e}, R={R_kf:.0f})")
    print(f"  IMM Q = {Q_imm}")
    print(f"  IMM R = {R_imm}")

    # Also store a sidecar .npz with the same scenario for the comparison
    # script (it needs d, w_opt_arr, and the mode labels for reference).
    w_opt_per_mode = sc.wiener_weights_per_mode(s, L=cfg.L)
    w_opt_arr = sc.per_sample_wiener_array(s, w_opt_per_mode)
    npz_path = out_path.with_suffix(".npz")
    np.savez(npz_path,
             xf=xf, d=d,
             w_opt=w_opt_arr,
             mode_labels=s.mode_labels,
             fs=FS, L=L, M=M)
    print(f"wrote sidecar {npz_path}  ({npz_path.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
