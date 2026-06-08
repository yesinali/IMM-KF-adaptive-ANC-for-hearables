"""Streamlit-facing wrapper around the C-port binaries (`c_imm/imm_pure.exe`
and `c_imm/imm_blas.exe`).

The C binaries always run all three filters (NLMS, KF, IMM-KF) in one pass;
this helper dumps the scenario to a temp binary, invokes the requested
backend, reads back the per-filter residuals, and returns whichever one the
caller asked for. NR is computed by the caller from `e` and `s.d`.

For IMM-KF the C binary does not (yet) export the mode posterior, so for
mode-tracking plots the caller must still run the Python IMM in parallel.
This is documented in the UI; the C numbers are used for the *speed* claim
and the residual itself.
"""
from __future__ import annotations
import json
import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config as cfg
from .scenario import ANCScenario
from .utils import transition_matrix


ROOT = Path(__file__).resolve().parents[1]
C_DIR = ROOT / "c_imm"


# Map UI label -> binary filename
BINARIES = {
    "pure-c":   C_DIR / "imm_pure.exe",
    "openblas": C_DIR / "imm_blas.exe",
}


def is_available(label: str) -> bool:
    return label in BINARIES and BINARIES[label].exists()


def available_backends() -> list[str]:
    return [lbl for lbl in BINARIES if is_available(lbl)]


@dataclass
class CBackendResult:
    e_nlms: np.ndarray
    e_kf:   np.ndarray
    e_imm:  np.ndarray
    mu_history: np.ndarray  # (N, M) IMM mode posterior over time
    timing: dict
    backend: str


def _dump_scenario(scenario: ANCScenario,
                   filter_kind: str,
                   params: dict,
                   tmp_dir: Path,
                   L_override: int | None = None) -> Path:
    """Serialize the scenario + filter params to the binary format expected
    by `c_imm/main.c`.

    Whichever filter the user picks, we always pass full IMM + KF + NLMS
    config so the C binary can run all three. The caller selects which
    residual to use."""
    quiet = next(p for p in cfg.MODE_PARAMS if p.name == "quiet")
    # Defaults used by the C binary even when the user picked another method
    mu_nlms = float(params.get("mu", 0.10)) if filter_kind == "NLMS" else 0.10
    if filter_kind == "Kalman":
        Q_kf = float(params["sigma_q2"])
        R_kf = float(params["sigma_r2"])
    else:
        Q_kf = float(quiet.sigma_q2)
        R_kf = float(quiet.sigma_r2)
    W_lik = int(params.get("window", 200)) if filter_kind == "IMM-KF" else 200

    Pi = transition_matrix().astype(np.float64)
    Q_imm = np.array([p.sigma_q2 for p in cfg.MODE_PARAMS], dtype=np.float64)
    R_imm = np.array([p.sigma_r2 for p in cfg.MODE_PARAMS], dtype=np.float64)

    xf = np.ascontiguousarray(scenario.x_filt, dtype=np.float64)
    d  = np.ascontiguousarray(scenario.d,      dtype=np.float64)
    N  = int(len(d))

    in_path = tmp_dir / "scenario.bin"
    with open(in_path, "wb") as fp:
        L_dump = int(L_override) if L_override is not None else int(cfg.L)
        fp.write(struct.pack("<iiiii",
                             N, L_dump, int(len(cfg.MODE_PARAMS)),
                             int(scenario.fs), W_lik))
        fp.write(struct.pack("<ddd", mu_nlms, Q_kf, R_kf))
        fp.write(np.ascontiguousarray(Pi).tobytes(order="C"))
        fp.write(Q_imm.tobytes())
        fp.write(R_imm.tobytes())
        fp.write(xf.tobytes())
        fp.write(d.tobytes())
    return in_path


def _read_residuals(path: Path, N_expected: int
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with open(path, "rb") as fp:
        N = int(np.frombuffer(fp.read(4), dtype=np.int32)[0])
        M = int(np.frombuffer(fp.read(4), dtype=np.int32)[0])
        if N != N_expected:
            raise RuntimeError(f"residual file N mismatch: {N} != {N_expected}")
        e_nlms = np.frombuffer(fp.read(N * 8), dtype=np.float64).copy()
        e_kf   = np.frombuffer(fp.read(N * 8), dtype=np.float64).copy()
        e_imm  = np.frombuffer(fp.read(N * 8), dtype=np.float64).copy()
        mu_hist = np.frombuffer(fp.read(N * M * 8), dtype=np.float64).copy()
        mu_hist = mu_hist.reshape(N, M)
    return e_nlms, e_kf, e_imm, mu_hist


def run(scenario: ANCScenario,
        filter_kind: str,
        params: dict,
        backend: str = "pure-c",
        L_override: int | None = None) -> CBackendResult:
    """Run a C-port simulation and return all three residuals + timing JSON.

    `L_override` lets parameter sweeps vary the FIR length without touching
    the global `cfg.L`. When unset, `cfg.L` is used."""
    if not is_available(backend):
        raise FileNotFoundError(
            f"C backend `{backend}` not available — expected {BINARIES.get(backend)}. "
            f"Build it with `make pure` / `make blas` in c_imm/.")

    binary = BINARIES[backend]
    with tempfile.TemporaryDirectory(prefix="anc_c_") as tmp_str:
        tmp = Path(tmp_str)
        in_path  = _dump_scenario(scenario, filter_kind, params, tmp,
                                  L_override=L_override)
        out_path = tmp / "residuals.bin"
        proc = subprocess.run(
            [str(binary), str(in_path), str(out_path)],
            capture_output=True, text=True, check=True,
        )
        try:
            timing = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"could not parse JSON from {binary}: {exc}\nstdout:\n{proc.stdout}"
            )
        e_nlms, e_kf, e_imm, mu_hist = _read_residuals(out_path, int(len(scenario.d)))

    return CBackendResult(
        e_nlms=e_nlms, e_kf=e_kf, e_imm=e_imm,
        mu_history=mu_hist,
        timing=timing, backend=backend,
    )


def pick_residual(res: CBackendResult, filter_kind: str) -> np.ndarray:
    if filter_kind == "NLMS":   return res.e_nlms
    if filter_kind == "Kalman": return res.e_kf
    if filter_kind == "IMM-KF": return res.e_imm
    raise ValueError(filter_kind)


def filter_step_us(res: CBackendResult, filter_kind: str) -> float:
    if filter_kind == "NLMS":   return float(res.timing["per_sample_us_nlms"])
    if filter_kind == "Kalman": return float(res.timing["per_sample_us_kf"])
    if filter_kind == "IMM-KF": return float(res.timing["per_sample_us_imm"])
    raise ValueError(filter_kind)
