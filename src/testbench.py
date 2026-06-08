"""Reusable test-bench core: realistic-scenario algorithm residuals + metrics.

Shared by `scripts/20_render_testbench.py` (CLI: WAV files, plots) and
`app/pages/2_Music_and_Feel.py` (live interactive render). Keeping it here (a
clean, importable `src` module) lets both call the same orchestration — the
script's name starts with a digit and cannot be imported.

Backend policy: **C-preferred + Python fallback**. The C binary computes the
fixed trio (NLMS µ=0.10, KF wind-tuned, IMM v5) in one pass; the v6 (capped Q)
and v5+smooth variants exist only in Python. `run_algorithms(tags=...)` runs
just the requested banks, so the app can render only what the user picked.
"""
from __future__ import annotations
from typing import Sequence

import numpy as np

from . import config as cfg
from . import scenario as sc
from . import headphone as hp
from . import perceptual as pc
from . import c_backend
from .filters import NLMSFilter, KalmanANCFilter
from .imm import IMMKalmanANC
from .metrics import overall_nr_db


# 4 canonical hearable environments, 5 s each, blended by crossfade.
DEFAULT_TRAJ = (
    sc.ScenarioSegment("quiet",   5.0),
    sc.ScenarioSegment("traffic", 5.0),
    sc.ScenarioSegment("wind",    5.0),
    sc.ScenarioSegment("babble",  5.0),
)

V6_MODE_PARAMS = (
    cfg.ModeParams("quiet",   sigma_q2=1e-10, sigma_r2=100.0),
    cfg.ModeParams("babble",  sigma_q2=1e-6,  sigma_r2=10.0),
    cfg.ModeParams("traffic", sigma_q2=1e-10, sigma_r2=10.0),
    cfg.ModeParams("wind",    sigma_q2=1e-6,  sigma_r2=100.0),
)

# tag -> human label (tags are used in WAV filenames & app selectors)
ALGO_LABELS = {
    "nlms":     "NLMS µ=0.10",
    "kfwind":   "KF wind-tuned",
    "v5":       "IMM v5",
    "v6":       "IMM v6 (capped Q)",
    "v5smooth": "IMM v5 + ŵ smooth",
}

# Tags the C binary produces directly (one Kalman-mode call returns all three).
C_TRIO = ("nlms", "kfwind", "v5")


def run_python(s: sc.ANCScenario, filt, alpha_w_smooth: float = 0.0) -> np.ndarray:
    """Run a filter through the noise-only scenario, returning the residual e.
    Music-aware: the controller never sees the music; the listener does, via
    headphone.render_eardrum which adds music on top of this e."""
    xf, d = s.x_filt, s.d
    N, L = len(d), filt.L
    buf = np.zeros(L)
    e = np.zeros(N)
    w_smooth = np.zeros(L) if alpha_w_smooth > 0 else None
    for k in range(N):
        buf[1:] = buf[:-1]; buf[0] = xf[k]
        ek, w_now = filt.step(buf, d[k])
        if w_smooth is not None:
            w_smooth = (1 - alpha_w_smooth) * w_smooth + alpha_w_smooth * w_now
            e[k] = d[k] - w_smooth @ buf
        else:
            e[k] = ek
    return e


def _python_filter(tag: str, L: int):
    """(filter, alpha_w_smooth) for a tag, for the Python path."""
    wind = next(p for p in cfg.MODE_PARAMS if p.name == "wind")
    if tag == "nlms":
        return NLMSFilter(L=L, mu=0.10), 0.0
    if tag == "kfwind":
        return KalmanANCFilter(L=L, sigma_q2=wind.sigma_q2, sigma_r2=wind.sigma_r2), 0.0
    if tag == "v5":
        return IMMKalmanANC(L=L, likelihood_window=200), 0.0
    if tag == "v6":
        return IMMKalmanANC(L=L, mode_params=V6_MODE_PARAMS, likelihood_window=200), 0.0
    if tag == "v5smooth":
        return IMMKalmanANC(L=L, likelihood_window=200), 0.02
    raise ValueError(f"unknown algorithm tag '{tag}'")


def run_algorithms(s: sc.ANCScenario,
                   tags: Sequence[str] | None = None,
                   prefer_c: bool = True,
                   L: int = cfg.L) -> dict[str, np.ndarray]:
    """Return {tag: residual e} for the requested tags (default: all five).

    Uses the C backend for any requested trio tags when available (one call),
    and Python for the rest (the v6 / v5smooth variants, or everything if no C
    binary is built). Result preserves the requested tag order.
    """
    want = list(tags) if tags is not None else list(ALGO_LABELS)
    algos: dict[str, np.ndarray] = {}

    c_wanted = [t for t in want if t in C_TRIO]
    if prefer_c and c_wanted and c_backend.available_backends():
        backend = c_backend.available_backends()[0]
        wind = next(p for p in cfg.MODE_PARAMS if p.name == "wind")
        cres = c_backend.run(s, "Kalman",
                             {"sigma_q2": wind.sigma_q2, "sigma_r2": wind.sigma_r2},
                             backend=backend, L_override=L)
        pool = {"nlms": cres.e_nlms, "kfwind": cres.e_kf, "v5": cres.e_imm}
        for t in c_wanted:
            algos[t] = pool[t]

    for t in want:
        if t in algos:
            continue
        filt, sm = _python_filter(t, L)
        algos[t] = run_python(s, filt, alpha_w_smooth=sm)

    return {t: algos[t] for t in want}


def backend_in_use(prefer_c: bool = True) -> str:
    """'pure-c' / 'openblas' / 'python' — which backend the trio will use."""
    if prefer_c and c_backend.available_backends():
        return c_backend.available_backends()[0]
    return "python"


def compute_metrics(s: sc.ANCScenario, algos: dict[str, np.ndarray],
                    music: np.ndarray | None) -> dict[str, dict]:
    """Per-algorithm metric dict bridging energy NR and the felt experience."""
    fs, d = s.fs, s.d
    out: dict[str, dict] = {}
    for tag, e in algos.items():
        ren = hp.render_eardrum(d, e, music=None, fs=fs)
        anc_nr, passive_nr = pc.band_split_nr(d, e, fs, crossover=hp.ANC_BAND_FC)
        m = {
            "ctrl_nr_db": overall_nr_db(d, e),
            "dba_open": pc.dba_level(ren["open"], fs),
            "dba_off": pc.dba_level(ren["off"], fs),
            "dba_on": pc.dba_level(ren["on"], fs),
            "dba_reduction_anc": pc.dba_level(ren["off"], fs) - pc.dba_level(ren["on"], fs),
            "dba_reduction_total": pc.dba_level(ren["open"], fs) - pc.dba_level(ren["on"], fs),
            "anc_band_nr_db": anc_nr,
            "passive_band_nr_db": passive_nr,
            "musical_noise": pc.musical_noise_index(e, fs),
            "flux_burstiness": pc.flux_burstiness(e, fs),
        }
        if music is not None:
            user = music + e  # music-aware: listener hears music + residual
            m["si_sdr_db"] = pc.si_sdr(music, user)
            m["psnr_db"] = pc.music_psnr(music, user)
        out[tag] = m
    return out
