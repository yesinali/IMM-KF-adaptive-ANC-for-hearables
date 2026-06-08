"""Assemble a complete ANC scenario: mode trajectory + signals.

Signal flow (feedforward ANC):
    n(k)  -- mode-conditioned synthetic noise source
    x(k)  = n(k)                  (reference microphone, idealized to source)
    d(k)  = P_m(z) * n(k)         (primary path, possibly mode-dependent)
    xf(k) = S_m(z) * x(k)         (filtered reference, possibly mode-dependent)
    e(k)  = d(k) - xf(k)^T w(k) + v(k)   (the model the IMM-KF sees)

With `mode_conditioned_plants=True`, each mode owns its own random P, S IR,
so the optimal Wiener controller is mode-dependent and the regime-switching
benefit of IMM becomes observable.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from . import config as cfg
from . import noise as ns
from . import paths as pth


@dataclass
class ScenarioSegment:
    mode: str
    duration_sec: float


@dataclass
class ANCScenario:
    fs: int
    mode_labels: np.ndarray
    n_source: np.ndarray
    x_ref: np.ndarray
    d: np.ndarray
    x_filt: np.ndarray
    primary_irs: dict           # mode_name -> IR
    secondary_irs: dict
    segments: Sequence[ScenarioSegment]
    mode_conditioned: bool = False

    @property
    def duration_sec(self) -> float:
        return len(self.n_source) / self.fs

    @property
    def primary_ir(self) -> np.ndarray:
        """First mode's IR (for legacy single-plant scenarios)."""
        return next(iter(self.primary_irs.values()))

    @property
    def secondary_ir(self) -> np.ndarray:
        return next(iter(self.secondary_irs.values()))


DEFAULT_TRAJECTORY: tuple[ScenarioSegment, ...] = (
    ScenarioSegment("quiet",   30.0),
    ScenarioSegment("traffic", 30.0),
    ScenarioSegment("wind",    30.0),
    ScenarioSegment("babble",  30.0),
    ScenarioSegment("quiet",   30.0),
)


def _mode_index(name: str) -> int:
    return cfg.MODE_NAMES.index(name)


def build_scenario(segments: Sequence[ScenarioSegment] = DEFAULT_TRAJECTORY,
                   rng: np.random.Generator | None = None,
                   fs: int = cfg.FS,
                   mode_conditioned_plants: bool = False,
                   crossfade_sec: float = 0.0) -> ANCScenario:
    """Generate a full scenario.

    If `mode_conditioned_plants` is True, each mode gets its own random
    (P, S) pair, so segment boundaries also switch the acoustic plant.

    `crossfade_sec` blends consecutive segments over an equal-power overlap
    instead of switching the environment (and its plant) instantaneously. This
    removes the artificial click/transient artefacts that hard segment cuts
    inject at every boundary, modelling how real environments fade into one
    another. `crossfade_sec=0` reproduces the legacy hard-cut behaviour
    exactly (same rng draw order, same total length, identical samples).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if mode_conditioned_plants:
        primary_irs = {m: pth.primary_path(rng) for m in cfg.MODE_NAMES}
        secondary_irs = {m: pth.secondary_path(rng) for m in cfg.MODE_NAMES}
    else:
        p_shared = pth.primary_path(rng)
        s_shared = pth.secondary_path(rng)
        primary_irs = {m: p_shared for m in cfg.MODE_NAMES}
        secondary_irs = {m: s_shared for m in cfg.MODE_NAMES}

    # Per-segment generation order is identical regardless of assembly, so the
    # rng draws (hence numerics) match the legacy path for crossfade_sec=0.
    n_segments = []
    d_segments = []
    xf_segments = []
    mode_idx = []
    for seg in segments:
        n_samples = int(round(seg.duration_sec * fs))
        n_seg = ns.generate_noise(seg.mode, n_samples, rng, fs=fs)
        n_segments.append(n_seg)
        d_segments.append(pth.apply_fir(primary_irs[seg.mode], n_seg))
        xf_segments.append(pth.apply_fir(secondary_irs[seg.mode], n_seg))
        mode_idx.append(_mode_index(seg.mode))

    overlap = int(round(crossfade_sec * fs))
    n_source, d, x_filt, labels = _assemble_segments(
        n_segments, d_segments, xf_segments, mode_idx, overlap)

    return ANCScenario(
        fs=fs,
        mode_labels=labels,
        n_source=n_source,
        x_ref=n_source.copy(),
        d=d,
        x_filt=x_filt,
        primary_irs=primary_irs,
        secondary_irs=secondary_irs,
        segments=tuple(segments),
        mode_conditioned=mode_conditioned_plants,
    )


def _assemble_segments(n_segments: list[np.ndarray],
                       d_segments: list[np.ndarray],
                       xf_segments: list[np.ndarray],
                       mode_idx: list[int],
                       overlap: int):
    """Overlap-add the per-segment signals with an equal-power crossfade of
    `overlap` samples. Returns (n_source, d, x_filt, mode_labels).

    With overlap=0 this is a plain concatenation (legacy behaviour). Otherwise
    each boundary blends the tail of one segment with the head of the next using
    sqrt(t)/sqrt(1-t) ramps (equal power for incoherent noise), and the same
    weights drive the noise, the primary signal d, and the filtered reference
    xf so their controller-relevant relationship is preserved.
    """
    K = len(n_segments)
    lengths = [len(s) for s in n_segments]

    # Clamp overlap so it never exceeds half of the shortest interior segment.
    if overlap > 0 and K > 1:
        overlap = min(overlap, min(lengths) // 2)
    overlap = max(0, overlap)

    starts = [0] * K
    for i in range(1, K):
        starts[i] = starts[i - 1] + lengths[i - 1] - overlap
    total = starts[-1] + lengths[-1]

    n_source = np.zeros(total)
    d = np.zeros(total)
    x_filt = np.zeros(total)
    labels = np.zeros(total, dtype=np.int8)

    if overlap > 0:
        t = (np.arange(overlap) + 0.5) / overlap
        fade_in = np.sqrt(t)
        fade_out = np.sqrt(1.0 - t)

    for i in range(K):
        Li = lengths[i]
        win = np.ones(Li)
        if overlap > 0:
            if i > 0:
                win[:overlap] = fade_in
            if i < K - 1:
                win[-overlap:] = fade_out
        sl = slice(starts[i], starts[i] + Li)
        n_source[sl] += win * n_segments[i]
        d[sl] += win * d_segments[i]
        x_filt[sl] += win * xf_segments[i]

    # Ground-truth label switches at each overlap centre (legacy boundary when
    # overlap=0). Segment i owns [prev_switch, switch_i).
    half = overlap // 2
    prev = 0
    for i in range(K):
        switch = (starts[i + 1] + half) if i < K - 1 else total
        labels[prev:switch] = mode_idx[i]
        prev = switch

    return n_source, d, x_filt, labels


def _toeplitz(r: np.ndarray) -> np.ndarray:
    L = len(r)
    R = np.empty((L, L))
    for i in range(L):
        for j in range(L):
            R[i, j] = r[abs(i - j)]
    return R


def _wiener_from_corrs(r_xx: np.ndarray, r_xd: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    L = len(r_xx)
    R = _toeplitz(r_xx)
    return np.linalg.solve(R + ridge * np.eye(L), r_xd)


def wiener_weights(scenario: ANCScenario, L: int = cfg.L) -> np.ndarray:
    """Global Wiener filter over the full scenario (single-plant approximation)."""
    xf = scenario.x_filt
    d = scenario.d
    n = len(xf)
    r_xx = np.array([np.dot(xf[k:], xf[: n - k]) / max(1, n - k) for k in range(L)])
    r_xd = np.array([np.dot(xf[: n - k], d[k:]) / max(1, n - k) for k in range(L)])
    return _wiener_from_corrs(r_xx, r_xd)


def wiener_weights_per_mode(scenario: ANCScenario, L: int = cfg.L) -> dict:
    """Mode-conditioned Wiener filters, computed from each mode's segments.

    Returns dict[mode_name] -> length-L weight vector.
    """
    accum = {m: {"r_xx": np.zeros(L), "r_xd": np.zeros(L), "n": 0}
             for m in cfg.MODE_NAMES}
    idx = 0
    for seg in scenario.segments:
        n_samples = int(round(seg.duration_sec * scenario.fs))
        xf_seg = scenario.x_filt[idx: idx + n_samples]
        d_seg = scenario.d[idx: idx + n_samples]
        a = accum[seg.mode]
        for k in range(L):
            a["r_xx"][k] += np.dot(xf_seg[k:], xf_seg[: n_samples - k])
            a["r_xd"][k] += np.dot(xf_seg[: n_samples - k], d_seg[k:])
        a["n"] += n_samples
        idx += n_samples

    out = {}
    for mode_name, a in accum.items():
        if a["n"] < L * 10:
            continue
        out[mode_name] = _wiener_from_corrs(a["r_xx"] / a["n"], a["r_xd"] / a["n"])
    return out


def per_sample_wiener_array(scenario: ANCScenario,
                            w_opt_per_mode: dict) -> np.ndarray:
    """Expand the per-mode dict into a per-sample (N, L) array indexed by mode_labels."""
    N = len(scenario.d)
    L = next(iter(w_opt_per_mode.values())).shape[0]
    arr = np.zeros((N, L))
    for mode_idx, mode_name in enumerate(cfg.MODE_NAMES):
        if mode_name in w_opt_per_mode:
            mask = scenario.mode_labels == mode_idx
            arr[mask] = w_opt_per_mode[mode_name]
    return arr
