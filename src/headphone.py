"""Virtual-headphone rendering: turn raw ANC simulator outputs into audio that
*sounds like* what a real ANC earbud delivers to the eardrum.

The simulator gives, per algorithm:
    d(k)  -- the noise reaching the eardrum with no cancellation
    e(k)  -- the residual after the (full-band) adaptive controller
so the anti-noise the controller produced is  y(k) = d(k) - e(k).

Two physical facts that the bare simulator ignores, and which dominate how
ANC *feels*:

1. **Active ANC only works at low frequencies.** A real feedforward/hybrid loop
   cancels roughly 20 Hz - 1.5 kHz; above that, secondary-path phase/group-delay
   uncertainty makes active cancellation ineffective (and it can even add noise).
   So the realistically-cancelled noise keeps the controller's residual `e` only
   in the low band, and falls back to the raw `d` in the high band:

       n_active = highpass(d) + lowpass(e)          (the "ANC band split")

2. **The passive seal attenuates highs.** Even with ANC off, the ear-tip blocks a
   lot of high-frequency energy while passing the lows. Modelled as a high-shelf
   cut `H_pass` (≈0 dB at DC, ≈ -22 dB above a few kHz), applied to the noise
   component only (music is injected by the driver inside the seal).

The eardrum scene then has four listenable variants:

    open : no earbud at all (loudest)            -> d
    off  : earbud in, ANC off                    -> music + H_pass·d
    on   : earbud in, ANC on                     -> music + H_pass·n_active
    ref  : perfect silence + music (the goal)    -> music

Crucially, all variants are written with a SINGLE shared gain (`write_common_gain`)
so the loudness ladder open > off > on is preserved. Per-clip peak normalization
(the old `normalize_for_wav`) destroys exactly the "it got quiet" sensation that
ANC is about, so it is deliberately not used here.

All filtering is zero-phase (`sosfiltfilt`): this is an offline render, so we add
no artificial group delay that would itself be an audible artefact.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from . import config as cfg


# ---- Default virtual-headphone parameters -----------------------------------
ANC_BAND_FC = 1200.0      # Hz   active ANC is effective below this
PASSIVE_FC = 900.0        # Hz   passive seal starts cutting above this
PASSIVE_HF_ATTEN_DB = -22.0   # dB high-frequency attenuation of the passive seal
_FILTER_ORDER = 4


def _lowpass(x: np.ndarray, fs: int, fc: float, order: int = _FILTER_ORDER) -> np.ndarray:
    """Zero-phase Butterworth low-pass."""
    sos = butter(order, fc / (fs / 2.0), btype="low", output="sos")
    return sosfiltfilt(sos, x)


def band_split(x: np.ndarray, fs: int, fc: float) -> tuple[np.ndarray, np.ndarray]:
    """Complementary zero-phase split: returns (low, high) with low + high == x."""
    low = _lowpass(x, fs, fc)
    return low, x - low


def passive_isolation(x: np.ndarray, fs: int,
                      fc: float = PASSIVE_FC,
                      hf_atten_db: float = PASSIVE_HF_ATTEN_DB) -> np.ndarray:
    """High-shelf cut modelling the ear-tip's passive isolation.

    Lows pass at ~0 dB; highs are attenuated by `hf_atten_db`. Implemented as a
    complementary split so the shelf is clean and zero-phase:
        out = low + g_hf * high,    g_hf = 10**(hf_atten_db/20)
    """
    g_hf = 10.0 ** (hf_atten_db / 20.0)
    low, high = band_split(x, fs, fc)
    return low + g_hf * high


def active_residual(d: np.ndarray, e: np.ndarray, fs: int,
                    anc_fc: float = ANC_BAND_FC) -> np.ndarray:
    """Band-limit the controller to the frequencies where active ANC actually
    works: keep its residual `e` in the low band, fall back to the raw `d` in
    the high band.

        n_active = highpass(d) + lowpass(e)
    """
    e_low = _lowpass(e, fs, anc_fc)
    d_low = _lowpass(d, fs, anc_fc)
    d_high = d - d_low
    return d_high + e_low


def render_eardrum(d: np.ndarray, e: np.ndarray,
                   music: np.ndarray | None = None,
                   fs: int = cfg.FS,
                   *,
                   anc_fc: float = ANC_BAND_FC,
                   passive_fc: float = PASSIVE_FC,
                   passive_hf_atten_db: float = PASSIVE_HF_ATTEN_DB) -> dict[str, np.ndarray]:
    """Build the four eardrum variants (see module docstring).

    `d`, `e` are the simulator's no-ANC noise and post-ANC residual. If `music`
    is given, it is mixed in as the wanted program (delivered cleanly by the
    driver); otherwise the variants are noise-only (the loudness-ladder demo).
    Returns a dict with keys 'open', 'off', 'on', and (if music) 'ref'.
    """
    n_active = active_residual(d, e, fs, anc_fc=anc_fc)
    n_off = passive_isolation(d, fs, passive_fc, passive_hf_atten_db)
    n_on = passive_isolation(n_active, fs, passive_fc, passive_hf_atten_db)

    out: dict[str, np.ndarray] = {"open": np.asarray(d, dtype=np.float64).copy()}
    if music is not None:
        music = np.asarray(music, dtype=np.float64)
        out["off"] = music + n_off
        out["on"] = music + n_on
        out["ref"] = music.copy()
    else:
        out["off"] = n_off
        out["on"] = n_on
    return out


def common_gain(signals: dict[str, np.ndarray],
                peak: float = 0.9,
                ref_key: str | None = None) -> float:
    """One shared playback gain so relative loudness is preserved across a set.

    Gain is `peak / max_peak`, where `max_peak` is the largest absolute sample
    over *all* signals (or over `signals[ref_key]` if given). Apply this same
    gain to every clip in the set — never peak-normalize clips individually.
    """
    if ref_key is not None:
        ref_peak = float(np.max(np.abs(signals[ref_key])) + 1e-12)
    else:
        ref_peak = max(float(np.max(np.abs(s)) + 1e-12) for s in signals.values())
    return peak / ref_peak


def write_common_gain(signals: dict[str, np.ndarray], fs: int,
                      out_dir: Path, prefix: str = "",
                      peak: float = 0.9, ref_key: str | None = None) -> float:
    """Write a whole comparison set to WAV with one shared gain (see
    `common_gain`). Returns the gain G that was applied.

    Files land at `out_dir/{prefix}{name}.wav`. Samples are clip-guarded to
    [-1, 1] but NOT individually normalized.
    """
    import soundfile as sf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g = common_gain(signals, peak=peak, ref_key=ref_key)
    for name, sig in signals.items():
        y = np.clip(np.asarray(sig, dtype=np.float64) * g, -1.0, 1.0)
        sf.write(out_dir / f"{prefix}{name}.wav", y.astype(np.float32), fs)
    return g


def to_wav_bytes(signal: np.ndarray, fs: int, gain: float = 1.0) -> bytes:
    """In-memory WAV bytes for Streamlit's `st.audio`, with an explicit shared
    gain (use `common_gain` to compute one gain for a whole set, then pass it
    here for every clip so the loudness ladder is preserved). Clip-guarded only.
    """
    import io
    import soundfile as sf

    y = np.clip(np.asarray(signal, dtype=np.float64) * gain, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, y.astype(np.float32), fs, format="WAV")
    return buf.getvalue()
