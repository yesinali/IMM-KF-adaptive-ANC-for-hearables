"""Music and Feel — live virtual-headphone listening demo (the 'feel' side).

Pick algorithm(s), music-vs-noise SNR and crossfade; the page renders LIVE
(C backend when available, Python otherwise) and lets you hear:

  * the LOUDNESS LADDER  — open ear -> earbud ANC off -> earbud ANC on, all at
    a single shared gain so you actually hear the world get quieter;
  * a MUSIC A/B          — clean music vs music+noise (no ANC) vs music+residual
    (ANC on), music-aware (the controller adapts on the noise only).

Everything is produced on the fly via src/headphone.py + src/testbench.py — no
need to pre-run scripts/20_render_testbench. Part of the unified app
(`streamlit run app/streamlit_app.py`).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config as cfg                                 # noqa: E402
from src import scenario as sc                                # noqa: E402
from src import headphone as hp                               # noqa: E402
from src import testbench as tb                               # noqa: E402
from src.music import load_music_clip                         # noqa: E402

st.set_page_config(page_title="Music and Feel", page_icon="🎵", layout="wide")
MUSIC_DIR = ROOT / "musics"


# ---- cached compute ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_scenario(seed: int, crossfade: float, noise_source: str):
    rng = np.random.default_rng(seed)
    return sc.build_scenario(segments=tb.DEFAULT_TRAJ, rng=rng,
                             mode_conditioned_plants=True, crossfade_sec=crossfade)


@st.cache_data(show_spinner=False)
def get_residuals(seed: int, crossfade: float, noise_source: str, tags: tuple):
    s = get_scenario(seed, crossfade, noise_source)
    return tb.run_algorithms(s, tags=list(tags))


@st.cache_data(show_spinner=False)
def get_music(path_str: str, fs: int, duration_sec: float, target_rms: float):
    return load_music_clip(Path(path_str), target_fs=fs,
                           duration_sec=duration_sec, target_rms=target_rms)


def match_len(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == n:
        return x
    return x[:n] if len(x) > n else np.pad(x, (0, n - len(x)))


# ---- header & controls ------------------------------------------------------

st.title("🎵 Music and Feel")
st.caption("Hear what a real ANC earbud would deliver — rendered live through the passive-isolation "
           "+ low-frequency-only ANC chain, at a shared gain so the loudness drop is audible.")

with st.sidebar:
    st.header("🎚️ Render settings")
    algos = st.multiselect(
        "Algorithms to compare", options=list(tb.ALGO_LABELS),
        default=["v5", "nlms"], format_func=lambda t: tb.ALGO_LABELS[t],
        help="Trio (NLMS/KF/IMM-v5) renders fast via C; v6 & v5+smooth need Python.")
    snr = st.slider("Music–noise SNR (dB)", -6, 20, 6,
                    help="0 = equal power (harsh). Real hearable use is ~+6…+15 dB.")
    crossfade = st.slider("Crossfade between modes (s)", 0.0, 2.0, 0.75, 0.25)
    seed = int(st.number_input("Random seed", 1, 9999, 7))

    music_files = sorted([p.name for p in MUSIC_DIR.glob("*.flac")]
                         + [p.name for p in MUSIC_DIR.glob("*.wav")])
    options = music_files + ["⬆️ Upload my own"]
    music_choice = st.selectbox("Music", options, index=0 if music_files else len(options) - 1)
    uploaded = st.file_uploader("WAV/FLAC", type=["wav", "flac"]) if music_choice == "⬆️ Upload my own" else None

if not algos:
    st.info("👈 Pick at least one algorithm in the sidebar.")
    st.stop()

variant = [t for t in algos if t not in tb.C_TRIO]
spin = (f"Rendering via {tb.backend_in_use()} backend"
        + (" + Python for v6/v5smooth (first run is slower)…" if variant else "…"))

with st.spinner(spin):
    s = get_scenario(seed, crossfade, cfg.NOISE_SOURCE)
    fs = s.fs
    noise_rms = float(np.sqrt(np.mean(s.d ** 2)))
    target_rms = noise_rms * (10.0 ** (snr / 20.0))

    # music (built-in or uploaded)
    if uploaded is not None:
        import soundfile as sf
        import io
        audio, src_fs = sf.read(io.BytesIO(uploaded.getvalue()), dtype="float64", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if src_fs != fs:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(int(src_fs), fs)
            audio = resample_poly(audio, fs // g, int(src_fs) // g)
        rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-12)
        music = audio * (target_rms / rms)
        music = match_len(music, len(s.d))
    elif music_files:
        music = match_len(get_music(str(MUSIC_DIR / music_choice), fs, len(s.d) / fs, target_rms),
                          len(s.d))
    else:
        st.error("No music in musics/ — upload a file.")
        st.stop()

    residuals = get_residuals(seed, crossfade, cfg.NOISE_SOURCE, tuple(algos))

st.success(f"✅ {s.duration_sec:.1f}s scenario · backend **{tb.backend_in_use()}** · "
           f"input SNR **{snr:+d} dB** · music **{music_choice}**")


# ---- 1) Loudness ladder (noise only, one shared gain) -----------------------

st.header("🔊 1. Loudness ladder — hear it get quiet")
st.markdown("All clips share **one gain** (no per-clip normalization), so play them in order and "
            "feel the level drop. Noise only, no music.")

base = hp.render_eardrum(s.d, residuals[algos[0]], music=None, fs=fs)
ladder = {"open": base["open"], "off": base["off"]}
for t in algos:
    ladder[f"on_{t}"] = hp.render_eardrum(s.d, residuals[t], music=None, fs=fs)["on"]
g_lad = hp.common_gain(ladder)

cols = st.columns(2 + len(algos))
cols[0].markdown("**Open ear**\n\n_no earbud_")
cols[0].audio(hp.to_wav_bytes(ladder["open"], fs, g_lad), format="audio/wav")
cols[1].markdown("**ANC OFF**\n\n_passive seal only_")
cols[1].audio(hp.to_wav_bytes(ladder["off"], fs, g_lad), format="audio/wav")
metrics = tb.compute_metrics(s, {t: residuals[t] for t in algos}, music)
for i, t in enumerate(algos):
    m = metrics[t]
    cols[2 + i].markdown(f"**ANC ON**\n\n_{tb.ALGO_LABELS[t]}_  ·  −{m['dba_reduction_anc']:.1f} dB(A)")
    cols[2 + i].audio(hp.to_wav_bytes(ladder[f"on_{t}"], fs, g_lad), format="audio/wav")


# ---- 2) Music A/B (one shared gain) -----------------------------------------

st.markdown("---")
st.header("🎼 2. Music A/B — does it keep the music intact?")
st.markdown("Music-aware: the controller adapts on the **noise only**; you hear `music + residual`. "
            "Goal: as close to the clean reference as possible.")

mbase = hp.render_eardrum(s.d, residuals[algos[0]], music=music, fs=fs)
mset = {"ref": mbase["ref"], "off": mbase["off"]}
for t in algos:
    mset[f"on_{t}"] = hp.render_eardrum(s.d, residuals[t], music=music, fs=fs)["on"]
g_mus = hp.common_gain(mset)

mc = st.columns(2 + len(algos))
mc[0].markdown("**🎵 Clean**\n\n_reference_")
mc[0].audio(hp.to_wav_bytes(mset["ref"], fs, g_mus), format="audio/wav")
mc[1].markdown("**🚧 No ANC**\n\n_music + noise_")
mc[1].audio(hp.to_wav_bytes(mset["off"], fs, g_mus), format="audio/wav")
for i, t in enumerate(algos):
    m = metrics[t]
    mc[2 + i].markdown(f"**{tb.ALGO_LABELS[t]}**\n\nSI-SDR {m.get('si_sdr_db', float('nan')):+.1f} dB")
    mc[2 + i].audio(hp.to_wav_bytes(mset[f"on_{t}"], fs, g_mus), format="audio/wav")


# ---- 3) Metrics table -------------------------------------------------------

st.markdown("---")
st.header("📊 3. Metrics — numbers behind the feel")
import pandas as pd
rows = []
for t in algos:
    m = metrics[t]
    rows.append({
        "Algorithm": tb.ALGO_LABELS[t],
        "Controller NR [dB]": f"{m['ctrl_nr_db']:+.2f}",
        "Perceived drop [dB(A)]": f"{m['dba_reduction_anc']:+.2f}",
        "Low-band NR (<1.2k)": f"{m['anc_band_nr_db']:+.2f}",
        "High-band NR (>1.2k)": f"{m['passive_band_nr_db']:+.2f}",
        "SI-SDR [dB]": f"{m.get('si_sdr_db', float('nan')):+.2f}",
        "Musical-noise": f"{m['musical_noise']:.2f}",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.caption("Perceived drop & musical-noise come from src/perceptual.py; SI-SDR measures music "
           "recovery. Watch for the algo-vs-perception story: highest NR is not always the most natural.")

# Algo-vs-perception flag when comparing IMM v5 with a smoother baseline.
if "v5" in algos and len(algos) > 1:
    others = [t for t in algos if t != "v5"]
    if any(metrics[t]["musical_noise"] < metrics["v5"]["musical_noise"] for t in others):
        st.warning("⚖️ IMM v5 has the highest noise reduction here **and** the highest musical-noise "
                   "index — the classic metric-vs-perception tension. Listen above and judge by ear.")
