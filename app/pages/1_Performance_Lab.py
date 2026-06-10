"""Performance Lab — live interactive ANC simulation (numbers & plots side).

Pick a scenario + algorithm + backend, run the controller, and inspect the
engineering metrics (NR, mode tracking, real-time factor), the perceptual
bridge metrics (dBA loudness drop, band-split NR, musical-noise index), and the
time/NR/posterior/per-frequency plots. Audio A/B is the raw d vs e by default,
with an optional virtual-headphone render toggle.

Results of every run this session are kept in history; use the selector above
the results to switch between runs — all plots, metrics and audio re-render
for the chosen run.

Part of the unified app — launched from `streamlit run app/streamlit_app.py`.
"""
from __future__ import annotations
import io
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config as cfg                                         # noqa: E402
from src import scenario as sc                                       # noqa: E402
from src import paths as pth                                         # noqa: E402
from src import headphone as hp                                      # noqa: E402
from src import perceptual as pc                                     # noqa: E402
from src import testbench as tb                                      # noqa: E402
from src.filters import NLMSFilter, KalmanANCFilter                  # noqa: E402
from src.imm import IMMKalmanANC                                     # noqa: E402
from src.anc import simulate_anc                                     # noqa: E402
from src.metrics import noise_reduction_db_sliding, overall_nr_db    # noqa: E402
from src import c_backend                                            # noqa: E402

st.set_page_config(page_title="Performance Lab", page_icon="🎛️", layout="wide")

MAX_HISTORY = 8
if "run_history" not in st.session_state:
    st.session_state.run_history = []
if "run_counter" not in st.session_state:
    st.session_state.run_counter = 0


# =====================================================================
# Helpers
# =====================================================================

def to_wav_bytes(audio: np.ndarray, fs: int) -> bytes:
    """Raw engineering A/B: peak-normalized per clip (full-scale playback)."""
    audio = audio / (np.max(np.abs(audio)) + 1e-12) * 0.9
    buf = io.BytesIO()
    sf.write(buf, audio.astype(np.float32), fs, format="WAV")
    return buf.getvalue()


def build_filter(kind: str, L: int, params: dict):
    if kind == "NLMS":
        return NLMSFilter(L=L, mu=params["mu"])
    if kind == "Kalman":
        return KalmanANCFilter(L=L, sigma_q2=params["sigma_q2"], sigma_r2=params["sigma_r2"])
    if kind == "IMM-KF":
        return IMMKalmanANC(L=L, likelihood_window=params["window"])
    raise ValueError(kind)


def build_synthetic_scenario(durations: dict, mode_conditioned: bool, seed: int,
                             crossfade: float):
    rng = np.random.default_rng(seed)
    segments = [sc.ScenarioSegment(m, float(d)) for m, d in durations.items() if d > 0]
    return sc.build_scenario(segments=segments, rng=rng,
                             mode_conditioned_plants=mode_conditioned,
                             crossfade_sec=crossfade)


def build_uploaded_scenario(audio_bytes: bytes, seed: int):
    n_src, fs = sf.read(io.BytesIO(audio_bytes), dtype="float64", always_2d=False)
    if n_src.ndim > 1:
        n_src = n_src.mean(axis=1)
    if fs != cfg.FS:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(cfg.FS, int(fs))
        n_src = resample_poly(n_src, cfg.FS // g, int(fs) // g)
        fs = cfg.FS
    n_src = n_src / (np.std(n_src) + 1e-12)
    rng = np.random.default_rng(seed)
    p_ir = pth.primary_path(rng)
    s_ir = pth.secondary_path(rng)
    d = pth.apply_fir(p_ir, n_src)
    x_filt = pth.apply_fir(s_ir, n_src)
    labels = np.zeros(len(n_src), dtype=np.int8)
    return sc.ANCScenario(
        fs=fs, mode_labels=labels, n_source=n_src, x_ref=n_src.copy(),
        d=d, x_filt=x_filt,
        primary_irs={cfg.MODE_NAMES[0]: p_ir},
        secondary_irs={cfg.MODE_NAMES[0]: s_ir},
        segments=(sc.ScenarioSegment(cfg.MODE_NAMES[0], len(n_src) / fs),),
        mode_conditioned=False,
    )


def nr_color(nr_db: float) -> str:
    return "🟢" if nr_db >= 8 else "🟡" if nr_db >= 4 else "🟠" if nr_db >= 0 else "🔴"


def tracking_color(acc: float) -> str:
    return "🟢" if acc >= 70 else "🟡" if acc >= 50 else "🟠" if acc >= 30 else "🔴"


# =====================================================================
# Header
# =====================================================================

st.title("🎛️ Performance Lab")
st.caption("Live ANC simulation — pure performance: noise reduction, mode tracking, "
           "real-time factor, and the perceptual bridge metrics. Configure on the left, hit Run.")


# =====================================================================
# Sidebar
# =====================================================================

# Widget defaults (seeded once so the demo presets can override them by key).
_W_DEFAULTS = {"w_method": "IMM-KF (4 modes)", "w_logq": -8, "w_logr": 2,
               "w_mu": 0.10, "w_window": 200, "w_modeplants": True}
for _k, _v in _W_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

with st.sidebar:
    st.header("🚀 Demo presets")
    st.caption("One click sets the algorithm **and runs it** on the current scenario — "
               "ideal for the recorded demo.")
    if st.button("🏆 IMM-KF (v5) — proposed", use_container_width=True):
        st.session_state.update(w_method="IMM-KF (4 modes)", w_window=200, w_modeplants=True,
                                queued_run=True)
    if st.button("🪤 Quiet-Kalman trap (log₁₀Q=−12)", use_container_width=True,
                 help="The project's central finding: a fixed slow Kalman that is perfect for "
                      "quiet rooms collapses on the dynamic scenario."):
        st.session_state.update(w_method="Kalman (single mode)", w_logq=-12, w_logr=2,
                                w_modeplants=True, queued_run=True)
    if st.button("📏 NLMS baseline (µ=0.10)", use_container_width=True):
        st.session_state.update(w_method="NLMS", w_mu=0.10, w_modeplants=True, queued_run=True)
    st.markdown("---")

    st.header("⚙️ 1. Audio source")
    source_type = st.radio("Source", ["Synthetic", "Upload WAV"], horizontal=True)

    durations = mode_plants = uploaded = None
    crossfade = 0.0
    seed = 7

    if source_type == "Synthetic":
        st.markdown("**Mode trajectory** — segment durations (s)")
        durations = {
            "quiet":   st.slider("🔇 Quiet", 0, 15, 5),
            "traffic": st.slider("🚗 Traffic", 0, 15, 5),
            "wind":    st.slider("💨 Wind", 0, 15, 5),
            "babble":  st.slider("🗣️ Babble", 0, 15, 5),
        }
        mode_plants = st.checkbox("Mode-conditioned plants (dynamic test)", key="w_modeplants",
                                  help="ON ⇒ each mode also switches the acoustic plant P(z),S(z).")
        crossfade = st.slider("Crossfade between modes (s)", 0.0, 2.0, 0.0, 0.25,
                              help="Blend consecutive environments instead of hard cuts. "
                                   "0 = legacy hard switch.")
        seed = st.number_input("Random seed", 1, 9999, 7)
    else:
        uploaded = st.file_uploader("WAV (mono, resampled to 16 kHz)", type=["wav"])
        seed = st.number_input("Path seed (random P/S)", 1, 9999, 7)

    st.header("🧮 2. Algorithm")
    method = st.selectbox("Method", ["NLMS", "Kalman (single mode)", "IMM-KF (4 modes)"],
                          key="w_method")
    L = st.slider("Filter length L", 16, 256, 64, 16)

    params: dict = {}
    if method == "NLMS":
        params["mu"] = st.slider("Step size µ", 0.001, 0.5, step=0.001, format="%.3f", key="w_mu")
        filter_kind = "NLMS"
    elif method.startswith("Kalman"):
        log_q = st.slider("log₁₀(σ²_q) — process noise", -14, -4, key="w_logq")
        log_r = st.slider("log₁₀(σ²_r) — measurement noise", -2, 4, key="w_logr")
        params["sigma_q2"] = 10.0 ** log_q
        params["sigma_r2"] = 10.0 ** log_r
        filter_kind = "Kalman"
    else:
        params["window"] = st.slider("Likelihood window (samples)", 1, 1000, key="w_window")
        filter_kind = "IMM-KF"
        with st.expander("ℹ️ IMM bank — v5 calibration (config.py)"):
            st.markdown("\n".join(
                f"- **{p.name}** · σ²q = {p.sigma_q2:g} · σ²r = {p.sigma_r2:g}"
                for p in cfg.MODE_PARAMS))
            st.caption("Per-mode (Q, R) of the 4 parallel KFs — the v5 calibration used "
                       "throughout the report (+16.06 dB Monte Carlo).")

    st.markdown("---")
    st.subheader("⚡ Compute backend")
    backend_options = ["🐍 Python NumPy"]
    backend_map = {"🐍 Python NumPy": "python"}
    for lbl in c_backend.available_backends():
        ui_lbl = "⚡ Pure C" if lbl == "pure-c" else "🔧 OpenBLAS"
        backend_options.append(ui_lbl)
        backend_map[ui_lbl] = lbl
    backend_ui = st.radio("Run the filter in:", backend_options, index=0,
                          help="Python exports the IMM mode posterior (needed for that plot). "
                               "Pure C is ~9× faster, machine-eps identical residual.")
    backend_choice = backend_map[backend_ui]

    st.markdown("---")
    headphone_ab = st.checkbox("🎧 Audio A/B through virtual headphone", value=False,
                               help="ON ⇒ play the eardrum render (passive isolation + "
                                    "low-frequency-only ANC) at a shared gain, so you hear the "
                                    "loudness drop. OFF ⇒ raw d vs e, each peak-normalized.")
    run_button = st.button("▶️ Run ANC simulation", type="primary", use_container_width=True)


# =====================================================================
# Run on click — results are stored in history, rendering happens below
# from the selected history entry (so the page survives widget reruns).
# =====================================================================

def _same(a, b):
    return all(a.get(k) == b.get(k) for k in ("method", "backend", "L", "params_summary", "scenario_summary"))


# A demo-preset click queues a run with the widget values it just set.
do_run = run_button or bool(st.session_state.pop("queued_run", False))

if do_run:
    with st.spinner("🏗️ Building scenario..."):
        if source_type == "Synthetic":
            if all(d == 0 for d in durations.values()):
                st.error("Pick at least one segment with duration > 0.")
                st.stop()
            s = build_synthetic_scenario(durations, mode_plants, int(seed), float(crossfade))
        else:
            if uploaded is None:
                st.error("Upload a WAV file first.")
                st.stop()
            s = build_uploaded_scenario(uploaded.getvalue(), int(seed))

    backend_info: dict = {"label": "python", "step_us": None}
    with st.spinner(f"🧮 Running {method} via {backend_ui}..."):
        import time as _time
        if backend_choice == "python":
            filt = build_filter(filter_kind, L, params)
            _t0 = _time.perf_counter()
            result = simulate_anc(s, filt, log_mu=(filter_kind == "IMM-KF"))
            _wall = _time.perf_counter() - _t0
            backend_info["label"] = "Python NumPy"
            backend_info["step_us"] = 1e6 * _wall / max(len(s.d), 1)
        else:
            cres = c_backend.run(s, filter_kind, params, backend=backend_choice)
            result = {"e": c_backend.pick_residual(cres, filter_kind)}
            backend_info["label"] = "Pure C" if backend_choice == "pure-c" else "OpenBLAS"
            backend_info["step_us"] = c_backend.filter_step_us(cres, filter_kind)
            if filter_kind == "IMM-KF":
                py = simulate_anc(s, build_filter("IMM-KF", L, params), log_mu=True)
                result["mu_history"] = py["mu_history"]

    e = result["e"]
    nr = overall_nr_db(s.d, e)
    # Perceptual bridge metrics (reuse the shared test-bench core).
    pm = tb.compute_metrics(s, {"cur": e}, None)["cur"]

    mu_hist = result.get("mu_history")
    mode_track_pct = None
    if mu_hist is not None and source_type == "Synthetic":
        mode_track_pct = float((mu_hist.argmax(axis=1) == s.mode_labels).mean()) * 100

    if filter_kind == "NLMS":
        params_summary = f"µ={params.get('mu', 0.10):.3f}"
    elif filter_kind == "Kalman":
        params_summary = f"log₁₀Q={np.log10(params['sigma_q2']):.0f}, log₁₀R={np.log10(params['sigma_r2']):.0f}"
    else:
        params_summary = f"W={params.get('window', 200)}"

    if source_type == "Synthetic":
        nonzero = [(k, v) for k, v in durations.items() if v > 0]
        scenario_summary = "+".join(f"{m[:3]}{int(d)}" for m, d in nonzero) + f" seed={int(seed)}"
        if mode_plants:
            scenario_summary += " dyn"
        if crossfade > 0:
            scenario_summary += f" xf{crossfade:.2f}"
    else:
        scenario_summary = "uploaded WAV"

    st.session_state.run_counter += 1
    entry = {
        "id": st.session_state.run_counter, "method": method, "backend": backend_info["label"],
        "L": L, "params_summary": params_summary, "scenario_summary": scenario_summary,
        "NR_db": nr, "dba_db": pm["dba_reduction_anc"], "mode_tracking_pct": mode_track_pct,
        # Full data so the run can be re-rendered later from history:
        "pm": pm, "step_us": backend_info["step_us"], "fs": s.fs,
        "duration_sec": s.duration_sec, "source_type": source_type,
        "d": np.asarray(s.d, dtype=np.float32), "e": np.asarray(e, dtype=np.float32),
        "mu_history": None if mu_hist is None else np.asarray(mu_hist, dtype=np.float16),
        "mode_labels": np.asarray(s.mode_labels) if source_type == "Synthetic" else None,
        "wav_e": to_wav_bytes(e, s.fs),
    }

    hist = st.session_state.run_history
    if hist and _same(hist[-1], entry):
        hist[-1] = entry
    else:
        hist.append(entry)
    if len(hist) > MAX_HISTORY:
        del hist[: len(hist) - MAX_HISTORY]
    st.session_state.selected_run_id = entry["id"]


hist = st.session_state.run_history
if not hist:
    st.info("👈 Configure on the left, then click **Run**. "
            "For the listening demo, open **Music and Feel** in the sidebar.")
    st.stop()


# =====================================================================
# Run selector — switch between this session's runs
# =====================================================================

by_id = {en["id"]: en for en in hist}
ids = [en["id"] for en in reversed(hist)]  # newest first


def _fmt_run(rid: int) -> str:
    en = by_id[rid]
    trk = "" if en["mode_tracking_pct"] is None else f" · trk {en['mode_tracking_pct']:.0f}%"
    return (f"#{en['id']} · {en['method']} · {en['scenario_summary']} · "
            f"{en['params_summary']} · NR {en['NR_db']:+.1f} dB{trk}")


sel_default = st.session_state.get("selected_run_id", ids[0])
sel_index = ids.index(sel_default) if sel_default in ids else 0
if len(ids) > 1:
    chosen = st.selectbox("📂 Showing results of run", ids, index=sel_index, format_func=_fmt_run,
                          help="Switch between this session's runs — all metrics, plots and "
                               "audio below re-render for the chosen run.")
    st.session_state.selected_run_id = chosen
else:
    chosen = ids[sel_index]
entry = by_id[chosen]

# Unpack the selected run.
d = entry["d"]
e = entry["e"]
fs = entry["fs"]
nr = entry["NR_db"]
pm = entry["pm"]
run_method = entry["method"]
mu_hist = entry["mu_history"]
mode_labels = entry["mode_labels"]

st.success(f"✅ Run #{entry['id']} — {entry['duration_sec']:.1f}s @ {fs} Hz, "
           f"L={entry['L']}, method=**{run_method}**, backend **{entry['backend']}**")


# =====================================================================
# KPI cards — engineering row + perceptual row
# =====================================================================

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{nr_color(nr)} Overall NR", f"{nr:+.2f} dB",
          help="Total noise reduction. >8 great, 4–8 decent, <4 poor.")
c2.metric("🕐 Audio length", f"{entry['duration_sec']:.1f} s")
if entry["mode_tracking_pct"] is not None:
    acc = entry["mode_tracking_pct"]
    c3.metric(f"{tracking_color(acc)} Mode tracking", f"{acc:.1f}%",
              help="IMM argmax vs ground truth. Random=25%.")
else:
    c3.metric("Method", run_method)
step_us = entry["step_us"]
if step_us is not None:
    rtf = step_us / (1e6 / fs)
    c4.metric(f"⚡ {entry['backend']}", f"{step_us:.2f} µs/sample",
              f"RTF {rtf:.2f}", delta_color="inverse" if rtf > 1.0 else "normal",
              help="Per-sample latency. RTF<1 = real-time capable.")
else:
    c4.metric("Filter length L", entry["L"])

p1, p2, p3, p4 = st.columns(4)
p1.metric("🔉 Perceived drop", f"{pm['dba_reduction_anc']:+.1f} dB(A)",
          help="A-weighted loudness drop, ANC on vs passive earbud — the 'feel' headline.")
p2.metric("Low-band NR (<1.2k)", f"{pm['anc_band_nr_db']:+.1f} dB",
          help="Reduction where active ANC works (the rumble).")
p3.metric("High-band NR (>1.2k)", f"{pm['passive_band_nr_db']:+.1f} dB",
          help="Reduction above the ANC band (passive seal territory).")
p4.metric("🎵 Musical-noise", f"{pm['musical_noise']:.2f}",
          help="Spectral-kurtosis artefact index. Lower = more natural residual.")


# =====================================================================
# Run history table
# =====================================================================

if len(hist) >= 2:
    with st.expander(f"📚 Compare with previous runs ({len(hist)} saved, this session)", expanded=True):
        st.caption("▶ marks the run shown above — use the **selector at the top** to switch.")
        cols = st.columns([0.5, 1.4, 1.0, 1.4, 0.6, 0.9, 0.8, 0.8, 0.8, 2.2])
        for col, label in zip(cols, ["**#**", "**Method**", "**Backend**", "**Scenario**", "**L**",
                                     "**Params**", "**NR**", "**dBA↓**", "**Trk%**", "**Residual**"]):
            col.markdown(label)
        for en in reversed(hist):
            cols = st.columns([0.5, 1.4, 1.0, 1.4, 0.6, 0.9, 0.8, 0.8, 0.8, 2.2])
            marker = "▶" if en["id"] == entry["id"] else " "
            cols[0].write(f"{marker}{en['id']}")
            cols[1].write(en["method"]); cols[2].write(en["backend"])
            cols[3].caption(en["scenario_summary"]); cols[4].write(en["L"])
            cols[5].caption(en["params_summary"]); cols[6].write(f"{en['NR_db']:+.1f}")
            cols[7].write(f"{en['dba_db']:+.1f}")
            cols[8].write("—" if en["mode_tracking_pct"] is None else f"{en['mode_tracking_pct']:.0f}")
            cols[9].audio(en["wav_e"], format="audio/wav")
        if st.button("🧹 Clear history"):
            st.session_state.run_history = []
            st.session_state.run_counter = 0
            st.session_state.pop("selected_run_id", None)
            st.rerun()


# =====================================================================
# Audio A/B
# =====================================================================

st.subheader("🎧 Listen: original vs ANC residual")
a1, a2 = st.columns(2)
if headphone_ab:
    ren = hp.render_eardrum(d, e, music=None, fs=fs)
    g = hp.common_gain(ren)
    with a1:
        st.markdown("**🔊 Earbud, ANC OFF** — `music+H_pass·d` (passive only)")
        st.audio(hp.to_wav_bytes(ren["off"], fs, g), format="audio/wav")
    with a2:
        st.markdown(f"**🔇 Earbud, ANC ON** — {run_method} (shared gain, hear the drop)")
        st.audio(hp.to_wav_bytes(ren["on"], fs, g), format="audio/wav")
    st.caption("Virtual-headphone render at a shared gain: the ANC-on clip should be audibly quieter.")
else:
    with a1:
        st.markdown("**🔊 Original noise** `d(k)` — without ANC")
        st.audio(to_wav_bytes(d, fs), format="audio/wav")
    with a2:
        st.markdown(f"**🔇 ANC residual** `e(k)` — with {run_method}")
        st.audio(entry["wav_e"], format="audio/wav")
    st.caption("Raw d vs e, each peak-normalized (engineering view). Toggle the sidebar option "
               "for the loudness-preserving virtual-headphone render.")


# =====================================================================
# Visualization tabs
# =====================================================================

st.subheader("📊 Visualization")
tab1, tab_sg, tab2, tab3, tab4, tab_ov = st.tabs(
    ["📈 Time domain", "🌈 Spectrogram", "📉 NR over time", "🎯 Mode posteriors",
     "🎚️ NR per frequency", "🆚 Overlay runs"])

with tab1:
    t = np.arange(len(d)) / fs
    step = max(1, len(t) // 5000)
    fig, ax = plt.subplots(2, 1, figsize=(11, 4.5), sharex=True)
    ax[0].plot(t[::step], d[::step], lw=0.5, color="#d62728")
    ax[0].set_ylabel("d(k)"); ax[0].grid(True, ls=":", alpha=0.4)
    ax[0].set_title("Original noise at the eardrum (without ANC)")
    ax[1].plot(t[::step], e[::step], lw=0.5, color="#2ca02c")
    ax[1].set_ylabel("e(k)"); ax[1].set_xlabel("time [s]"); ax[1].grid(True, ls=":", alpha=0.4)
    ax[1].set_title(f"Residual with {run_method}")
    fig.tight_layout(); st.pyplot(fig)

with tab_sg:
    st.caption("Both panels share one color scale — watch the band below the dashed line "
               "(active-ANC territory) go dark when ANC is on.")
    nfft = 512
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, sharey=True,
                             constrained_layout=True)
    im_last = None
    vmin = vmax = None
    for ax, sig, title in zip(axes, [d, e],
                              ["Without ANC — d(k)", f"With {run_method} — e(k)"]):
        _, _, _, im = ax.specgram(np.asarray(sig, dtype=np.float64), NFFT=nfft, Fs=fs,
                                  noverlap=nfft // 2, cmap="magma", vmin=vmin, vmax=vmax)
        if vmin is None:  # lock the color scale to the ANC-off panel
            vmin, vmax = im.get_clim()
        ax.axhline(hp.ANC_BAND_FC, color="w", ls="--", lw=0.9, alpha=0.8)
        ax.set_ylabel("Frequency [Hz]"); ax.set_title(title, fontsize=10)
        im_last = im
    axes[1].set_xlabel("time [s]")
    fig.colorbar(im_last, ax=axes, shrink=0.85, label="PSD [dB]")
    st.pyplot(fig)

with tab2:
    win = int(0.5 * fs)
    if len(e) > win:
        nr_t = noise_reduction_db_sliding(d, e, win)
        t_nr = (np.arange(len(nr_t)) + win - 1) / fs
        fig, ax = plt.subplots(figsize=(11, 3.5))
        for lo, hi, c in [(10, 30, "green"), (5, 10, "gold"), (0, 5, "orange"), (-20, 0, "red")]:
            ax.axhspan(lo, hi, color=c, alpha=0.10)
        ax.plot(t_nr, nr_t, color="#1f77b4", lw=1)
        ax.axhline(nr, color="#1f77b4", lw=1, ls="--", label=f"mean = {nr:+.2f} dB")
        ax.set_xlabel("time [s]"); ax.set_ylabel("NR [dB]")
        ax.set_ylim(min(-5, nr_t.min() - 2), max(15, nr_t.max() + 2))
        ax.legend(loc="lower right"); ax.grid(True, ls=":", alpha=0.4); st.pyplot(fig)
    else:
        st.info("Audio too short for sliding NR.")

with tab3:
    if mu_hist is not None:
        fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        t = np.arange(len(mu_hist)) / fs
        if mode_labels is not None:
            ax[0].step(t, mode_labels, where="post", color="k", lw=1.2)
            ax[0].set_yticks(range(cfg.N_MODES)); ax[0].set_yticklabels(cfg.MODE_NAMES)
            ax[0].set_ylabel("true mode"); ax[0].set_title("Ground-truth mode trajectory")
        else:
            ax[0].text(0.5, 0.5, "No ground-truth labels for uploaded audio",
                       ha="center", va="center", transform=ax[0].transAxes); ax[0].set_axis_off()
        ax[0].grid(True, ls=":", alpha=0.4)
        mu = np.asarray(mu_hist, dtype=np.float32)
        for j, name in enumerate(cfg.MODE_NAMES):
            ax[1].plot(t, mu[:, j], label=name, lw=0.9)
        ax[1].set_ylabel(r"$\mu_j(k)$"); ax[1].set_xlabel("time [s]"); ax[1].set_ylim(-0.05, 1.05)
        ax[1].set_title("IMM mode posteriors over time"); ax[1].legend(loc="upper right", ncol=2)
        ax[1].grid(True, ls=":", alpha=0.4); fig.tight_layout(); st.pyplot(fig)
    else:
        st.info("ℹ️ Mode posteriors are only available for the IMM-KF method (Python backend).")

with tab4:
    st.caption("Per-frequency noise reduction (1/3-octave). Active ANC dominates below the dashed "
               "line; above it the passive seal does the work.")
    fc, nrf = pc.third_octave_nr(d, e, fs, f_lo=40.0, f_hi=min(8000.0, fs / 2))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.semilogx(fc, nrf, "-o", ms=3, color="#1f77b4")
    ax.axvline(hp.ANC_BAND_FC, color="grey", ls="--", lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("NR [dB]")
    ax.grid(True, which="both", ls=":", alpha=0.4); fig.tight_layout(); st.pyplot(fig)

with tab_ov:
    if len(hist) < 2:
        st.info("Run at least two configurations — their NR-over-time curves will overlay "
                "here for a direct comparison (e.g. IMM vs the quiet-Kalman trap).")
    else:
        sel_runs = st.multiselect(
            "Runs to overlay", ids, default=ids, format_func=_fmt_run,
            help="Sliding 0.5 s noise reduction of each run on one axis. "
                 "The run selected at the top is drawn thicker.")
        fig, ax = plt.subplots(figsize=(11, 4))
        for lo, hi, c in [(10, 30, "green"), (5, 10, "gold"), (0, 5, "orange"), (-20, 0, "red")]:
            ax.axhspan(lo, hi, color=c, alpha=0.08)
        for rid in sorted(sel_runs):
            en = by_id[rid]
            if "nr_curve" not in en:
                w = int(0.5 * en["fs"])
                if len(en["e"]) <= w:
                    continue
                nr_t = noise_reduction_db_sliding(en["d"], en["e"], w)
                t_nr = (np.arange(len(nr_t)) + w - 1) / en["fs"]
                en["nr_curve"] = (t_nr.astype(np.float32), nr_t.astype(np.float32))
            t_nr, nr_t = en["nr_curve"]
            ax.plot(t_nr, nr_t, lw=2.2 if rid == entry["id"] else 1.1,
                    label=f"#{en['id']} {en['method']} · {en['params_summary']} "
                          f"({en['NR_db']:+.1f} dB)")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("time [s]"); ax.set_ylabel("NR [dB]")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, ls=":", alpha=0.4); fig.tight_layout(); st.pyplot(fig)
        st.caption("Curves come straight from history — no re-simulation. Runs on different "
                    "scenarios share the time axis but are not directly comparable.")
