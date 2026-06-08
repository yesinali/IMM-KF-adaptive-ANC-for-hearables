"""IMM-KF ANC — unified demo (Home).

Multipage Streamlit app. This entry page is the landing/overview; the actual
demos live in the sidebar pages:

    Performance Lab  — live interactive simulation: pick a scenario + algorithm,
                       run it, inspect NR / mode-posterior / per-frequency plots
                       and the engineering metrics. The "pure ANC performance" side.
    Music and Feel   — virtual-headphone listening demo: hear the loudness ladder
                       (open ear -> ANC off -> ANC on) and music A/B, rendered live
                       with the passive-isolation + low-frequency-ANC chain. The
                       "what it feels like" side.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="IMM-KF ANC Demo", page_icon="🎧", layout="wide")

st.title("🎧 Adaptive Active Noise Cancellation for Hearables")
st.caption("EE4084 — IMM-KF demo. Use the sidebar to open a page: **Performance Lab** "
           "(numbers & plots) or **Music and Feel** (listen to it).")

col_a, col_b = st.columns([2, 1])
with col_a:
    st.info(
        "👈 **Pick a page in the sidebar.**\n\n"
        "- **🎛️ Performance Lab** — run the controller live, see NR / mode tracking / "
        "per-frequency reduction and engineering metrics.\n"
        "- **🎵 Music and Feel** — hear what a real ANC earbud would deliver: the "
        "loudness ladder and a music A/B, rendered through the virtual-headphone chain."
    )
with col_b:
    st.metric("Project NR (Monte Carlo, N=15)", "+16.06 dB", "std: 1.77 dB")

st.markdown("---")

with st.expander("📖 **How does this demo work?**", expanded=True):
    st.markdown("""
    ### The big picture
    In an Active-Noise-Cancellation (**ANC**) headphone, a microphone listens to the surrounding
    noise and the speaker plays an inverted "anti-noise" wave to cancel it at your eardrum. The hard
    part is the **adaptive filter** that decides exactly what anti-noise to play, sample by sample.

    ### The problem this project addresses
    Classical adaptive filters (NLMS) and single-mode Kalman filters have **one** speed setting:
    - **Slow**: great in quiet rooms, can't keep up with sudden noise (wind, traffic).
    - **Fast**: tracks transitions but adds residual noise in quiet moments.

    No single setting is optimal across all environments.

    ### Our proposal: IMM-KF
    The **Interacting Multiple Model Kalman Filter** runs **4 parallel Kalman filters** — one per
    `quiet` / `babble` / `traffic` / `wind` — and blends them with a Bayesian posterior over which
    mode is active, so the controller automatically adopts the right speed for the environment.

    ### Testing it without a real headphone
    We can't ship to a physical earbud, so we evaluate two ways:
    1. **Numbers** (Performance Lab) — noise reduction (dB), per-frequency reduction, mode tracking,
       real-time factor.
    2. **Feel** (Music and Feel) — render the eardrum signal through a *virtual headphone* (passive
       isolation + low-frequency-only active ANC) and write the before/after at a *shared gain*, so
       you actually hear the world get quieter — the thing a metric can't convey.
    """)

with st.expander("🧪 **Suggested experiments**"):
    st.markdown("""
    1. **Performance Lab → default IMM run.** Press *Run* with defaults → ~15-18 dB reduction on the
       dynamic-plant scenario (Monte Carlo N=15: +16.06 ± 1.77 dB).
    2. **The "quiet Kalman trap".** Performance Lab → `Kalman (single mode)`, `log_q = -12`,
       *Mode-conditioned plants* ON → only ~2 dB NR. The project's central finding: a fixed slow
       setting **fails** in non-stationary environments.
    3. **Music and Feel → loudness ladder.** Listen to *open ear → ANC off → ANC on* in order. The
       low rumble vanishes under ANC while the passive seal handles the highs.
    4. **Music and Feel → algorithm A/B.** Switch between IMM v5 and NLMS under music. IMM removes
       more noise but listen for musical-noise artefacts — the algo-vs-perception story.
    """)

st.markdown("---")
st.caption(
    "_Per-sample latency at L=64: Python 495 µs, pure C 56 µs (8.8× speedup, real-time at 16 kHz). "
    "See `figures/08_c_port_speed.png`. The Music and Feel page renders live via the C backend when "
    "available, Python otherwise._"
)
