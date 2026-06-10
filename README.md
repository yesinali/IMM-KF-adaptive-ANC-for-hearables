# Adaptive Active Noise Cancellation for Hearables using Interacting Multiple Model (IMM) Filters

> A Bayesian adaptive active-noise-cancellation (ANC) controller that runs **four
> parallel Kalman filters** — one per acoustic environment — and blends them with
> an Interacting Multiple Model (IMM) posterior, so the controller automatically
> adopts the right adaptation speed for whatever the wearer walks into.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)

EE4084 course project, Marmara University. Pure simulation — no hardware required.

---

## Overview

An ANC headphone plays an inverted "anti-noise" wave to cancel ambient noise at the
eardrum. The hard part is the **adaptive filter** that decides, sample by sample,
exactly what anti-noise to play. Classical controllers (NLMS, single-mode Kalman)
are governed by **one** speed setting:

- **Slow** — precise in a quiet room, but can't keep up with sudden noise (wind, traffic).
- **Fast** — tracks transitions, but adds residual hiss when it's quiet.

No single setting is optimal across environments, yet a real wearer moves between them
within seconds.

**This project's answer is the FxIMM-KF**: model the adaptive FIR coefficients as the
hidden state of a random-walk Kalman filter, run **four** such filters tuned for
`quiet` / `babble` / `traffic` / `wind`, and combine them every sample with a Bayesian
mode posterior. The controller never needs to be told which environment it's in.

Two things make this repo more than a number-cruncher:

1. **It's testable without a headphone.** A virtual-headphone signal chain renders the
   eardrum signal (passive isolation + low-frequency-only active ANC) so the output
   *sounds like* a real earbud.
2. **You can hear it, not just measure it.** A loudness ladder written at a single shared
   gain lets you actually feel the world get quieter — the thing a dB number can't convey.

---

## Highlights

- **FxIMM-KF controller** — 4-mode Interacting Multiple Model bank of linear Kalman
  filters over the filtered-x ANC model. Vectorized NumPy implementation in
  [`src/imm.py`](src/imm.py).
- **Bit-exact C port** — the NLMS / KF / IMM filters reimplemented in C
  ([`c_imm/`](c_imm/)), machine-epsilon identical to NumPy, **~8.8× faster**
  (56 µs/sample vs 495 µs at L=64 → comfortably real-time at 16 kHz).
- **Virtual-headphone test bench** — passive isolation + low-frequency-only active ANC,
  rendered at a shared gain ([`src/headphone.py`](src/headphone.py),
  [`scripts/20_render_testbench.py`](scripts/20_render_testbench.py)).
- **Perceptual metrics** that bridge numbers and feel — A-weighted (dBA) loudness drop,
  band-split & 1/3-octave NR, and a musical-noise / transient index
  ([`src/perceptual.py`](src/perceptual.py)).
- **Unified interactive app** — a multipage Streamlit demo: a *Performance Lab* (one-click
  demo presets, session run history with an instant run selector, overlay comparison of
  NR curves across runs, spectrogram / mode-posterior / per-frequency plots) and a
  *Music & Feel* page (hear the loudness ladder and a music A/B), [`app/`](app/).
- **Reproducible evaluation** — Monte Carlo over random acoustic plants:
  **+16.06 ± 1.77 dB** noise reduction (N = 15).

---

## Repository structure

```
adaptive-anc-imm-kf/
├── src/                        numerical core
│   ├── config.py               sample rate, FIR lengths, per-mode (Q, R) table
│   ├── noise.py                synthetic per-mode noise generators
│   ├── noise_recorded.py       drop-in real-recording noise source
│   ├── paths.py                random primary / secondary acoustic FIRs
│   ├── scenario.py             scenario assembler (+ crossfade) + Wiener solver
│   ├── filters.py              NLMS + single-mode Kalman ANC filters
│   ├── imm.py                  vectorized IMM-Kalman bank
│   ├── anc.py                  sample-by-sample simulation driver
│   ├── metrics.py              NR, misalignment, NEES/NIS consistency
│   ├── perceptual.py           dBA loudness, band-split NR, musical-noise index, SI-SDR
│   ├── headphone.py            virtual-headphone render (passive + LF-ANC, shared gain)
│   ├── testbench.py            shared scenario→algorithm→metrics orchestration
│   └── c_backend.py            wrapper around the C-port binaries
├── scripts/                    runnable experiments & the test-bench renderer
├── app/
│   ├── streamlit_app.py        unified demo — home / launcher
│   └── pages/                  Performance Lab + Music & Feel
├── c_imm/                      C port of the filter bank (sources + Makefile)
├── noise_samples/              drop real ambient recordings here (see its README)
├── musics/                     drop your own program music here (see its README)
├── docs/USER_GUIDE.md          step-by-step quick-start guide
├── report/                     IEEE-style project report (LaTeX sources + PDF)
└── requirements.txt
```

---

## Installation

### Prerequisites
- **Python ≥ 3.10** (required)
- A C compiler + `make` (**optional** — only for the faster C backend; the project runs
  fully in Python without it)

### 1. Python (required)

```bash
# from the repository root
python -m venv .venv

# activate it:
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\Activate.ps1         # Windows PowerShell

pip install -r requirements.txt
```

Installs NumPy, SciPy, Matplotlib, SoundFile, Streamlit and pandas. That's all you need
to run every script and the interactive app.

### 2. C port (optional — for real-time-speed runs)

The C port makes the IMM filter ~9× faster. If you skip it, `src/c_backend.py` simply
reports no backend and everything falls back to NumPy automatically.

**Windows** — install [MSYS2](https://www.msys2.org/), then in an MSYS2 MinGW-64 shell:

```bash
pacman -S mingw-w64-x86_64-gcc make            # compiler + make
pacman -S mingw-w64-x86_64-openblas            # optional, only for the BLAS build
cd c_imm
make pure                                      # builds imm_pure.exe (no extra deps)
make blas                                      # optional: imm_blas.exe (needs OpenBLAS)
```

**Linux / macOS** — `gcc`/`clang` + `make` are enough:

```bash
cd c_imm
make pure                                      # builds imm_pure.exe (portable scalar C)
make blas                                      # optional, needs libopenblas
```

> The build targets keep the `.exe` suffix on every platform because
> `src/c_backend.py` looks for `c_imm/imm_pure.exe` / `imm_blas.exe`. See
> [`c_imm/README.md`](c_imm/README.md) for the binary I/O format and verification steps.

---

## Usage

### Interactive demo (recommended starting point)

```bash
streamlit run app/streamlit_app.py
```

Opens a multipage app in your browser:

- **🎛️ Performance Lab** — pick a scenario + algorithm + backend, run the controller live,
  and inspect noise reduction, mode tracking, real-time factor, per-frequency NR, and the
  perceptual bridge metrics. Three **demo presets** (proposed IMM-v5 / quiet-Kalman trap /
  NLMS baseline) configure and run with one click; every run is kept in a session
  **history** so you can flip between results instantly and **overlay** their NR-over-time
  curves on one axis — the fastest way to *see* the IMM-vs-fixed-filter gap.
- **🎵 Music & Feel** — render live and *listen*: the loudness ladder (open ear → ANC off →
  ANC on, at a shared gain) and a music A/B that shows whether the controller keeps the
  music intact while removing the noise.

A friendly step-by-step walkthrough is in [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).

### Render the full test bench (files + metrics + plots)

```bash
python -m scripts.20_render_testbench                 # needs a music file (see below)
python -m scripts.20_render_testbench --no-music      # noise-only loudness ladder
```

Writes a coherent WAV set, a metric table, `metrics.json` and plots into
`figures/testbench/`.

### Selected scripts

| Command | What it does |
|---|---|
| `python -m scripts.01_inspect_paths` | Plot the primary/secondary acoustic paths |
| `python -m scripts.02_inspect_scenario` | Build a scenario + dump a WAV + plots |
| `python -m scripts.03_baseline_run` | Static NLMS / KF baselines |
| `python -m scripts.05_dynamic_imm` | Dynamic-plant IMM run |
| `python -m scripts.06_monte_carlo --runs 15` | Monte Carlo evaluation |
| `python -m scripts.07_consistency_test` | NEES/NIS filter-consistency check |
| `python -m scripts.full_metrics` | Full metric battery + NR(f) plots |
| `python -m scripts.c_port_compare` | Verify C port vs NumPy + time it |

All script outputs land in `figures/` (git-ignored).

### Bring your own data

The repository ships **no audio** (copyright + size). To use real material:

- **Music** — drop a `.flac`/`.wav` into [`musics/`](musics/). The Music & Feel page falls
  back to an upload widget if the folder is empty.
- **Recorded noise** — drop clips into `noise_samples/<mode>/` and run with
  `ANC_NOISE_SOURCE=recorded`. A DEMAND / ESC-50 mapping is in
  [`noise_samples/README.md`](noise_samples/README.md). Modes with no clips fall back to the
  synthetic generators.

---

## How it works (in brief)

- **State = the adaptive FIR coefficients.** Treat the controller weights `w_k` as the
  hidden state of a random-walk Kalman filter: `w_k = w_{k-1} + q`, observed through the
  *filtered-x* model `d(k) = xfᵀ(k) w_k + v`. The observation is linear in `w_k`, so a plain
  **linear** Kalman filter suffices — no EKF/UKF needed.
- **Four modes, one posterior.** Each environment gets its own `(Q, R)` tuning. The IMM
  *mixes* the four filters' states every sample, runs a Kalman update per mode, and updates a
  Markov mode posterior from each mode's innovation likelihood. The combined estimate is the
  posterior-weighted blend.
- **Virtual headphone.** For listening, the simulator's residual `e` (full-band) is mapped to
  a realistic eardrum signal: active cancellation kept only in the low band
  (`highpass(d) + lowpass(e)`) plus a passive-isolation high-shelf cut — then the whole
  comparison set is written at **one shared gain** so the loudness drop is audible.

---

## Results

- **Monte Carlo (N = 15, random plants):** IMM-KF reaches **+16.06 ± 1.77 dB** noise
  reduction — the highest mean *and* the tightest variance among the methods tried.
- **The quiet-Kalman trap:** a fixed slow Kalman that wins on a static plant collapses to
  ~+2 dB once the acoustic environment changes — the project's central finding that a single
  fixed setting fails in non-stationary use.
- **Numbers vs perception:** the highest-NR controller is not always the most natural to the
  ear (aggressive modes can introduce musical-noise artefacts). The perceptual metrics and
  the Music & Feel page make this trade-off audible and measurable.

---

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

Developed for **EE4084**, Marmara University. Synthetic noise/path generators stand in for the
DEMAND and ESC-50 environmental-audio datasets.
