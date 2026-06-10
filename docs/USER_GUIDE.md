# User Guide — IMM-KF Adaptive ANC Demo

This guide walks you through running the project's **interactive demo** from scratch on your own machine. It is written step by step so you can follow it even without a programming background.

---

## 1. What does this app do?

In short: **a PC simulation of a headphone's noise-cancellation (ANC) algorithm.**

The environment around a headphone changes constantly (quiet room → traffic → wind → crowd). Classical single-setting filters either stay too conservative to react, or run too aggressively and inject noise of their own. This project implements **IMM-KF**, a Bayesian method — a smart filter that runs 4 different tunings in parallel and **blends them automatically**. In the demo you compare this algorithm against classical methods and **hear the difference with your own ears.**

What you can do in the demo:
- Generate a synthetic noise scenario (quiet/traffic/wind/babble)
- Or upload your own WAV file
- Pick one of 3 algorithms (NLMS / Kalman / IMM)
- Tweak the parameters with sliders
- Listen to the original noise and the post-ANC residual **side by side**
- Inspect the performance in plots

---

## 2. System requirements

| Requirement | Detail |
|---|---|
| **Operating system** | Windows 10/11, macOS, or Linux (all work) |
| **Python** | 3.10 or newer |
| **Disk** | ~500 MB (including Python packages) |
| **RAM** | 4 GB is plenty |
| **Browser** | Chrome, Edge, Firefox or Safari (current version) |
| **Internet** | Only needed during installation (to download packages) |

**Important:** you need speakers or headphones to listen to the audio.

---

## 3. Installation (one-time, ~5 minutes)

### Step 3.1 — Check whether Python is installed

Open PowerShell or CMD (Windows key → type "powershell" → enter) and run:

```powershell
python --version
```

**If you see "Python 3.10.x" or newer:** skip to Step 3.3.

**If you see "command not found" or nothing at all:** go to Step 3.2.

### Step 3.2 — Install Python (if missing)

1. Go to https://www.python.org/downloads/
2. Click the yellow **"Download Python 3.12"** button (or the latest version)
3. Run the downloaded `.exe`
4. **VERY IMPORTANT:** on the first screen, tick **"Add Python to PATH"** (otherwise the commands below won't work)
5. Click "Install Now" and wait
6. When it finishes, close and reopen PowerShell (so the PATH change takes effect)
7. Repeat Step 3.1 — Python should now be found

### Step 3.3 — Get the project folder

Depending on how you received the project:
- **As a ZIP:** extract it somewhere (e.g. `Documents\anc_project\`)
- **From GitHub:** use the folder you got with `git clone <URL>`

The folder should contain subfolders like `src/`, `scripts/`, `app/`.

### Step 3.4 — Go to the project folder (in the terminal)

In PowerShell, change into the project folder:

```powershell
cd "C:\Users\YOUR_USERNAME\Documents\anc_project"
```

(Adjust the path to your setup. `cd` means "change directory".)

To verify you're in the right place:

```powershell
ls
```

You should see things like: `src`, `scripts`, `app`, `requirements.txt`, `README.md`...

### Step 3.5 — Install the required packages

```powershell
pip install -r requirements.txt
```

This takes about **1–3 minutes** and installs:
- NumPy (numerical computation)
- SciPy (signal processing)
- Matplotlib (plots)
- Soundfile (audio files)
- Streamlit (the UI)

If it ends with "Successfully installed..." you're done.

**If you get an error:** see Section 7 (Troubleshooting).

---

## 4. Running the demo

### Step 4.1 — Start Streamlit

Still inside the project folder, run:

```powershell
streamlit run app/streamlit_app.py
```

The first run may ask for an e-mail — just press **enter** to skip (it's optional).

The terminal will then show something like:

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

### Step 4.2 — Open the browser

**If it opens automatically:** great, you're set.

**If not:** open Chrome / Edge / Firefox and type **`http://localhost:8501`** into the address bar.

The demo interface will appear.

### Step 4.3 — Stopping the demo

Press **Ctrl + C** in the terminal window. The server shuts down.

---

## 5. Interface guide

The interface has 2 main areas:

### 5.1 Left panel (sidebar) — settings

#### 🚀 Demo presets (at the top — the shortcut)
Three ready-made buttons: **IMM-KF (v5)**, **Quiet-Kalman trap**, **NLMS baseline**. Clicking
one configures the algorithm + parameters and **starts the simulation automatically** — start
here if you don't want to set everything below by hand.

#### Audio source
- **Synthetic:** built-in synthetic scenario
- **Upload WAV:** use your own audio file

#### When Synthetic is selected:
- **Quiet / Traffic / Wind / Babble sliders:** duration (seconds) of each noise type. E.g. quiet=5, traffic=10 gives 5 s of silence followed by 10 s of traffic noise.
- **Mode-conditioned plants:** leave it checked — the harder (dynamic) test. It simulates the acoustic environment of the earbud changing along with each noise type.
- **Random seed:** use the same number to repeat the same random test; change it (1–9999) for different scenarios.

#### When Upload WAV is selected:
- Drag your WAV in or pick it via "Browse files". It is converted to 16 kHz automatically.

#### Method — 3 options:
- **NLMS:** the classical adaptive filter, industry standard
- **Kalman (single mode):** single-mode Bayesian filter, fixed tuning
- **IMM-KF (4 modes):** the smart Bayesian filter this project proposes

#### Filter length L:
FIR filter length. **Leave it at 64** — no need to change.

#### Method-specific setting:
- **If NLMS:** `µ` (step size, adaptation speed). Try 0.10 to start.
- **If Kalman:** `log10(σ_q²)` and `log10(σ_r²)` — Q and R expressed as powers of 10. Slow Kalman: Q=-12, R=0. Fast Kalman: Q=-5, R=2.
- **If IMM:** `Likelihood window` — temporal smoothing of the mode decision. **Leave at 200.** An expander below shows the v5 per-mode (Q, R) calibration used in the report.

#### Compute backend
- **Python NumPy:** the default; required for the IMM mode-posterior plot.
- **Pure C / OpenBLAS:** appear if the C port is compiled (see the README installation
  section). ~9× faster, bit-identical results.

#### Run button
The big blue button — press it after configuring things by hand (not needed if you used a preset).

### 5.2 Main panel — results

After a run finishes (5 seconds to 3 minutes depending on algorithm and backend):

#### Run selector and history
**Every run in the session is kept** (up to 8). Pick an older run from the
**"📂 Showing results of run"** menu above the results and all cards, audio and plots
switch to that run **instantly** — nothing is recomputed. The
**"📚 Compare with previous runs"** expander shows the runs side by side (each row
includes an audio player for the residual).

#### Top area: 2 rows of KPI cards
- **Overall NR:** total noise reduction (dB). **Higher is better.**
- **Audio length / Mode tracking / backend speed (µs-sample, RTF):** run info
- Second row, **perceptual metrics:** perceived loudness drop in dB(A), low/high-band NR,
  musical-noise index (lower = more natural residual)

#### Audio comparison
**Two audio players:**
- **Left:** the original noise d(k) — what reaches your ear with ANC off
- **Right:** the residual e(k) — what reaches your ear with ANC on

**Listen to both in turn** — can you hear the difference? This is the best part of the demo.
(Tick the "virtual headphone" box in the sidebar to hear the loudness-preserving
headphone-simulation version instead.)

#### Visualization — 6 tabs:
- **📈 Time domain:** waveforms (original on top, post-ANC below). Expect the amplitude to shrink.
- **🌈 Spectrogram:** before/after spectrograms on a shared color scale — the low band goes dark when ANC is on.
- **📉 NR over time:** NR through time (dB plot). Higher = better.
- **🎯 Mode posteriors:** meaningful for IMM only. Shows which mode the IMM picks at every instant.
- **🎚️ NR per frequency:** reduction in 1/3-octave bands — active ANC works at low frequencies.
- **🆚 Overlay runs:** NR curves of past runs **on one axis** — the fastest way to see the
  IMM-vs-fixed-filter gap in a single plot.

---

## 6. Suggested experiments

Try these in order and compare the results:

### Experiment 1 — IMM's baseline performance
1. Select Synthetic, keep the defaults (5 s per mode)
2. Keep mode-conditioned plants **checked**
3. Method: **IMM-KF** (or just press the 🏆 preset)
4. Run (~30 s)
5. **Listen to the audio A/B, check the NR**
6. Open the Mode posteriors tab — how did the IMM track the modes?

### Experiment 2 — IMM vs classical NLMS
1. Same scenario, same seed (e.g. 7)
2. Method: **NLMS**, µ = 0.10
3. Run, then compare NR and audio
4. Is the IMM's NR higher than the classical method's? **Expected: yes.**

### Experiment 3 — the slow-Kalman collapse (the project's central finding)
1. Same scenario (keep mode-conditioned plants **on** — this is critical)
2. Press the **🪤 Quiet-Kalman trap** preset at the top of the sidebar (or set it up
   manually: Kalman (single mode), `log_q = -12`, `log_r = 2`)
3. Expectation: **very low NR** (e.g. +2 dB), because the slow filter cannot adapt to environment changes
4. Listen — the noise will sound almost untouched
5. Open the **🆚 Overlay runs** tab: this run and Experiment 1's IMM curve share one
   axis — the gap is visible at a glance

This experiment confirms the project's main thesis: "a single fixed tuning fails in changing environments."

### Experiment 4 — test with your own audio
1. Record 5–10 seconds of noise with your phone (traffic, café, etc.)
2. Save as WAV and copy it to your computer
3. Select "Upload WAV" in the demo and upload it
4. Pick IMM, Run
5. How did the algorithm do on *your* noise?

---

## 7. Troubleshooting

### "streamlit: command not found"
Streamlit is installed but not on PATH. Try:
```powershell
python -m streamlit run app/streamlit_app.py
```

### "ModuleNotFoundError: No module named 'XYZ'"
Packages didn't install cleanly. Reinstall:
```powershell
pip install -r requirements.txt --upgrade
```

### "pip: command not found"
Your Python install is old or broken. Redo Step 3.2 and make sure **"Add Python to PATH"** is ticked.

### Streamlit opens but shows "Network error"
Antivirus or firewall may be blocking it. Disable temporarily to test; add an exception if that fixes it.

### The browser opens but Run throws an error
Check the terminal — the Python traceback there tells you what went wrong.

### The demo runs but is very slow
IMM-KF is slow in pure Python. For faster results:
- Shorten the scenario segments (3–5 s each is enough)
- Or pick NLMS / Kalman (much faster)
- Or compile the C port and select the **Pure C** backend (see the README)

### No sound / can't listen
- Click the audio player and press play
- If the browser asks for permission, allow it
- Check your system audio output

### Port already in use (port 8501)
A previous Streamlit server is still running. Go to its terminal and press Ctrl+C. Or:
```powershell
streamlit run app/streamlit_app.py --server.port 8502
```
Then open `http://localhost:8502`.

---

## 8. Bonus: running the scripts from the command line

Outside the demo, you can drive the project from the command line:

```powershell
python -m scripts.01_inspect_paths       # acoustic path plots
python -m scripts.02_inspect_scenario    # scenario visualization + WAV
python -m scripts.03_baseline_run        # static baseline comparison
python -m scripts.05_dynamic_imm         # dynamic IMM test
python -m scripts.06_monte_carlo --runs 5  # Monte Carlo (5 runs, ~10 min)
```

Outputs are saved into the `figures/` folder.

---

## 9. Quick reference

| What do you want? | Command / action |
|---|---|
| Check Python | `python --version` |
| Install packages | `pip install -r requirements.txt` |
| Start the demo | `streamlit run app/streamlit_app.py` |
| Stop the demo | Ctrl+C in the terminal |
| Open in browser | `http://localhost:8501` |
| Refresh the page without restarting | F5 in the browser |

---

## Contact

For the technical documentation see [README.md](../README.md); for the methodology and
results see the report under [`report/`](../report/).

If you get stuck on a step, note which step and the exact error — the full terminal
output helps a lot.

Happy experimenting!
