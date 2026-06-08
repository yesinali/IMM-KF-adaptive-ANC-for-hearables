"""Comprehensive metric comparison of all candidate ANC algorithms.

Metrics computed for each method:
  1. NR(f)        : frequency-dependent noise reduction (1/3-octave smoothed)
  2. Pe           : residual noise power = mean(e²)
  3. MSE          : mean squared error vs. zero target (identical to Pe here)
  4. Convergence  : time after each mode boundary to reach 90% of segment NR
  5. Misadjustment: (J_ss - J_min)/J_min, where J_min = power of Wiener
                    optimum residual e_wiener(n) = d - w_opt(n)^T xf(n)
  6. ΔSNR         : in-band SNR change.  We don't have a separate
                    signal-of-interest, so we use the speech band (300-3 kHz)
                    against out-of-band as a proxy: ΔSNR = NR_in - NR_out.
  7. Insertion Loss IL = 10 log10(Pe_off / Pe_on)  (overall NR)
  8. Frequency response of cancellation: |E(f)|/|D(f)| line plot

Methods compared:
  - NLMS µ=0.10
  - KF wind-tuned (single best fixed Q from MC)
  - IMM v5
  - IMM v6 (A: capped Q)
  - IMM v5 + ŵ smoothing (C, α=0.02)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src import scenario as sc
from src.filters import NLMSFilter, KalmanANCFilter
from src.imm import IMMKalmanANC
from src.metrics import overall_nr_db, noise_reduction_db_sliding


TRAJ = (
    sc.ScenarioSegment("quiet",   5.0),
    sc.ScenarioSegment("traffic", 5.0),
    sc.ScenarioSegment("wind",    5.0),
    sc.ScenarioSegment("babble",  5.0),
)
SEG_STARTS = [0.0, 5.0, 10.0, 15.0]
SEG_DUR = 5.0
FS = cfg.FS


V6_MODE_PARAMS = (
    cfg.ModeParams("quiet",   sigma_q2=1e-10, sigma_r2=100.0),
    cfg.ModeParams("babble",  sigma_q2=1e-6,  sigma_r2=10.0),
    cfg.ModeParams("traffic", sigma_q2=1e-10, sigma_r2=10.0),
    cfg.ModeParams("wind",    sigma_q2=1e-6,  sigma_r2=100.0),
)


# ----- Runner ----------------------------------------------------------------

def run_filter(scenario, filt, alpha_w_smooth: float = 0.0) -> np.ndarray:
    xf = scenario.x_filt; d = scenario.d
    N = len(d); L = filt.L
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


# ----- Metrics ---------------------------------------------------------------

def freq_nr(d: np.ndarray, e: np.ndarray, fs: int,
            n_octave_smooth: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """NR(f) = 10 log10(|D(f)|^2 / |E(f)|^2), smoothed to ~1/3-octave bins."""
    D = np.abs(np.fft.rfft(d))
    E = np.abs(np.fft.rfft(e))
    freqs = np.fft.rfftfreq(len(d), d=1.0/fs)
    P_d = D * D + 1e-30
    P_e = E * E + 1e-30
    nr = 10.0 * np.log10(P_d / P_e)
    # Smooth on a log-frequency grid: each output bin averages a window of
    # n_octave_smooth raw FFT bins. Crude 1/3-octave proxy.
    k = max(1, n_octave_smooth)
    csum = np.cumsum(np.insert(nr, 0, 0.0))
    smooth = (csum[k:] - csum[:-k]) / k
    f_smooth = freqs[k//2 : k//2 + len(smooth)]
    return f_smooth, smooth


def residual_power(e: np.ndarray) -> float:
    return float(np.mean(e * e))


def insertion_loss_db(d: np.ndarray, e: np.ndarray) -> float:
    return 10.0 * np.log10(np.mean(d*d) / max(np.mean(e*e), 1e-30))


def wiener_jmin(scenario) -> tuple[float, np.ndarray]:
    """Use the per-sample Wiener weights as 'optimal' and compute the
    irreducible residual power J_min = mean((d - w_opt^T xf)^2)."""
    w_opt_per_mode = sc.wiener_weights_per_mode(scenario, L=cfg.L)
    w_opt_arr = sc.per_sample_wiener_array(scenario, w_opt_per_mode)
    N = len(scenario.d)
    L = cfg.L
    buf = np.zeros(L)
    e_w = np.zeros(N)
    for k in range(N):
        buf[1:] = buf[:-1]; buf[0] = scenario.x_filt[k]
        e_w[k] = scenario.d[k] - w_opt_arr[k] @ buf
    return float(np.mean(e_w * e_w)), e_w


def misadjustment(e: np.ndarray, j_min: float) -> float:
    """Per-segment steady-state J_ss / J_min minus 1, averaged over segments.
    Steady-state = last 60% of each segment (skip the first 2 s transient)."""
    M = 0.0
    n_seg = 0
    for start in SEG_STARTS:
        s0 = int((start + 2.0) * FS)
        s1 = int((start + SEG_DUR) * FS)
        if s1 > len(e): s1 = len(e)
        if s0 >= s1: continue
        j_ss = float(np.mean(e[s0:s1] ** 2))
        M += (j_ss - j_min) / max(j_min, 1e-30)
        n_seg += 1
    return M / max(n_seg, 1)


def convergence_time_ms(d: np.ndarray, e: np.ndarray, fs: int) -> dict:
    """For each mode boundary, find how long after the switch the sliding NR
    reaches 90% of the asymptote NR. Returns dict per segment."""
    win = int(0.10 * fs)        # 100 ms sliding window
    nr_t = noise_reduction_db_sliding(d, e, win)
    t_ax = (np.arange(len(nr_t)) + win // 2) / fs
    out = {}
    for seg_idx, start in enumerate(SEG_STARTS):
        s0 = start
        s1 = start + SEG_DUR
        # Asymptote = mean NR over the last 3 s of this segment
        a_mask = (t_ax >= s0 + 2.0) & (t_ax < s1)
        if not a_mask.any():
            out[seg_idx] = float("nan"); continue
        asymp = float(np.mean(nr_t[a_mask]))
        target = 0.90 * asymp
        # Look from segment start onward
        c_mask = (t_ax >= s0) & (t_ax < s1)
        idx = np.where(c_mask & (nr_t >= target))[0]
        if len(idx) == 0:
            out[seg_idx] = float("nan"); continue
        t_reach = t_ax[idx[0]]
        out[seg_idx] = 1000.0 * max(t_reach - s0, 0.0)
    return out


def speech_band_dsnr(d: np.ndarray, e: np.ndarray, fs: int) -> float:
    """ΔSNR proxy: NR in speech band (300-3 kHz) minus NR out of band."""
    D = np.fft.rfft(d); E = np.fft.rfft(e)
    f = np.fft.rfftfreq(len(d), d=1.0/fs)
    in_band = (f >= 300) & (f < 3000)
    out_band = (f >= 50) & (~in_band) & (f < 8000)
    pd_in  = float(np.sum(np.abs(D[in_band])**2))
    pe_in  = float(np.sum(np.abs(E[in_band])**2))
    pd_out = float(np.sum(np.abs(D[out_band])**2))
    pe_out = float(np.sum(np.abs(E[out_band])**2))
    nr_in  = 10*np.log10(pd_in / max(pe_in,  1e-30))
    nr_out = 10*np.log10(pd_out / max(pe_out, 1e-30))
    return nr_in - nr_out


# ----- Main ------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(7)
    s = sc.build_scenario(segments=TRAJ, rng=rng, mode_conditioned_plants=True)

    # The single fixed-Q KF benchmark — use the wind-tuned one because that
    # was the best fixed KF in the MC.
    wind = next(p for p in cfg.MODE_PARAMS if p.name == "wind")

    print("Running filters...")
    runs: dict[str, np.ndarray] = {}
    runs["NLMS µ=0.10"]                  = run_filter(s, NLMSFilter(L=64, mu=0.10))
    runs["KF wind-tuned"]                = run_filter(s, KalmanANCFilter(L=64, sigma_q2=wind.sigma_q2, sigma_r2=wind.sigma_r2))
    runs["IMM v5"]                       = run_filter(s, IMMKalmanANC(L=64, likelihood_window=200))
    runs["IMM v6 (capped Q)"]            = run_filter(s, IMMKalmanANC(L=64, mode_params=V6_MODE_PARAMS, likelihood_window=200))
    runs["IMM v5 + ŵ smooth α=0.02"]     = run_filter(s, IMMKalmanANC(L=64, likelihood_window=200), alpha_w_smooth=0.02)

    print("Computing Wiener floor J_min...")
    j_min, e_wiener = wiener_jmin(s)
    runs["Wiener floor (J_min)"] = e_wiener

    print()
    print("=" * 110)
    print(f"{'Method':<30}  {'IL/NR':>7}  {'Pe':>10}  {'M':>6}  "
          f"{'ΔSNR':>7}  {'τc q→t':>8}  {'τc t→w':>8}  {'τc w→b':>8}")
    print(f"{'':<30}  {'[dB]':>7}  {'(MSE)':>10}  {'(-)':>6}  "
          f"{'[dB]':>7}  {'[ms]':>8}  {'[ms]':>8}  {'[ms]':>8}")
    print("=" * 110)
    summary = {}
    for name, e in runs.items():
        IL = insertion_loss_db(s.d, e)
        Pe = residual_power(e)
        M = misadjustment(e, j_min)
        dsnr = speech_band_dsnr(s.d, e, s.fs)
        conv = convergence_time_ms(s.d, e, s.fs)
        summary[name] = dict(IL=IL, Pe=Pe, M=M, dSNR=dsnr, conv=conv)
        print(f"  {name:<28}  {IL:>+7.2f}  {Pe:>10.2e}  {M:>+6.2f}  "
              f"{dsnr:>+7.2f}  {conv.get(1, float('nan')):>8.1f}  "
              f"{conv.get(2, float('nan')):>8.1f}  "
              f"{conv.get(3, float('nan')):>8.1f}")
    print("=" * 110)
    print("  IL/NR       = Insertion Loss = overall NR")
    print("  Pe          = Residual noise power (= MSE)")
    print("  M           = Misadjustment (J_ss - J_min) / J_min, ideally 0")
    print("  ΔSNR        = NR(300-3k) - NR(out of band), speech-band selectivity")
    print("  τc q→t      = Convergence time (ms) after quiet→traffic switch")

    # ============== NR(f) overlay =================================
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, e in runs.items():
        f_, nr_ = freq_nr(s.d, e, s.fs, n_octave_smooth=32)
        keep = (f_ >= 50) & (f_ <= 8000)
        ls = "--" if "Wiener" in name else "-"
        lw = 2.0 if "Wiener" in name else 1.4
        ax.semilogx(f_[keep], nr_[keep], ls, lw=lw, label=name)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("NR(f) = 10 log₁₀ |D|²/|E|²  [dB]")
    ax.set_title("Per-frequency Noise Reduction — Insertion Loss across the audible band")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = ROOT / "figures" / "13_nr_per_freq.png"
    fig.savefig(out, dpi=120); print(f"\nsaved {out}")

    # ============== Convergence comparison =========================
    fig, ax = plt.subplots(figsize=(11, 5))
    win = int(0.10 * s.fs)
    t_ax = (np.arange(len(s.d) - win + 1) + win // 2) / s.fs
    for name, e in runs.items():
        nr_t = noise_reduction_db_sliding(s.d, e, win)
        ls = "--" if "Wiener" in name else "-"
        ax.plot(t_ax, nr_t, ls, lw=1.2, label=name)
    for ts in [5, 10, 15]:
        ax.axvline(ts, color="grey", ls="--", lw=0.6, alpha=0.5)
    ax.set_xlabel("time [s]"); ax.set_ylabel("Sliding NR (100 ms window) [dB]")
    ax.set_title("Sliding-window NR across mode boundaries (transients live in the dips after t=5/10/15s)")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    out = ROOT / "figures" / "13_convergence.png"
    fig.savefig(out, dpi=120); print(f"saved {out}")

    # ============== Save raw numbers ===============================
    npz = ROOT / "figures" / "13_full_metrics.npz"
    names = list(runs.keys())
    np.savez(npz,
             method_names=np.array(names),
             IL=np.array([summary[n]["IL"] for n in names]),
             Pe=np.array([summary[n]["Pe"] for n in names]),
             M=np.array([summary[n]["M"] for n in names]),
             dSNR=np.array([summary[n]["dSNR"] for n in names]),
             conv_q_t=np.array([summary[n]["conv"].get(1, float("nan")) for n in names]),
             conv_t_w=np.array([summary[n]["conv"].get(2, float("nan")) for n in names]),
             conv_w_b=np.array([summary[n]["conv"].get(3, float("nan")) for n in names]),
             j_min=j_min)
    print(f"saved raw numbers to {npz}")


if __name__ == "__main__":
    main()
