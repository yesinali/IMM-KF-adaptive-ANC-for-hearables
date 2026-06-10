"""Music + noise ANC test.

Real-world hearable case: user is listening to music while ambient noise leaks
into the ear canal. The adaptive controller's reference microphone hears the
external noise but not the music; the error microphone (in the canal) hears
*both* music and the leaked noise. A successful controller cancels the noise
and leaves the music intact.

Pipeline:
    d(k)  = music(k) + alpha * noise_at_eardrum(k)     # what error mic sees
    xf(k) = alpha * filtered_noise_reference(k)         # outside-mic reference
    e(k)  = d(k) - w(k)·xf(k) ≈ music(k) + residual_noise

We score each algorithm by how close e(k) is to clean music:
  - SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)  -- higher = better
  - PSNR vs reference music                              -- higher = better
  - Residual noise power in the e(k)                     -- lower  = better

WAVs are exported so the user can listen and judge subjectively.

Run:
    python -m scripts.music_noise_test
    python -m scripts.music_noise_test --music "musics/Ara Durak - Afra.flac"
    python -m scripts.music_noise_test --snr 0   # default 0 dB music vs noise
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import resample_poly, spectrogram

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src import scenario as sc
from src.filters import NLMSFilter, KalmanANCFilter
from src.imm import IMMKalmanANC
from src.metrics import overall_nr_db
from src.music import load_music_clip  # noqa: F401  (re-exported; original home)


TRAJ = (
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


# ============== Audio I/O ====================================================
# load_music_clip moved to src/music.py (shared with the app and the test bench).

def normalize_for_wav(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x)) + 1e-12)
    return (x / peak * 0.9).astype(np.float32)


# ============== ANC runner ===================================================

def run_filter(d: np.ndarray, xf: np.ndarray, filt,
               alpha_w_smooth: float = 0.0) -> np.ndarray:
    """Naïve ANC: filter sees the full d (music + noise) and tries to drive
    it to zero. The filter cannot distinguish music from noise, so it
    attempts to cancel both. Kept for the 'system-flaw demo'."""
    N = len(d); L = filt.L
    buf = np.zeros(L); e = np.zeros(N)
    w_smooth = np.zeros(L) if alpha_w_smooth > 0 else None
    for k in range(N):
        buf[1:] = buf[:-1]; buf[0] = xf[k]
        e_inner, w_now = filt.step(buf, d[k])
        if w_smooth is not None:
            w_smooth = (1 - alpha_w_smooth) * w_smooth + alpha_w_smooth * w_now
            e[k] = d[k] - w_smooth @ buf
        else:
            e[k] = e_inner
    return e


def run_filter_music_aware(noise_eardrum: np.ndarray, music: np.ndarray,
                           xf: np.ndarray, filt,
                           alpha_w_smooth: float = 0.0) -> np.ndarray:
    """Music-aware ANC: the filter's adaptation sees only the noise-derived
    error signal e_filter = noise_eardrum - anti_noise. This is what a
    real hearable does: the controller knows what music it is playing
    and subtracts that music's eardrum contribution from the error mic
    before adapting, so adaptation is driven only by noise residuals.

    The user, however, hears the full acoustic field at the eardrum:
        e_user(k) = music(k) + (noise_eardrum(k) - anti_noise(k))
    """
    N = len(noise_eardrum); L = filt.L
    buf = np.zeros(L); e_user = np.zeros(N)
    w_smooth = np.zeros(L) if alpha_w_smooth > 0 else None
    for k in range(N):
        buf[1:] = buf[:-1]; buf[0] = xf[k]
        e_noise, w_now = filt.step(buf, noise_eardrum[k])
        if w_smooth is not None:
            w_smooth = (1 - alpha_w_smooth) * w_smooth + alpha_w_smooth * w_now
            y = w_smooth @ buf
            e_user[k] = music[k] + (noise_eardrum[k] - y)
        else:
            e_user[k] = music[k] + e_noise
    return e_user


# ============== Metrics ======================================================

def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio in dB.

    estimate = alpha * reference + e_distortion
    SI-SDR = 10 log10( ||alpha*ref||^2 / ||distortion||^2 )
    """
    ref = reference - reference.mean()
    est = estimate  - estimate.mean()
    alpha = float(np.dot(est, ref) / (np.dot(ref, ref) + 1e-30))
    target = alpha * ref
    distortion = est - target
    return 10.0 * np.log10(np.sum(target ** 2) / (np.sum(distortion ** 2) + 1e-30))


def music_psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """PSNR against the clean music reference (treating reference peak as 1)."""
    mse = float(np.mean((estimate - reference) ** 2))
    if mse <= 0:
        return float("inf")
    peak = float(np.max(np.abs(reference)) + 1e-12)
    return 10.0 * np.log10(peak ** 2 / mse)


def residual_noise_power_db(noise_only_residual: np.ndarray) -> float:
    p = float(np.mean(noise_only_residual ** 2))
    return 10.0 * np.log10(p + 1e-30)


# ============== Main =========================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--music", type=str,
                    default=str(ROOT / "musics" / "Ara Durak - Afra.flac"),
                    help="path to a FLAC/WAV music file")
    ap.add_argument("--snr", type=float, default=0.0,
                    help="music-vs-noise input SNR in dB. 0 means equal power.")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    # Build the noise scenario FIRST so we can match the music RMS to it.
    rng = np.random.default_rng(args.seed)
    s = sc.build_scenario(segments=TRAJ, rng=rng, mode_conditioned_plants=True)

    noise_rms = float(np.sqrt(np.mean(s.d ** 2)))
    # For SNR_in = args.snr (dB), the music RMS must satisfy
    #     20 log10(music_rms / noise_rms) = args.snr
    target_music_rms = noise_rms * (10.0 ** (args.snr / 20.0))

    music_path = Path(args.music)
    print(f"Loading music: {music_path.name}")
    music = load_music_clip(music_path, target_fs=cfg.FS,
                            duration_sec=20.0,
                            target_rms=target_music_rms)
    assert len(s.d) == len(music), "scenario duration must match music duration"

    # Now noise stays at its calibration scale, and music sits at the
    # requested SNR above (or below) it.
    noise_at_eardrum = s.d
    xf_mixed = s.x_filt
    d_mixed = music + noise_at_eardrum
    music_rms = float(np.sqrt(np.mean(music ** 2)))
    in_snr = 20 * np.log10(music_rms / max(noise_rms, 1e-30))
    print(f"music RMS = {music_rms:.4f}, noise RMS = {noise_rms:.4f}")
    print(f"input SNR (music vs noise at eardrum) = {in_snr:+.2f} dB")

    # Wind tuned KF: best fixed-Q from MC
    wind = next(p for p in cfg.MODE_PARAMS if p.name == "wind")

    print("\nRunning algorithms (music-aware: filter adapts on noise only)...")
    runner = run_filter_music_aware  # the correct model
    results: dict[str, np.ndarray] = {}
    results["NLMS µ=0.10"]              = runner(noise_at_eardrum, music, xf_mixed, NLMSFilter(L=64, mu=0.10))
    results["KF wind-tuned"]            = runner(noise_at_eardrum, music, xf_mixed, KalmanANCFilter(L=64, sigma_q2=wind.sigma_q2, sigma_r2=wind.sigma_r2))
    results["IMM v5"]                   = runner(noise_at_eardrum, music, xf_mixed, IMMKalmanANC(L=64, likelihood_window=200))
    results["IMM v6 (capped Q)"]        = runner(noise_at_eardrum, music, xf_mixed, IMMKalmanANC(L=64, mode_params=V6_MODE_PARAMS, likelihood_window=200))
    results["IMM v5 + ŵ smooth α=0.02"] = runner(noise_at_eardrum, music, xf_mixed, IMMKalmanANC(L=64, likelihood_window=200), alpha_w_smooth=0.02)

    # No-ANC baseline: user just hears music + noise (no anti-noise)
    results["No ANC (baseline)"] = d_mixed.copy()

    print()
    print("=" * 92)
    print(f"{'Method':<30}  {'SI-SDR':>8}  {'PSNR':>8}  {'NR_at_noise':>11}  {'Δ vs No-ANC':>12}")
    print(f"{'':<30}  {'[dB]':>8}  {'[dB]':>8}  {'[dB]':>11}  {'SI-SDR [dB]':>12}")
    print("=" * 92)
    baseline_sisdr = si_sdr(music, results["No ANC (baseline)"])
    table = {}
    for name, e in results.items():
        sdr = si_sdr(music, e)
        psnr = music_psnr(music, e)
        # NR measured on the noise component only — estimate as e - music
        residual_noise = e - music
        nr = 10 * np.log10(np.mean(noise_at_eardrum**2) / (np.mean(residual_noise**2) + 1e-30))
        delta_sdr = sdr - baseline_sisdr
        table[name] = dict(sdr=sdr, psnr=psnr, nr=nr, delta=delta_sdr)
        print(f"  {name:<28}  {sdr:>+8.2f}  {psnr:>+8.2f}  {nr:>+11.2f}  {delta_sdr:>+12.2f}")
    print("=" * 92)
    print("  SI-SDR        = Scale-Invariant Signal-to-Distortion Ratio (music recovery)")
    print("  PSNR          = peak-signal-to-noise ratio vs clean music")
    print("  NR_at_noise   = noise-only residual reduction (filter's job)")
    print("  Δ vs No-ANC   = how much *better* than just doing nothing")

    # -------- Spectrogram comparison --------
    print("\nRendering spectrograms...")
    methods_to_plot = [
        ("Clean music (reference)",      music),
        ("d = music + noise (No ANC)",   d_mixed),
        ("NLMS",                          results["NLMS µ=0.10"]),
        ("IMM v5",                        results["IMM v5"]),
        ("IMM v6 (capped Q)",             results["IMM v6 (capped Q)"]),
        ("IMM v5 + ŵ smooth",             results["IMM v5 + ŵ smooth α=0.02"]),
    ]
    fig, axes = plt.subplots(len(methods_to_plot), 1, figsize=(11, 14),
                             sharex=True, sharey=True)
    f0, t0, S0 = spectrogram(music, fs=cfg.FS, nperseg=512, noverlap=384)
    db0 = 10 * np.log10(S0 + 1e-12)
    vmin = float(np.percentile(db0, 5))
    vmax = float(np.percentile(db0, 99))
    for ax, (name, sig) in zip(axes, methods_to_plot):
        f_, t_, S = spectrogram(sig, fs=cfg.FS, nperseg=512, noverlap=384)
        db = 10 * np.log10(S + 1e-12)
        im = ax.pcolormesh(t_, f_, db, cmap="magma", vmin=vmin, vmax=vmax,
                           shading="auto")
        ax.set_ylabel("Hz")
        ax.set_title(name, fontsize=10)
        ax.set_ylim(0, 6000)
        for ts in [5, 10, 15]:
            ax.axvline(ts, color="cyan", ls="--", lw=0.7, alpha=0.7)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"Spectrogram: music + noise scenario, input SNR = {args.snr:+.1f} dB",
                 fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.85, label="dB",
                 location="right", pad=0.02)
    out = ROOT / "figures" / "14_music_noise_spectrograms.png"
    fig.savefig(out, dpi=120); print(f"saved {out}")

    # -------- WAV exports --------
    out_dir = ROOT / "figures"
    print(f"\nExporting WAVs to {out_dir}/music_*.wav ...")
    sf.write(out_dir / "music_clean.wav",         normalize_for_wav(music),     cfg.FS)
    sf.write(out_dir / "music_noisy_noanc.wav",   normalize_for_wav(d_mixed),   cfg.FS)
    sf.write(out_dir / "music_nlms.wav",          normalize_for_wav(results["NLMS µ=0.10"]),              cfg.FS)
    sf.write(out_dir / "music_kfwind.wav",        normalize_for_wav(results["KF wind-tuned"]),            cfg.FS)
    sf.write(out_dir / "music_v5.wav",            normalize_for_wav(results["IMM v5"]),                   cfg.FS)
    sf.write(out_dir / "music_v6.wav",            normalize_for_wav(results["IMM v6 (capped Q)"]),        cfg.FS)
    sf.write(out_dir / "music_v5_smooth.wav",     normalize_for_wav(results["IMM v5 + ŵ smooth α=0.02"]), cfg.FS)
    print("  music_clean.wav        - clean reference (what you'd love to hear)")
    print("  music_noisy_noanc.wav  - what you'd hear without ANC")
    print("  music_nlms.wav         - after NLMS")
    print("  music_kfwind.wav       - after fixed-Q KF (wind)")
    print("  music_v5.wav           - after IMM v5")
    print("  music_v6.wav           - after IMM v6 (capped Q)")
    print("  music_v5_smooth.wav    - after IMM v5 + ŵ smoothing")

    # -------- Persist raw numbers --------
    npz = ROOT / "figures" / "14_music_noise_metrics.npz"
    names = list(results.keys())
    np.savez(npz,
             method_names=np.array(names),
             si_sdr=np.array([table[n]["sdr"] for n in names]),
             psnr=np.array([table[n]["psnr"] for n in names]),
             nr_at_noise=np.array([table[n]["nr"] for n in names]),
             delta_sdr=np.array([table[n]["delta"] for n in names]),
             in_snr=in_snr, music_file=str(music_path.name))
    print(f"\nsaved raw numbers to {npz}")


if __name__ == "__main__":
    main()
