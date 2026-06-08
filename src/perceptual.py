"""Perceptual metrics that bridge the gap between "numbers" and "what a human
feels" when listening to the ANC residual.

Overall NR (10 log10 var(d)/var(e)) answers "how much energy was removed" but
not "how much quieter does it *sound*" or "does the residual sound natural".
This module adds three numbers that track the felt experience:

  1. dBA loudness reduction  -- A-weighted (IEC 61672) level drop, the standard
     proxy for perceived loudness change. "+12 dBA quieter" is something a
     listener actually experiences; raw dB over-credits sub-bass the ear barely
     hears.
  2. Band-split / third-octave NR -- where the reduction lives. A real ANC earbud
     kills lows and leaves highs, so the split into the active-ANC band vs the
     passive band is exactly the "the rumble's gone but I still hear voices"
     character.
  3. Musical-noise / artefact index -- the tonal-bursts-coming-and-going and
     transient clicks that the Streamlit listening survey catches subjectively.
     Quantified from the spectrogram so the report can show the algo-vs-perception
     story with a number, not just an opinion.

No new dependencies: the A-weighting filter is built from its analog prototype
with scipy's bilinear transform.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import bilinear, tf2sos, sosfilt, stft


# ---- A-weighting (IEC 61672 / ANSI S1.4) ------------------------------------

def a_weighting_sos(fs: int) -> np.ndarray:
    """Second-order-sections for an A-weighting filter at sample rate `fs`.

    Standard analog prototype (poles at 20.6, 107.7, 737.9, 12194 Hz, double
    at the outer two), normalized to 0 dB at 1 kHz, then bilinear-transformed.
    """
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    a1000 = 1.9997  # gain (dB) to normalize the response to 0 dB at 1 kHz
    nums = [(2 * np.pi * f4) ** 2 * 10 ** (a1000 / 20.0), 0.0, 0.0, 0.0, 0.0]
    dens = np.polymul([1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
                      [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2])
    dens = np.polymul(np.polymul(dens, [1, 2 * np.pi * f3]),
                      [1, 2 * np.pi * f2])
    b, a = bilinear(nums, dens, fs)
    return tf2sos(b, a)


def dba_level(x: np.ndarray, fs: int) -> float:
    """A-weighted RMS level in dB (relative, uncalibrated to SPL)."""
    y = sosfilt(a_weighting_sos(fs), x)
    return 20.0 * np.log10(np.sqrt(np.mean(y * y)) + 1e-20)


def dba_reduction(d: np.ndarray, e: np.ndarray, fs: int) -> float:
    """Perceived (A-weighted) loudness reduction in dB: dBA(d) - dBA(e).

    Calibration-independent, so the uncalibrated reference level cancels.
    """
    return dba_level(d, fs) - dba_level(e, fs)


# ---- Where the reduction lives ---------------------------------------------

def _band_power(F: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    band = (freqs >= lo) & (freqs < hi)
    return float(np.sum(np.abs(F[band]) ** 2))


def band_split_nr(d: np.ndarray, e: np.ndarray, fs: int,
                  crossover: float = 1200.0) -> tuple[float, float]:
    """NR (dB) below vs at/above `crossover` Hz -> (anc_band_nr, passive_band_nr).

    The active loop should win big below the crossover ("rumble gone") and do
    little above it (where the passive seal, not active ANC, does the work).
    """
    F_d = np.fft.rfft(d)
    F_e = np.fft.rfft(e)
    freqs = np.fft.rfftfreq(len(d), d=1.0 / fs)
    nyq = fs / 2.0
    lo_nr = 10.0 * np.log10(_band_power(F_d, freqs, 0.0, crossover) /
                            max(_band_power(F_e, freqs, 0.0, crossover), 1e-30))
    hi_nr = 10.0 * np.log10(_band_power(F_d, freqs, crossover, nyq) /
                            max(_band_power(F_e, freqs, crossover, nyq), 1e-30))
    return lo_nr, hi_nr


def third_octave_nr(d: np.ndarray, e: np.ndarray, fs: int,
                    f_lo: float = 25.0, f_hi: float | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """NR (dB) in standard 1/3-octave bands. Returns (center_freqs, nr_db)."""
    if f_hi is None:
        f_hi = fs / 2.0
    F_d = np.fft.rfft(d)
    F_e = np.fft.rfft(e)
    freqs = np.fft.rfftfreq(len(d), d=1.0 / fs)
    ratio = 2.0 ** (1.0 / 6.0)  # half-band factor for 1/3 octave
    centers, nrs = [], []
    # walk down then up from 1 kHz on the base-2 1/3-octave grid
    grid = []
    f = 1000.0
    while f / ratio >= f_lo:
        f /= ratio ** 2  # step a full third-octave (factor 2^(1/3))
    while f * ratio <= f_hi:
        grid.append(f)
        f *= 2.0 ** (1.0 / 3.0)
    for fc in grid:
        lo, hi = fc / ratio, fc * ratio
        pd = _band_power(F_d, freqs, lo, hi)
        pe = _band_power(F_e, freqs, lo, hi)
        if pd <= 0:
            continue
        centers.append(fc)
        nrs.append(10.0 * np.log10(pd / max(pe, 1e-30)))
    return np.array(centers), np.array(nrs)


# ---- How natural does the residual sound -----------------------------------

def musical_noise_index(e: np.ndarray, fs: int,
                        nperseg: int = 512, noverlap: int = 384) -> float:
    """Spectral kurtosis of the residual's spectrogram — a musical-noise proxy.

    Musical noise = isolated time-frequency energy blobs flicking on and off.
    They make the distribution of spectrogram power heavy-tailed, so its excess
    kurtosis rises. A clean, naturally-textured residual has near-Gaussian
    log-power and a low index; one riddled with tonal bursts scores high.
    Reported as the mean per-frequency-bin excess kurtosis of the log power.
    """
    _, _, Z = stft(e, fs=fs, nperseg=nperseg, noverlap=noverlap)
    logp = np.log(np.abs(Z) ** 2 + 1e-12)  # (n_freq, n_time)
    mu = logp.mean(axis=1, keepdims=True)
    sd = logp.std(axis=1, keepdims=True) + 1e-12
    z = (logp - mu) / sd
    kurt = np.mean(z ** 4, axis=1) - 3.0  # excess kurtosis per freq bin
    return float(np.mean(kurt))


def flux_burstiness(e: np.ndarray, fs: int,
                    nperseg: int = 512, noverlap: int = 384) -> float:
    """Burstiness of spectral change — a transient/click proxy.

    Spectral flux = frame-to-frame L2 change of the magnitude spectrum. Smooth
    adaptation gives steady flux; transient clicks at mode switches give spiky
    flux. Reported as the coefficient of variation (std/mean) of the flux, so a
    few sharp spikes over an otherwise-flat trace score high.
    """
    _, _, Z = stft(e, fs=fs, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Z)
    flux = np.sqrt(np.sum(np.diff(mag, axis=1) ** 2, axis=0))
    return float(np.std(flux) / (np.mean(flux) + 1e-12))


# ---- Music-recovery quality (for the music-aware ANC case) ------------------

def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio in dB. Higher = the estimate
    recovers the reference (clean music) better.

    estimate = alpha*reference + distortion, SI-SDR = 10 log10(||alpha*ref||²/||distortion||²).
    """
    ref = reference - reference.mean()
    est = estimate - estimate.mean()
    alpha = float(np.dot(est, ref) / (np.dot(ref, ref) + 1e-30))
    target = alpha * ref
    distortion = est - target
    return 10.0 * np.log10(np.sum(target ** 2) / (np.sum(distortion ** 2) + 1e-30))


def music_psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Peak-signal-to-noise ratio (dB) of the estimate vs the clean reference."""
    mse = float(np.mean((estimate - reference) ** 2))
    if mse <= 0:
        return float("inf")
    peak = float(np.max(np.abs(reference)) + 1e-12)
    return 10.0 * np.log10(peak ** 2 / mse)
