"""Small numerical helpers: dB conversions, RMS, SNR scaling, transition matrix."""
from __future__ import annotations
import numpy as np

from . import config as cfg


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def db20(x: float) -> float:
    return 20.0 * np.log10(max(x, 1e-20))


def db10(x: float) -> float:
    return 10.0 * np.log10(max(x, 1e-20))


def scale_to_snr(signal: np.ndarray, noise: np.ndarray, target_snr_db: float) -> np.ndarray:
    """Return a rescaled `noise` so that 10log10(P_signal/P_noise) = target_snr_db."""
    p_sig = float(np.mean(signal ** 2))
    p_noise = float(np.mean(noise ** 2))
    if p_noise <= 0.0 or p_sig <= 0.0:
        return noise
    desired_p_noise = p_sig / (10.0 ** (target_snr_db / 10.0))
    return noise * np.sqrt(desired_p_noise / p_noise)


def transition_matrix(n_modes: int = cfg.N_MODES,
                      p_diag: float = cfg.PI_DIAG) -> np.ndarray:
    """Symmetric Markov transition matrix with `p_diag` on the diagonal."""
    p_off = (1.0 - p_diag) / (n_modes - 1)
    pi = np.full((n_modes, n_modes), p_off)
    np.fill_diagonal(pi, p_diag)
    return pi
