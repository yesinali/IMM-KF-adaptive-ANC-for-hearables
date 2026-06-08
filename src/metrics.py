"""Evaluation metrics: noise reduction, misalignment, NEES/NIS consistency.

Sliding-window stats use a cumulative-sum trick for O(N) cost.

NEES/NIS background
-------------------
For a consistent Bayesian filter (i.e., one whose reported covariance P_k is
calibrated against the true estimation error), the Normalized Estimation
Error Squared (NEES) and Normalized Innovation Squared (NIS) are the standard
chi-squared-based consistency tests (Bar-Shalom, Li & Kirubarajan 2001).

NEES:  eps_k = (w_hat_k - w_true)^T P_k^-1 (w_hat_k - w_true)
       E[eps_k] = L (state dimension); time-averaged NEES should lie in the
       95% chi^2 band [L_low, L_high] with N samples or N MC runs.

NIS:   nu_k = innovation; S_k = innovation variance
       E[nu_k^2 / S_k] = 1; same chi^2-band logic with dof=1.
"""
from __future__ import annotations
import numpy as np


def sliding_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Length (N - w + 1) sliding-window mean of x."""
    if w <= 1 or w > len(x):
        return x.copy()
    csum = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    return (csum[w:] - csum[:-w]) / w


def noise_reduction_db_sliding(d: np.ndarray, e: np.ndarray, window: int) -> np.ndarray:
    """10 log10( <d^2>_win / <e^2>_win ) over a sliding window."""
    pd = sliding_mean(d * d, window)
    pe = sliding_mean(e * e, window)
    return 10.0 * np.log10((pd + 1e-20) / (pe + 1e-20))


def overall_nr_db(d: np.ndarray, e: np.ndarray) -> float:
    return float(10.0 * np.log10((np.mean(d * d) + 1e-20) /
                                 (np.mean(e * e) + 1e-20)))


def misalignment_db(misalignment: np.ndarray) -> np.ndarray:
    """Convert linear normalized misalignment to dB."""
    return 10.0 * np.log10(misalignment + 1e-12)


def nees(w_hat: np.ndarray, w_true: np.ndarray, P: np.ndarray,
         ridge: float = 1e-9) -> float:
    """Single-sample NEES: (w_hat - w_true)^T P^-1 (w_hat - w_true).

    `ridge` adds a small diagonal to P before solve, guarding against
    numerical singularity when the covariance has nearly collapsed.
    """
    L = len(w_hat)
    err = w_hat - w_true
    P_reg = P + ridge * np.eye(L)
    return float(err @ np.linalg.solve(P_reg, err))


def nis(innovation: float, S: float, ridge: float = 1e-15) -> float:
    """Single-sample NIS for a scalar observation: nu^2 / S."""
    return float(innovation * innovation / max(float(S), ridge))


def chi2_bounds(n_samples: int, dof: int = 1,
                confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided chi^2 confidence band for the time-averaged NEES/NIS.

    Under filter consistency, the sum of `n_samples` independent chi^2(dof)
    random variates is chi^2(n_samples * dof); the sample mean therefore
    has bounds chi^2(N*dof) / N at the appropriate quantiles.

    Returns (lower, upper) such that
        P(lower <= average <= upper) = confidence
    when the filter is consistent.
    """
    from scipy.stats import chi2
    alpha = 1.0 - confidence
    N = n_samples
    lower = chi2.ppf(alpha / 2.0, N * dof) / N
    upper = chi2.ppf(1.0 - alpha / 2.0, N * dof) / N
    return float(lower), float(upper)
