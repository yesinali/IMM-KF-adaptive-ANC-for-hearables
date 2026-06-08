"""Adaptive filters: NLMS and single-mode (filtered-x) Kalman.

Both filters operate on the filtered reference vector xf(k) (length-L tapped
delay) and the primary signal d(k). The cancellation residual returned by
each `step` is exactly the Kalman innovation e(k) = d(k) - xf(k)^T w_hat(k).

State-space model used by KalmanANCFilter (Sayed 2003, §30):
    w(k+1) = w(k) + q,   q ~ N(0, sigma_q2 * I_L)
    d(k)   = xf(k)^T w(k) + v,   v ~ N(0, sigma_r2)
"""
from __future__ import annotations
import numpy as np


class NLMSFilter:
    """Normalized LMS controller (filtered-x convention)."""

    def __init__(self, L: int, mu: float = 0.1, eps: float = 1e-3):
        self.L = L
        self.mu = float(mu)
        self.eps = float(eps)
        self.w = np.zeros(L)

    def step(self, xf_vec: np.ndarray, d: float) -> tuple[float, np.ndarray]:
        e = float(d - self.w @ xf_vec)
        norm = float(xf_vec @ xf_vec) + self.eps
        self.w = self.w + (self.mu * e / norm) * xf_vec
        return e, self.w


class KalmanANCFilter:
    """Single-mode (filtered-x) Kalman with state = adaptive FIR weights.

    Exposes `self.S` (current innovation variance) and `self.P` (current
    posterior covariance) for downstream consistency tests (NEES/NIS).
    """

    def __init__(self, L: int, sigma_q2: float, sigma_r2: float,
                 p0: float = 1.0):
        self.L = L
        self.sigma_q2 = float(sigma_q2)
        self.sigma_r2 = float(sigma_r2)
        self.w = np.zeros(L)
        self.P = p0 * np.eye(L)
        self.S = float(sigma_r2)  # initialized to prior measurement variance

    def step(self, xf_vec: np.ndarray, d: float) -> tuple[float, np.ndarray]:
        # Predict: random walk -> mean unchanged, covariance += Q.
        # Q = sigma_q2 * I, so we add sigma_q2 only to the diagonal.
        P_pred = self.P + self.sigma_q2 * np.eye(self.L)
        # Update (H = xf_vec as a row vector).
        Pxf = P_pred @ xf_vec
        S = float(xf_vec @ Pxf) + self.sigma_r2
        K = Pxf / S
        e = float(d - self.w @ xf_vec)
        self.w = self.w + K * e
        self.P = P_pred - np.outer(K, Pxf)
        self.S = S
        return e, self.w
