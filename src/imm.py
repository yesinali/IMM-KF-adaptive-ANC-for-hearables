"""Interacting Multiple Model (IMM) filter for ANC.

Vectorized stacked-state implementation: instead of holding a list of
KalmanANCFilter instances, we keep one (M, L, L) covariance tensor and one
(M, L) weight matrix and run all four IMM steps with NumPy ops. This is
~10x faster than the loop version and matches it numerically.

Reference: Blom & Bar-Shalom 1988.
"""
from __future__ import annotations
from typing import Sequence

import numpy as np

from . import config as cfg
from .utils import transition_matrix


class IMMKalmanANC:
    def __init__(self,
                 L: int,
                 mode_params: Sequence[cfg.ModeParams] = cfg.MODE_PARAMS,
                 transition: np.ndarray | None = None,
                 p0: float = 1.0,
                 likelihood_window: int = 1):
        """`likelihood_window` (samples) applies exponential smoothing to the
        per-mode log-likelihood used in the posterior update. W=1 reproduces
        the textbook single-sample IMM; W>>1 reduces sample-level posterior
        noise and helps the mode posterior commit to the true mode under
        transient-heavy regimes (e.g., wind)."""
        M = len(mode_params)
        self.L = L
        self.M = M
        self.mode_names = tuple(p.name for p in mode_params)
        # Stacked state.
        self.W = np.zeros((M, L))
        self.P = np.tile(p0 * np.eye(L), (M, 1, 1))
        self.sigma_q2 = np.array([p.sigma_q2 for p in mode_params])
        self.sigma_r2 = np.array([p.sigma_r2 for p in mode_params])
        self.Pi = transition if transition is not None else transition_matrix(M)
        self.mu = np.full(M, 1.0 / M)
        # Combined posterior estimate (returned by step; used for NEES/NIS).
        self.w = np.zeros(L)
        self.P_combined = p0 * np.eye(L)
        self.S = float(np.mean(self.sigma_r2))
        self._eye_L = np.eye(L)
        # Smoothed log-likelihood used for the posterior update.
        self.alpha = 1.0 / max(1, likelihood_window)
        self._log_lik_smooth = np.zeros(M)

    def step(self, xf: np.ndarray, d: float) -> tuple[float, np.ndarray]:
        M, L = self.M, self.L

        # ----- 1. Mixing -----
        cbar = self.Pi.T @ self.mu                                   # (M,)
        mu_cond = (self.Pi * self.mu[:, None]) / (cbar[None, :] + 1e-15)  # (M, M)
        w_mix = mu_cond.T @ self.W                                   # (M, L)

        # Mixed covariances: P_mix[j] = Σ_i mu_cond[i,j] (P[i] + outer(W[i]-w_mix[j]))
        # Vectorize over i for each j to keep memory bounded at O(M·L²).
        P_mix = np.empty((M, L, L))
        for j in range(M):
            diffs = self.W - w_mix[j]                                # (M, L)
            outers = diffs[:, :, None] * diffs[:, None, :]           # (M, L, L)
            P_mix[j] = (mu_cond[:, j, None, None] * (self.P + outers)).sum(axis=0)

        # ----- 2. Mode-conditioned KF update -----
        # P_pred[j] = P_mix[j] + sigma_q2[j] * I
        P_pred = P_mix + self.sigma_q2[:, None, None] * self._eye_L
        # Pxf[j] = P_pred[j] @ xf
        Pxf = P_pred @ xf                                            # (M, L)
        S = (xf[None, :] * Pxf).sum(axis=1) + self.sigma_r2          # (M,)
        K = Pxf / S[:, None]                                         # (M, L)
        innov = d - w_mix @ xf                                       # (M,)
        self.W = w_mix + K * innov[:, None]
        # P_new[j] = P_pred[j] - outer(K[j], Pxf[j])
        self.P = P_pred - K[:, :, None] * Pxf[:, None, :]

        # ----- 3. Mode probability update (log-space for stability) -----
        log_lik = -0.5 * (np.log(2 * np.pi * S) + innov * innov / S)
        # Exponential smoothing — reduces single-sample posterior noise.
        self._log_lik_smooth = (1 - self.alpha) * self._log_lik_smooth + self.alpha * log_lik
        ll = self._log_lik_smooth - self._log_lik_smooth.max()
        Lambda = np.exp(ll)
        post = Lambda * cbar
        self.mu = post / (post.sum() + 1e-30)

        # ----- 4. Combination (also produce combined P and S for NEES/NIS) -----
        self.w = (self.mu[:, None] * self.W).sum(axis=0)
        # Gaussian-mixture combined covariance:
        diff_w = self.W - self.w[None, :]                       # (M, L)
        outers_w = diff_w[:, :, None] * diff_w[:, None, :]      # (M, L, L)
        self.P_combined = (self.mu[:, None, None]
                           * (self.P + outers_w)).sum(axis=0)   # (L, L)
        # Combined innovation and its mixture variance:
        e_combined = float(d - self.w @ xf)
        diff_e = innov - e_combined                              # (M,)
        self.S = float((self.mu * (S + diff_e * diff_e)).sum())
        return e_combined, self.w
