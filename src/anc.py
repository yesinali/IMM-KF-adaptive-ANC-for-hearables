"""Drive an adaptive filter through a full scenario, sample by sample.

The scenario provides the time series xf(k) and d(k). At each sample we slide
a length-L tap buffer over xf, feed (buffer, d(k)) to the filter, and log
the residual e(k) plus the running misalignment against the Wiener weights.
"""
from __future__ import annotations
from typing import Protocol

import numpy as np

from . import config as cfg
from .scenario import ANCScenario


class _AdaptiveFilter(Protocol):
    L: int
    def step(self, xf_vec: np.ndarray, d: float) -> tuple[float, np.ndarray]: ...


def simulate_anc(scenario: ANCScenario,
                 filt: _AdaptiveFilter,
                 w_opt: np.ndarray | None = None,
                 log_weights_every: int | None = None,
                 log_mu: bool = False,
                 log_nees: bool = False,
                 log_nis: bool = False,
                 nees_decimate: int = 1) -> dict:
    """Run `filt` through the scenario. Returns dict with 'e', and optionally
    'misalignment', 'w_history', 'mu_history', 'nees', 'nis'.

    NEES requires `w_opt` (per-sample or global reference). NIS requires the
    filter to expose `.S` (innovation variance attribute), which is provided
    by KalmanANCFilter and IMMKalmanANC. For IMM, the combined-mixture P and
    S are used (these are the consistency targets for the *combined* estimator).

    `nees_decimate` (default 1): compute NEES every `nees_decimate` samples.
    Set to 100 to dramatically speed up consistency tests; the time-averaged
    NEES is unbiased under decimation as long as the filter is stationary
    over the decimation window.
    """
    xf = scenario.x_filt
    d = scenario.d
    N = len(d)
    L = filt.L

    buf = np.zeros(L)
    e_hist = np.zeros(N)

    # w_opt: None, (L,) global reference, or (N, L) per-sample reference.
    misalign = None
    w_opt_arr = None
    w_opt_norm2 = None
    if w_opt is not None:
        w_opt = np.asarray(w_opt)
        misalign = np.zeros(N)
        if w_opt.ndim == 1:
            w_opt_norm2 = float(w_opt @ w_opt + 1e-12)
        elif w_opt.ndim == 2:
            assert w_opt.shape == (N, L), f"w_opt shape {w_opt.shape} != ({N},{L})"
            w_opt_arr = w_opt
            w_opt_norm2 = np.einsum("ij,ij->i", w_opt_arr, w_opt_arr) + 1e-12
        else:
            raise ValueError("w_opt must be (L,) or (N, L)")

    w_history = None
    log_idx = None
    if log_weights_every:
        log_idx = np.arange(0, N, log_weights_every)
        w_history = np.zeros((len(log_idx), L))

    mu_history = None
    filt_mu = getattr(filt, "mu", None)
    if log_mu and isinstance(filt_mu, np.ndarray):
        mu_history = np.zeros((N, filt_mu.size))

    # Pick the right covariance attribute name: KF has .P, IMM has .P_combined.
    P_attr = "P_combined" if hasattr(filt, "P_combined") else "P"
    nees_decimate = max(1, int(nees_decimate))
    n_nees_pts = (N + nees_decimate - 1) // nees_decimate
    nees_hist = np.zeros(n_nees_pts) if (log_nees and w_opt is not None) else None
    nis_hist = np.zeros(N) if log_nis else None

    from . import metrics as _m  # local import to avoid cycles

    for k in range(N):
        # Shift tap buffer (newest sample first).
        buf[1:] = buf[:-1]
        buf[0] = xf[k]
        e, w_now = filt.step(buf, d[k])
        e_hist[k] = e
        if misalign is not None:
            ref = w_opt_arr[k] if w_opt_arr is not None else w_opt
            denom = w_opt_norm2[k] if w_opt_arr is not None else w_opt_norm2
            diff = w_now - ref
            misalign[k] = float(diff @ diff) / float(denom)
        if w_history is not None and k in log_idx:
            w_history[np.searchsorted(log_idx, k)] = w_now
        if mu_history is not None:
            mu_history[k] = filt.mu
        if nees_hist is not None and (k % nees_decimate == 0):
            ref = w_opt_arr[k] if w_opt_arr is not None else w_opt
            P_now = getattr(filt, P_attr)
            nees_hist[k // nees_decimate] = _m.nees(w_now, ref, P_now)
        if nis_hist is not None:
            S_now = getattr(filt, "S", None)
            if S_now is not None:
                nis_hist[k] = _m.nis(e, S_now)

    out = {"e": e_hist}
    if misalign is not None:
        out["misalignment"] = misalign
    if w_history is not None:
        out["w_history"] = w_history
        out["w_history_idx"] = log_idx
    if mu_history is not None:
        out["mu_history"] = mu_history
    if nees_hist is not None:
        out["nees"] = nees_hist
    if nis_hist is not None:
        out["nis"] = nis_hist
    return out
