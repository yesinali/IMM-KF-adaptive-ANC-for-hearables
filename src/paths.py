"""Synthetic acoustic path generation.

Primary path P(z): long FIR with exponentially decaying envelope, modelling
the in-ear path from the noise source to the eardrum.
Secondary path S(z): shorter FIR modelling the driver-to-error-mic path.

These are deliberately randomized per Monte Carlo run so the IMM-KF must
generalize, not memorize.
"""
from __future__ import annotations
import numpy as np

from . import config as cfg


def random_fir(length: int, decay_tau: float, rng: np.random.Generator) -> np.ndarray:
    """Random FIR with exp(-n/tau) envelope, unit RMS."""
    n = np.arange(length)
    envelope = np.exp(-n / float(decay_tau))
    h = rng.standard_normal(length) * envelope
    h = h / (np.sqrt(np.mean(h ** 2)) + 1e-12)
    return h


def primary_path(rng: np.random.Generator, length: int = cfg.P_LEN) -> np.ndarray:
    return random_fir(length, decay_tau=length / 4.0, rng=rng)


def secondary_path(rng: np.random.Generator, length: int = cfg.S_LEN) -> np.ndarray:
    return random_fir(length, decay_tau=length / 6.0, rng=rng)


def apply_fir(h: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Causal FIR convolution truncated to len(x). Equivalent to scipy.signal.lfilter(h, 1, x)."""
    y = np.convolve(x, h, mode="full")[: len(x)]
    return y
