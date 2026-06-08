"""Project-wide constants for the IMM-KF ANC simulator.

Values match the proposal (Section 3 mode table, Section 4 sampling).
"""
import os
from dataclasses import dataclass

# Noise source for scenario generation: "synthetic" (default, zero-download
# generators in noise.py) or "recorded" (real WAV/FLAC clips dropped into
# noise_samples/<mode>/, see src/noise_recorded.py). Override per-run with the
# ANC_NOISE_SOURCE environment variable. Recorded modes that have no clips fall
# back to synthetic automatically, so this is always safe to set.
NOISE_SOURCE = os.environ.get("ANC_NOISE_SOURCE", "synthetic").strip().lower()

FS = 16_000           # sample rate [Hz]
L = 64                # adaptive FIR length (taps)
P_LEN = 200           # primary path FIR length (taps)
S_LEN = 64            # secondary path FIR length (taps)

MODE_NAMES = ("quiet", "babble", "traffic", "wind")
N_MODES = len(MODE_NAMES)


@dataclass(frozen=True)
class ModeParams:
    """Per-mode (Q, R) pair used by each KF in the IMM bank."""
    name: str
    sigma_q2: float   # process noise variance (diagonal of Q)
    sigma_r2: float   # measurement noise variance


# Per-sample (Q, R) calibrated at FS=16 kHz, L=64, unit-RMS reference.
# Each Q is the variance of the random-walk drift *per sample*, not per second.
#
# v5 calibration: each mode's (Q, R) was picked by scripts/03c_calibrate_consistency.py,
# which runs that mode's noise on its *own* single, static plant and selects the
# pair with the best NR among those whose time-averaged NEES sits in the chi^2
# band. So consistency holds *per mode, on a static plant* — NOT for the combined
# IMM estimator on the dynamic mode-switching scenario.
#
# Caveats verified empirically (scripts/diag_v5_tension.py, diag_v5_mixing.py):
#   * The dynamic combined NR (+16 dB) comes almost entirely from IMM *mixing*,
#     not from mode classification: the agile 'babble' filter (Q=1e-2) tracks
#     best and mixing re-seeds the slow modes from it every sample. Disabling
#     mixing (Pi=I) collapses NR to ~+1.5 dB. The mode posterior itself
#     collapses onto the lowest-S mode ('traffic'), so per-sample mode-tracking
#     accuracy is low (~25-30%) and is *not* the right KPI for this system.
#   * Dynamic NEES looks "overconfident" only because the synthetic scenario
#     swaps the acoustic plant instantaneously every few seconds; the spike is a
#     post-switch transient. Steady-state NEES (>=1 s after each switch) is in
#     fact fine-to-underconfident (~21 combined). See 07_consistency_test.py.
MODE_PARAMS = (
    ModeParams("quiet",   sigma_q2=1e-10, sigma_r2=100.0),
    ModeParams("babble",  sigma_q2=1e-2,  sigma_r2=10.0),
    ModeParams("traffic", sigma_q2=1e-10, sigma_r2=10.0),
    ModeParams("wind",    sigma_q2=1e-4,  sigma_r2=100.0),
)

# Markov mode transition matrix: high self-persistence, uniform leakage.
PI_DIAG = 0.95
PI_OFF = (1.0 - PI_DIAG) / (N_MODES - 1)
