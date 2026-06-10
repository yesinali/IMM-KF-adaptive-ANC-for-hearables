"""Smoke tests: the core pipeline builds, runs, and cancels noise.

Kept deliberately short (~15 s total) so they can run on every push.
Run from the repository root:  python -m pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg                       # noqa: E402
from src import scenario as sc                      # noqa: E402
from src import c_backend                           # noqa: E402
from src.filters import NLMSFilter, KalmanANCFilter  # noqa: E402
from src.imm import IMMKalmanANC                    # noqa: E402
from src.anc import simulate_anc                    # noqa: E402
from src.metrics import overall_nr_db               # noqa: E402


@pytest.fixture(scope="module")
def short_scenario():
    """1 s quiet + 1 s traffic, mode-conditioned plants, fixed seed."""
    rng = np.random.default_rng(7)
    segments = [sc.ScenarioSegment("quiet", 1.0), sc.ScenarioSegment("traffic", 1.0)]
    return sc.build_scenario(segments=segments, rng=rng, mode_conditioned_plants=True)


def test_scenario_shapes(short_scenario):
    s = short_scenario
    n = int(2.0 * cfg.FS)
    assert len(s.d) == len(s.x_filt) == len(s.mode_labels) == n
    assert np.all(np.isfinite(s.d))


def test_nlms_reduces_noise(short_scenario):
    res = simulate_anc(short_scenario, NLMSFilter(L=cfg.L, mu=0.10))
    nr = overall_nr_db(short_scenario.d, res["e"])
    assert np.all(np.isfinite(res["e"]))
    assert nr > 1.0, f"NLMS should cancel some noise, got {nr:+.2f} dB"


def test_kalman_runs(short_scenario):
    res = simulate_anc(short_scenario,
                       KalmanANCFilter(L=cfg.L, sigma_q2=1e-8, sigma_r2=100.0))
    assert np.all(np.isfinite(res["e"]))


def test_imm_reduces_noise_and_exports_posterior(short_scenario):
    res = simulate_anc(short_scenario,
                       IMMKalmanANC(L=cfg.L, likelihood_window=200), log_mu=True)
    nr = overall_nr_db(short_scenario.d, res["e"])
    assert nr > 1.0, f"IMM should cancel some noise, got {nr:+.2f} dB"
    mu = res["mu_history"]
    assert mu.shape == (len(short_scenario.d), cfg.N_MODES)
    np.testing.assert_allclose(mu.sum(axis=1), 1.0, atol=1e-6)


def test_crossfade_zero_is_legacy():
    """crossfade_sec=0 must be bit-identical to the legacy hard-switch path."""
    segs = [sc.ScenarioSegment("quiet", 0.5), sc.ScenarioSegment("wind", 0.5)]
    a = sc.build_scenario(segments=segs, rng=np.random.default_rng(3),
                          mode_conditioned_plants=True)
    b = sc.build_scenario(segments=segs, rng=np.random.default_rng(3),
                          mode_conditioned_plants=True, crossfade_sec=0.0)
    np.testing.assert_array_equal(a.d, b.d)


@pytest.mark.skipif(not c_backend.available_backends(),
                    reason="C port not compiled (run `make pure` in c_imm/)")
def test_c_port_matches_python(short_scenario):
    s = short_scenario
    backend = c_backend.available_backends()[0]
    cres = c_backend.run(s, "NLMS", {"mu": 0.10}, backend=backend)
    e_c = c_backend.pick_residual(cres, "NLMS")
    e_py = simulate_anc(s, NLMSFilter(L=cfg.L, mu=0.10))["e"]
    np.testing.assert_allclose(e_c, e_py, atol=1e-9)
