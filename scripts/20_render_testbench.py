"""Single, coherent ANC test bench — the canonical "virtual but realistic"
listening + measurement environment (CLI / file-rendering front-end).

The orchestration core (scenario algorithms, metrics, backend policy) lives in
`src/testbench.py` so the live Streamlit app shares the exact same code. This
script adds the CLI, the WAV-set rendering with shared gain, and the plots.

One command renders the whole experience into figures/testbench/:
  * 3-level LOUDNESS LADDER (open ear -> ANC off -> ANC on), one shared gain;
  * the same under music (music-aware: controller adapts on noise, you hear
    music + residual);
  * the full metric battery (controller NR, dBA loudness drop, band-split NR,
    musical-noise/transient indices, SI-SDR/PSNR);
  * third-octave NR overlay + spectrogram comparison.

Realism layers: recorded noise (ANC_NOISE_SOURCE=recorded), gradual switches
(--crossfade), virtual headphone (src/headphone.py), common gain.

Run:
    python -m scripts.20_render_testbench
    python -m scripts.20_render_testbench --snr 0 --crossfade 0.75
    python -m scripts.20_render_testbench --no-music
    $env:ANC_NOISE_SOURCE = "recorded"; python -m scripts.20_render_testbench
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg
from src import scenario as sc
from src import headphone as hp
from src import perceptual as pc
from src import testbench as tb
from scripts.music_noise_test import load_music_clip


# ---- printing ---------------------------------------------------------------

def print_table(metrics: dict[str, dict], have_music: bool) -> None:
    print()
    print("=" * 108)
    cols = (f"{'Algorithm':<22}{'CtrlNR':>8}{'dBA↓(ANC)':>11}{'dBA↓(tot)':>11}"
            f"{'NR<1.2k':>9}{'NR>1.2k':>9}{'MusNoise':>9}{'Burst':>7}")
    if have_music:
        cols += f"{'SI-SDR':>8}"
    print(cols)
    print(f"{'':<22}{'[dB]':>8}{'[dB]':>11}{'[dB]':>11}{'[dB]':>9}{'[dB]':>9}"
          f"{'(low=ok)':>9}{'(low=ok)':>7}" + ("    [dB]" if have_music else ""))
    print("-" * 108)
    for tag, m in metrics.items():
        row = (f"{tb.ALGO_LABELS[tag]:<22}{m['ctrl_nr_db']:>+8.2f}"
               f"{m['dba_reduction_anc']:>+11.2f}{m['dba_reduction_total']:>+11.2f}"
               f"{m['anc_band_nr_db']:>+9.2f}{m['passive_band_nr_db']:>+9.2f}"
               f"{m['musical_noise']:>9.2f}{m['flux_burstiness']:>7.2f}")
        if have_music:
            row += f"{m.get('si_sdr_db', float('nan')):>+8.2f}"
        print(row)
    print("=" * 108)
    print("  CtrlNR     = raw controller noise reduction (energy removed)")
    print("  dBA↓(ANC)  = perceived loudness drop, ANC on vs passive earbud  <-- the 'feel' headline")
    print("  dBA↓(tot)  = perceived loudness drop vs no earbud at all")
    print("  NR<1.2k / >1.2k = where the reduction lives (active band vs passive band)")
    print("  MusNoise / Burst = musical-noise & transient indices (lower = more natural)")
    if have_music:
        print("  SI-SDR     = music recovery quality under the noise (higher = better)")


# ---- plots ------------------------------------------------------------------

def plot_third_octave(s, algos, out_dir):
    fig, ax = plt.subplots(figsize=(11, 5))
    for tag, e in algos.items():
        fc, nr = pc.third_octave_nr(s.d, e, s.fs, f_lo=40.0, f_hi=8000.0)
        ax.semilogx(fc, nr, "-o", ms=3, lw=1.4, label=tb.ALGO_LABELS[tag])
    ax.axvline(hp.ANC_BAND_FC, color="grey", ls="--", lw=0.8)
    ax.text(hp.ANC_BAND_FC * 1.05, ax.get_ylim()[1] * 0.92,
            "active-ANC band edge", color="grey", fontsize=8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("NR [dB] (1/3-octave)")
    ax.set_title("Where the noise reduction lives — active ANC wins the lows, passive owns the highs")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = out_dir / "20_third_octave_nr.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"saved {out}")


def plot_spectrograms(s, algos, out_dir):
    from scipy.signal import spectrogram
    show = [("Open ear (no earbud)", s.d),
            ("Earbud, ANC off", hp.passive_isolation(s.d, s.fs)),
            ("ANC on — IMM v5", hp.render_eardrum(s.d, algos["v5"], fs=s.fs)["on"]),
            ("ANC on — NLMS", hp.render_eardrum(s.d, algos["nlms"], fs=s.fs)["on"])]
    fig, axes = plt.subplots(len(show), 1, figsize=(11, 10), sharex=True, sharey=True)
    f0, t0, S0 = spectrogram(s.d, fs=s.fs, nperseg=512, noverlap=384)
    db0 = 10 * np.log10(S0 + 1e-12)
    vmin, vmax = np.percentile(db0, 5), np.percentile(db0, 99)
    im = None
    for ax, (name, sig) in zip(axes, show):
        f_, t_, S = spectrogram(sig, fs=s.fs, nperseg=512, noverlap=384)
        im = ax.pcolormesh(t_, f_, 10 * np.log10(S + 1e-12), cmap="magma",
                           vmin=vmin, vmax=vmax, shading="auto")
        ax.set_ylabel("Hz"); ax.set_title(name, fontsize=10); ax.set_ylim(0, 6000)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Virtual-headphone spectrograms: lows vanish under ANC, highs handled by the passive seal")
    fig.colorbar(im, ax=axes, shrink=0.85, label="dB", location="right", pad=0.02)
    out = out_dir / "20_spectrograms.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"saved {out}")


# ---- main -------------------------------------------------------------------

def _match_len(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == n:
        return x
    return x[:n] if len(x) > n else np.pad(x, (0, n - len(x)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--snr", type=float, default=0.0,
                    help="music-vs-noise SNR in dB (0 = equal power)")
    ap.add_argument("--crossfade", type=float, default=0.75,
                    help="mode-transition crossfade in seconds (0 = hard cuts)")
    ap.add_argument("--music", type=str,
                    default=str(ROOT / "musics" / "Ara Durak - Afra.flac"))
    ap.add_argument("--no-music", action="store_true",
                    help="render the noise-only loudness ladder only")
    args = ap.parse_args()

    # Tables use µ / ŵ / arrows; force UTF-8 so a cp1252/cp1254 console won't crash.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    out_dir = ROOT / "figures" / "testbench"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Noise source: {cfg.NOISE_SOURCE}   crossfade: {args.crossfade:.2f} s")
    rng = np.random.default_rng(args.seed)
    s = sc.build_scenario(segments=tb.DEFAULT_TRAJ, rng=rng,
                          mode_conditioned_plants=True, crossfade_sec=args.crossfade)
    fs = s.fs
    print(f"Scenario: {len(s.d)/fs:.2f} s ({len(s.d)} samples)")

    music = None
    if not args.no_music:
        mpath = Path(args.music)
        if mpath.exists():
            noise_rms = float(np.sqrt(np.mean(s.d ** 2)))
            target_rms = noise_rms * (10.0 ** (args.snr / 20.0))
            music = load_music_clip(mpath, target_fs=fs, duration_sec=len(s.d) / fs,
                                    target_rms=target_rms)
            music = _match_len(music, len(s.d))
            print(f"Music: {mpath.name}  (input SNR {args.snr:+.1f} dB)")
        else:
            print(f"Music file not found ({mpath.name}); rendering noise-only.")

    print(f"Algorithms via {tb.backend_in_use()} backend (+ Python variants)...")
    algos = tb.run_algorithms(s)

    # ---- WAV set 1: noise-only loudness ladder (one shared gain) ----
    base = hp.render_eardrum(s.d, algos["v5"], music=None, fs=fs)
    noise_set = {"noise_open": base["open"], "noise_off": base["off"]}
    for tag, e in algos.items():
        noise_set[f"noise_on_{tag}"] = hp.render_eardrum(s.d, e, music=None, fs=fs)["on"]
    g_noise = hp.write_common_gain(noise_set, fs, out_dir)
    print(f"\nLoudness ladder -> {out_dir} (shared gain {g_noise:.3g}): "
          f"noise_open, noise_off, noise_on_<algo>")

    # ---- WAV set 2: music + noise (one shared gain) ----
    if music is not None:
        mbase = hp.render_eardrum(s.d, algos["v5"], music=music, fs=fs)
        music_set = {"music_ref": mbase["ref"], "music_off": mbase["off"]}
        for tag, e in algos.items():
            music_set[f"music_on_{tag}"] = hp.render_eardrum(s.d, e, music=music, fs=fs)["on"]
        g_music = hp.write_common_gain(music_set, fs, out_dir)
        print(f"Music set      -> {out_dir} (shared gain {g_music:.3g}): "
              f"music_ref, music_off, music_on_<algo>")

    # ---- metrics ----
    metrics = tb.compute_metrics(s, algos, music)
    print_table(metrics, have_music=music is not None)

    meta = {
        "noise_source": cfg.NOISE_SOURCE,
        "crossfade_sec": args.crossfade,
        "snr_db": args.snr,
        "seed": args.seed,
        "duration_sec": len(s.d) / fs,
        "music_file": Path(args.music).name if music is not None else None,
        "algo_labels": tb.ALGO_LABELS,
        "anc_band_fc": hp.ANC_BAND_FC,
        "metrics": metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved {out_dir / 'metrics.json'}")

    plot_third_octave(s, algos, out_dir)
    plot_spectrograms(s, algos, out_dir)

    print("\nListen order (loudness ladder): noise_open -> noise_off -> noise_on_v5")
    print("Then music: music_ref -> music_off -> music_on_v5 -> music_on_nlms ...")


if __name__ == "__main__":
    main()
