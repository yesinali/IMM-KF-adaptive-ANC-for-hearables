# Recorded noise samples (optional, drop-in)

Place real ambient recordings here to render the test bench with authentic
noise instead of the synthetic generators. Then run with:

```powershell
$env:ANC_NOISE_SOURCE = "recorded"
python -m scripts.20_render_testbench
```

(or set `NOISE_SOURCE` in `src/config.py`). Modes that have no clips fall back
to synthetic automatically, so a partial set is fine.

## Layout

```
noise_samples/
    quiet/    *.wav | *.flac | *.ogg | *.mp3
    babble/
    traffic/
    wind/
```

Any sample rate / channel count works — clips are mixed to mono, resampled to
16 kHz, and normalized to unit RMS by `src/noise_recorded.py`. A few minutes per
mode is plenty (random slices are drawn per segment).

## The exact set used in the project (quick start)

No audio ships with this repository (size + licensing hygiene), but the set we
evaluated with is fully open and takes minutes to fetch:

| Mode | File(s) | Source |
|---|---|---|
| `quiet`   | DEMAND **OOFFICE**, channel 1 of the 16 kHz zip | office ambience |
| `babble`  | DEMAND **PCAFETER**, ch. 1 | cafeteria |
| `traffic` | DEMAND **STRAFFIC**, ch. 1 + a few ESC-50 `engine`/`train` clips | street traffic |
| `wind`    | DEMAND **NFIELD**, ch. 1 + a few ESC-50 `wind` clips | open field |

- **DEMAND** (CC BY 4.0 — Thiemann, Ito & Vincent, 2013): download the
  `<ENV>_16k.zip` files from https://zenodo.org/records/1227121, extract only
  `ch01.wav` from each, and drop them into the mode folders above.
- **ESC-50** (CC BY-NC 3.0 — Piczak, 2015): individual 5 s clips can be fetched
  straight from `https://github.com/karolpiczak/ESC-50` (`audio/` folder; pick
  filenames by category via `meta/esc50.csv`) — no need to download the full
  dataset.

## Full public-dataset mapping (to extend the set)

| Mode | DEMAND category | ESC-50 category |
|---|---|---|
| `quiet`   | `OOFFICE`, `OHALLWAY`, `DLIVING` (low HVAC floor) | `clock_tick`, `breathing` (quiet beds) |
| `babble`  | `PCAFETER`, `OMEETING`, `SPSQUARE` | — (no crowd class; use DEMAND) |
| `traffic` | `TCAR`, `TMETRO`, `STRAFFIC`, `TBUS` | `engine`, `train`, `car_horn` |
| `wind`    | `NPARK`, `NFIELD`, `NRIVER` (outdoor) | `wind` |

> Keep the clips you use noted in the report's data section so the evaluation is
> reproducible.
