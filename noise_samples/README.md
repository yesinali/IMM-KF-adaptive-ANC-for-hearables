DEMAND/ESC-50# Recorded noise samples (optional, drop-in)

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

## Suggested public-dataset mapping

Nothing is downloaded automatically (the datasets are large). Grab a handful of
clips and drop them in.

| Mode | DEMAND category | ESC-50 category |
|---|---|---|
| `quiet`   | `OOFFICE`, `OHALLWAY`, `DLIVING` (low HVAC floor) | `clock_tick`, `breathing` (quiet beds) |
| `babble`  | `PCAFETER`, `OMEETING`, `SPSQUARE` | `crowd` (overlapping talkers) |
| `traffic` | `TCAR`, `TMETRO`, `STRAFFIC`, `TBUS` | `engine`, `train`, `car_horn` |
| `wind`    | `NPARK`, `NFIELD`, `NRIVER` (outdoor) | `wind` |

- **DEMAND** — Diverse Environments Multichannel Acoustic Noise Database
  (16 kHz, 5-minute multichannel clips; use channel 1). https://zenodo.org/records/1227121
- **ESC-50** — Environmental Sound Classification, 5 s clips.
  https://github.com/karoldvl/ESC-50

> Keep the clips you use noted in the report's data section so the evaluation is
> reproducible.
