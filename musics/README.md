# Program music (bring your own)

No audio ships with this repository (copyright + size). Drop one or more music
files here to use the **Music & Feel** page and the music side of the test bench:

```
musics/
    your_track.flac
    another.wav
```

Any `.flac` / `.wav` at any sample rate works — clips are mixed to mono and
resampled to 16 kHz automatically.

- **Music & Feel page** — if this folder is empty it falls back to a file-upload
  widget, so the app still runs with nothing here.
- **Test bench** — `python -m scripts.20_render_testbench` looks for a file under
  `musics/` (override with `--music <path>`); use `--no-music` to render the
  noise-only loudness ladder instead.
