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

## Copyright-safe picks (what we used)

If you want music that is safe to redistribute or publish in a demo video:

| Work | License | Source |
|---|---|---|
| J.S. Bach — *Open Goldberg Variations* (Kimiko Ishizaka, piano) | **CC0** — both the recording *and* the composition are public domain | https://archive.org/details/OpenGoldbergVariations |
| "Funkorama" — Kevin MacLeod | **CC BY 4.0** — attribution required | https://incompetech.com |

Why these for an ANC demo: the slow **Aria** exposes musical-noise artifacts in
the gaps between notes; a fast variation tests music preservation under dense
content; Funkorama's bass and drums sit inside the active-ANC band (<1.2 kHz) —
the hardest case for cancelling noise without touching the music.

> ⚠️ Most classical recordings you find online (e.g. LP rips on archive.org)
> are **not** free even though the composition is public domain — the *recording*
> is still copyrighted. The Open Goldberg project is the notable exception:
> it was crowd-funded specifically to release the recording as CC0.

CC BY attribution line for Funkorama (e.g. in a video description):

> "Funkorama" — Kevin MacLeod (incompetech.com). Licensed under Creative
> Commons: By Attribution 4.0. https://creativecommons.org/licenses/by/4.0/
