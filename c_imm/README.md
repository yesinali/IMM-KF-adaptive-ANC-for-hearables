# `c_imm/` — C port of the NLMS / KF / IMM filter bank

Bit-for-bit C reimplementation of [src/filters.py](../src/filters.py) and
[src/imm.py](../src/imm.py), used to measure the real-time cost of the IMM-KF
ANC controller and verify our NumPy reference against an independent
implementation.

Two backends compile from the same `filters.c` and `main.c`:

| target | backend | links |
|---|---|---|
| `imm_pure.exe` | hand-written scalar C (`linalg_pure.c`) | `-lm` |
| `imm_blas.exe` | OpenBLAS (`linalg_blas.c`) | `-lopenblas -lm` |

All linear-algebra primitives are routed through `linalg.h` so the only
difference between the two builds is one source file.

## Build

Requires MSYS2 MinGW-64 (gcc, make). For the BLAS build also
`mingw-w64-x86_64-openblas`.

```bash
cd c_imm
make pure                  # imm_pure.exe   (no extra dependencies)
make blas                  # imm_blas.exe   (requires OpenBLAS)
make all                   # both
```

If `make` reports `Cannot create temporary file in C:\WINDOWS\`, point the
temp dir at a writable location:

```bash
TMPDIR=/tmp TMP=/tmp TEMP=/tmp make pure
```

## Run

```bash
# 1. Dump a deterministic 20 s scenario from the Python side
python -m scripts.c_port_dump   # writes c_imm/scenario.bin + .npz

# 2. Time each backend
./imm_pure.exe scenario.bin out_pure.bin
./imm_blas.exe scenario.bin out_blas.bin

# 3. Verify numerics + speed against NumPy
python -m scripts.c_port_compare
```

`c_port_compare.py` reports per-filter latency [µs/sample] and the max-abs
residual difference vs NumPy.

## File layout

| file | role |
|---|---|
| `linalg.h` | BLAS-style primitive interface (`matvec`, `ger`, `axpy`, ...) |
| `linalg_pure.c` | portable scalar implementation |
| `linalg_blas.c` | OpenBLAS wrappers (cblas_dgemv / dger / daxpy / ...) |
| `filters.h` | `nlms_t`, `kf_t`, `imm_t` structs + step/init/free API |
| `filters.c` | NLMS, single-mode KF, IMM-KF (mirrors the Python step-by-step) |
| `main.c` | binary I/O, runs all three filters end-to-end, emits timing JSON |
| `Makefile` | two build targets, `-O3 -march=native -ffast-math` |
| `scenario.bin` | dumped by `scripts/c_port_dump.py` (input) |
| `out_*.bin` | residuals written back to disk for cross-impl diff |

## Binary file format (little-endian)

Input (`scenario.bin`):

```
int32   N             # samples
int32   L             # FIR length
int32   M             # IMM modes (4)
int32   FS            # sample rate
int32   W_lik         # IMM likelihood smoothing window
float64 mu_nlms
float64 Q_kf, R_kf
float64 Pi[M*M]
float64 Q_imm[M]
float64 R_imm[M]
float64 xf[N]
float64 d[N]
```

Output (`out_pure.bin` / `out_blas.bin`):

```
int32   N
float64 e_nlms[N]
float64 e_kf[N]
float64 e_imm[N]
```

## Notes

- Tap-delay buffers live inside each filter and are shifted by `memmove`; the
  caller hands in one fresh `xf[k]` per step. This matches the Python loop in
  [src/anc.py](../src/anc.py:86) line-by-line.
- The IMM step is a literal translation of [src/imm.py](../src/imm.py:51), so
  numeric diffs against NumPy should be at machine epsilon (< 1e-12).
- `-ffast-math` is on for raw throughput; turn it off
  (`CFLAGS_EXTRA=-fno-fast-math`) if you need IEEE-bit-exact agreement.
