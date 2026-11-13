# The range guard's measured gaps at fb4dfe0: 2026-11-07

`benchmarks/probe_guard_gaps.c` at commit fb4dfe0 on `release/v1.6.0`, after the
guard changes merged into 1.6.0 (entry and C′ floors, each complex part on its own
scale, magnitudes below 2⁻¹⁰⁴², and BLAS-2's per-part guard). The same calls at
8b48826, before those changes, are in
[`../2026-11-06-8b48826-pretag/guard-gaps/`](../2026-11-06-8b48826-pretag/guard-gaps/).
Correctness only: no timing is read from these runs.

| | |
|---|---|
| Machine | MacBook Pro (Mac14,6), Apple M2 Max, 38-core GPU, 64 GB, macOS 27.0.1 |
| Library | fb4dfe0 (`v1.5.1-55-gfb4dfe0`), `make clean && make -j8 lib` in a fresh clone with 0 modified tracked files, Apple clang 17.0.0. Every run's banner read `apple-bottom 1.6.0 (fb4dfe06a197)`. |
| Runs | `default.csv` (`AB_MODE` unset), `forced_gpu.csv` (`AB_MODE=gpu AB_MIN_GPU_DIM=0`), `cpu.csv` (`AB_MODE=cpu`), one after another under machlock's exclusive bench lock, on AC, with other sessions' CPU jobs running (load about 60) |

Columns: the call, the data, the path it took, and for each part of the result the
largest error at sampled outputs against a Dot2 reference, in u² = 2⁻⁴⁸ of that
part's own scale (the probe's header defines the data and the scales).

## The calls 8b48826 let through

Every call that passed the guard at 8b48826 with a result degraded by FP32's
exponent range now runs on Accelerate, or on the GPU within 2.2 u² (uniform data:
at most 2.3 u² at 4096³). "Default" is the routing with `AB_MODE` unset; "forced"
is `AB_MODE=gpu AB_MIN_GPU_DIM=0`.

| Data | Call | Route | 8b48826 | fb4dfe0 |
|---|---|---|---|---|
| Imaginary parts 2⁻¹⁰⁰ of the real parts (`imag-100`) | `zgemm_` 'T','T' 4096³ | default | GPU, 1.4 × 10⁵ u² | Accelerate, 1.75 u² |
| | `zgemv_` 'N' 3072 | default | GPU, 2.9 × 10⁵ u² | Accelerate, 1.66 u² |
| | `zhemv_` 'U' 2048 | default | GPU, 3.7 × 10⁵ u² | Accelerate, 1.10 u² |
| | `zgemm_` 'N','N' 8 × 8 × 2 | forced | GPU, 2.7 × 10⁶ u² | Accelerate, 0.06 u² |
| | `zgemm_` 'N','N' 544 × 481 × 600 | forced | GPU, 2.9 × 10⁵ u² | Accelerate, 0.75 u² |
| Imaginary parts 2⁻¹³⁰ of the real parts (`imag-130`) | the first four calls | default, forced | GPU, 2.8 × 10¹⁴ u²: the imaginary part is zero | Accelerate, at most 1.75 u² |
| A near 2⁻¹⁰⁶⁰, B and x near 2¹⁰⁰⁰ (`tiny`) | `zgemm_` 'T','T' 4096³ | default | GPU, 1.9 × 10¹³ u²: zero | GPU, 1.87 u² |
| | `zgemm_` 'N','N' 8 × 8 × 2 | forced | GPU, 2.8 × 10¹⁴ u²: zero | GPU, 2.18 u² |
| | `dgemm_` 'N','N' 8 × 8 × 2 | forced | GPU, 2.8 × 10¹⁴ u²: zero | GPU, 0.98 u² |
| C near 2⁻⁹⁸ or 2⁻¹⁰⁰ with beta 1 or 2²⁰, in elements without products (`smallc`) | `dgemm_` 'N','N' 8 × 8 × 2 | forced | GPU, 5.2 × 10⁵ u² | Accelerate, 0 |
| C near 2⁻¹³⁰, beta 1 | `dgemm_` 'N','N' 8 × 8 × 2 | forced | GPU, 2.8 × 10¹⁴ u²: zero | Accelerate, 0 |

## Other changes in routing

- **Imaginary parts 2⁻⁸⁰ of the real parts (`imag-80`)** now run on Accelerate in
  every row that reached the GPU at 8b48826 (`zgemm_` 'T','T' 4096³, `zgemv_`,
  `zhemv_`, and `zgemm_` 8 × 8 × 2 forced), where their errors were at most 36.5 u².
  Their products fall below 2⁻⁷⁸, where the GPU's flushing can cost a part more
  than the guard allows.
- **Every other row** is unchanged from 8b48826, path and errors to the digits
  printed, in all three runs, including uniform and `spread` data and the
  `dev_zgemm` rows (the device API's GEMMs have no guard and no fallback). The
  `spread` imaginary parts (up to 58 u²) are 3M's, which `docs/vv/LIMITATIONS.md`
  (Complex products) describes.
- **`AB_MODE=cpu`** runs every BLAS-ABI call on Accelerate, at most 1.75 u²; the
  `dev_zgemm` rows stay on the GPU, since the device API does not read `AB_MODE`.
