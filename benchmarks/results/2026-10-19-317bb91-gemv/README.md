# BLAS-2 (DGEMV, ZGEMV, ZHEMV) on the M2 Max, 2026-10-19

`build/bench_gemv` (`benchmarks/bench_gemv.c`) under `machlock.py bench`
(exclusive use of the machine), library built from the working tree on
317bb91 that became the BLAS-2 commit (`ab_build_info()`:
`1.4.0 (317bb917c8b4-dirty)`).

Machine: MacBook Pro (Mac14,6), Apple M2 Max, 38-core GPU, 8P+4E CPU, 64 GB,
macOS 27.0.1 (26A434), macOS SDK 26.0, Apple clang 17.0.0, AC power.

| File | Run | |
|---|---|---|
| `bench_gemv.txt`, `bench_gemv.csv` | 00:45, load 2.3 | `bench_gemv --reps 9`: every routine and op at 4096², 8192² and 16384², four ways (below) |
| `sweep_square.txt`, `sweep_square.csv` | 00:47 | `--sweep`: staged GPU against Accelerate, square 256–16384 |
| `sweep_rect.txt`, `sweep_rect.csv` | 00:48 | `--rect --routines dgemv,zgemv`: 4096 × 1024, 16384 × 1024, 65536 × 256, 32768 × 2048, 131072 × 512, 262144 × 1024 and their transposes |
| `sweep_square_run1.*`, `sweep_rect_run1.*` | 00:06 and 00:08, load 2.7 and 3.6 | the same sweeps on the working tree before its last clean-up: the same kernels, with bitwise-identical results on 56 test calls; host-side staging of x and y has since been made faster by microseconds per call |

All calls compute y = A x (alpha 1, beta 0) with lda = M and unit
increments.

**`bench_gemv.csv`.** Each shape is timed four ways, as the median wall time
of 9 calls after 2 warm-up calls; GPU modes warm up for at least 0.5 s,
because the GPU raises its clocks only after about 0.3 s of sustained work:
- `resident`: `ab_dev_*gemv` / `ab_dev_zhemv` on A registered once with
  `ab_dev_register` (no copy);
- `staged`: `ab_*gemv_blas` / `ab_zhemv_blas` with the GPU path forced
  (`AB_MODE=gpu`), mapping A for each call, what a one-shot `dgemv_` costs;
- `accel`: Accelerate `cblas_*`;
- `cloopN`: plain C on N = 4, 8 or 12 threads (`dispatch_apply`), the
  fastest of the three.
`gbs` is the bytes of A the routine references (ZHEMV: one triangle) per
second; `err` the largest |y_i − ref_i| / (|A| |x|)_i over 64 sampled outputs
against a compensated FP64 reference.

**Sweeps.** Staged GPU against Accelerate in the same process, 3 warm-ups
and the median of 7 calls each; `speedup` is Accelerate time / GPU time.
These are the data behind the `AB_MODE=auto` rows (`kRoutes`) in
`src/blas2_abi.c`: a row covers only shapes measured at least 1.10× faster
in both runs with no slower shape inside it, and `tests/test_blas2.c`
checks the rule against all four sweep files. CSV columns: routine, op, m,
n, lda, `gpu_path`, `gpu_median_ms`, `accel_median_ms`, `speedup`.

| Staged GPU vs Accelerate | where ≥ 1.10× in both runs | elsewhere |
|---|---|---|
| DGEMV `'N'` | 1.63–1.78× at 2048², 2.8–3.7× from 3072² to 16384²; rectangles of 2048² elements and up 1.12–3.46× | ≤ 0.71× up to 1536² |
| DGEMV `'T'` | — | 1.04–1.32× from 4096², 12288² at 1.12× and 1.096×: too close to route; rectangles 0.55–1.25× |
| ZGEMV `'N'` | 1.22–1.83× from 2048² to 12288²; rectangles with both sides ≥ 512 1.21–1.81× | 16384² 1.10× and 1.32× (1.02× in an earlier full run, not kept); 256 × 65536 1.07–1.08× |
| ZGEMV `'T'`, `'C'` | — | 0.04–0.79× at every shape |
| ZHEMV `'U'`, `'L'` | 2.34–4.27× from 2048 to 16384 | 0.88–1.25× at 1536 |

The ZHEMV rows here measure the first, two-pass kernel. The kernel that
replaced it reads the triangle once; its measurements, and the data behind
the ZHEMV routing row since, are in `../2026-10-19-1bc503a-zhemv/`.

The GEMV rows measure kernels that converted A's elements with branches
and `ldexp`. Their successors convert as the single-read ZHEMV kernel
does, with bitwise-identical results; their measurements, and the data
behind the GEMV routing rows since, are in `../2026-10-19-ba345df-gemv/`.

Mapping A costs about 10 ms/GB for a 1 GB matrix and 22 ms/GB for a 4 GB
one, so the GPU's lead in one-shot calls shrinks at the largest sizes; the
rows stop at the largest size measured as a win. A matrix kept resident
pays none of it: see `README.md`, "BLAS-2".
