# GEMV with ZHEMV's element conversion, M2 Max, 2026-10-19

`build/bench_gemv --routines dgemv,zgemv` (`benchmarks/bench_gemv.c`) under
`machlock.py bench` (exclusive use of the machine), library built from the
working tree on ba345df that became the commit switching the GEMV kernels
to the conversion the ZHEMV kernel already used (`ab_build_info()`:
`1.4.0 (ba345df490bb-dirty)`; the routing table was set from these runs
afterwards, which `bench_gemv` does not use). The `*_before` files and the
`before` rows of `bench_gemv_alternating.csv` are the same benchmark linked
against the library at ba345df, whose GEMV kernels still converted A's
elements with branches and `ldexp` (built from `git archive ba345df`,
identity `1.4.0 (ba345df-base)`), run in the same lock for a comparison
under the same conditions. The two libraries give bitwise-identical results:
666 calls compared through `ab_dev_dgemv`, `ab_dev_zgemv` and
`ab_dev_zhemv`, every op, shapes up to 8192², ties, carries, FP32-subnormal
and wide-range elements, 511 of them on the GPU.

Machine: MacBook Pro (Mac14,6), Apple M2 Max, 38-core GPU, 8P+4E CPU, 64 GB,
macOS 27.0.1 (26A434), macOS SDK 26.0, Apple clang 17.0.0, AC power.

| File | Run | |
|---|---|---|
| `bench_gemv_run1.*`, `bench_gemv_before_run1.*`, `bench_gemv_before.*`, `bench_gemv.*` | 21:17–21:20 in that order (after, before, before, after), load 4.3–6.2 | `--reps 9`: every op at 4096², 8192² and 16384², four ways (as in `../2026-10-19-317bb91-gemv/README.md`) |
| `bench_gemv_alternating.csv` | 21:21–21:25, load 3.6–5.8 | `--sizes 2048,4096,8192 --reps 41`, three rounds of the two builds (after, before; before, after; after, before); `round` and `build` columns added |
| `sweep_square_run1.*`, `sweep_rect_run1.*` | 21:25–21:27, load 2.7–2.9 | `--sweep`, `--rect`: staged GPU against Accelerate, square 256–16384 and the rectangles of `../2026-10-19-317bb91-gemv/` |
| `sweep_square.*`, `sweep_rect.*` | 21:27–21:28, load 2.8–4.7 | the same, a second time |

Most of the load is the benchmark's own CPU modes (Accelerate, the threaded
C loop) and Spotlight indexing. All calls compute y = A x (alpha 1, beta 0)
with lda = M and unit increments; columns and modes as in
`../2026-10-19-317bb91-gemv/README.md`.

**What changed.** Each element of A is still converted from FP64 to
double-float once, in registers. The conversion now uses selects where it
returned early (zero and FP64-subnormal elements, elements beyond FP32's
range) and two exact multiplies where it called `ldexp`, as ZHEMV's has
since `../2026-10-19-1bc503a-zhemv/`; elements beyond FP32's range become
NaN instead of ±Inf, which sends the call to Accelerate as before.

**Resident** (`ab_dev_*gemv` on a registered matrix): time before / time
after, so above 1 is faster after. 2048²–8192²: the median of the three
rounds of `bench_gemv_alternating.csv`, the rounds' range in brackets;
16384²: the two pairs of the `--reps 9` runs.

| y = A x | 2048² | 4096² | 8192² | 16384² |
|---|---:|---:|---:|---:|
| DGEMV `'N'` | 1.22 (1.07–1.29) | 1.08 (1.08–1.11) | 1.06 (1.04–1.06) | 1.04, 1.05 |
| DGEMV `'T'` | 1.25 (1.10–1.36) | 1.00 (0.94–1.03) | 1.01 (1.00–1.02) | 1.02, 1.02 |
| ZGEMV `'N'` | 1.10 (1.03–1.31) | 1.02 (1.02–1.12) | 1.06 (1.03–1.07) | 1.03, 1.04 |
| ZGEMV `'T'` | 0.93 (0.92–1.14) | 1.08 (1.06–1.09) | 1.02 (1.01–1.03) | 1.01, 1.04 |
| ZGEMV `'C'` | 1.21 (0.90–1.22) | 1.02 (0.96–1.05) | 1.01 (0.99–1.05) | 1.01, 1.01 |

The `'N'` kernels gain 2–8% from 4096², the `'T'` and `'C'` kernels 0–8%,
less than ZHEMV did (the conversion bounded that kernel: removing it
doubled its speed). GEMV was already near the memory's limit: at 16384²
the `'T'` and `'C'` kernels read A at 353–359 GB/s of the M2 Max's 400 and
the `'N'` kernels at 317–320. At 2048² a call takes 0.22–0.47 ms, about
0.2 ms of it submit and wait, and single rounds swing by up to 30% either
way. The `--reps 9` pairs at 4096² and 8192² agree within their noise
(0.91–1.17 and 0.97–1.06). Staged calls, which also map A, came out at
0.98–1.08 in the alternating rounds: unchanged within their noise.

Rates after the change, GB/s: the median of the five runs of the new
build at 4096² and 8192² (two `--reps 9`, three `--reps 41`) and of the
two at 16384²; plain C is each run's fastest thread count.

| y = A x | GPU, A resident | GPU, A mapped per call | Accelerate | plain C, 4/8/12 threads |
|---|---:|---:|---:|---:|
| DGEMV `'N'` | 223 / 302 / 317 | 61 / 65 / 59 | 20 / 20 / 19 | 53 / 68 / 85 |
| DGEMV `'T'` | 229 / 320 / 354 | 61 / 72 / 60 | 60 / 55 / 51 | 115 / 116 / 114 |
| ZGEMV `'N'` | 274 / 316 / 320 | 64 / 63 / 42 | 39 / 38 / 37 | 77 / 77 / 86 |
| ZGEMV `'T'` | 277 / 336 / 359 | 64 / 66 / 44 | 117 / 118 / 108 | 123 / 123 / 117 |
| ZGEMV `'C'` | 276 / 334 / 358 | 65 / 66 / 43 | 116 / 120 / 108 | 124 / 112 / 119 |

Errors against the compensated reference are those of the earlier runs
(the bits are the same): 3.6×10⁻¹⁶ to 1.04×10⁻¹⁵ relative to |A||x|.

**Sweeps** (the data behind the `AB_MODE=auto` GEMV rows, `kRoutes` in
`src/blas2_abi.c`, which `tests/test_blas2.c` checks against all four
files): staged GPU against Accelerate in one process, 3 warm-ups and the
median of 7 calls each; `speedup` is Accelerate time / GPU time.

| Staged GPU vs Accelerate | ≥ 1.10× in both runs | elsewhere |
|---|---|---|
| DGEMV `'N'` | 1.59–1.69× at 2048², 2.1–3.6× from 3072² to 16384²; rectangles of 2048² elements and up 1.19–3.48× | ≤ 0.77× up to 1536² |
| DGEMV `'T'` | 6144²–16384² (1.10–1.40×) and five rectangles from 16384 × 1024 up (1.10–1.34×) | 0.94–1.00× at 4096²; below 1.10× on rectangles as large as those wins: 65536 × 256 and 256 × 65536 (0.90–1.00×), 131072 × 512 (1.03×), and in one run 1024 × 16384 (1.02×) and 1024 × 262144 (0.96×) |
| ZGEMV `'N'` | 1.39–1.87× from 3072² to 12288²; every rectangle but 256 × 65536, 1.19–1.86× | 2048² 1.17× and 1.02×; 16384² 1.06× and 1.24×; 256 × 65536 1.06× |
| ZGEMV `'T'`, `'C'` | — | 0.04–0.82× at every shape |

The rows: DGEMV `'N'` keeps 2048²–16384² elements (min side 256), and
DGEMV `'T'` stays on Accelerate as before, its wins too close to the line
and mixed with losses. ZGEMV `'N'` now starts at 3072² elements, where it
started at 2048²: 2048² measured 1.02× in the second run (GPU 1.37 ms,
against 1.21 ms in the first; Accelerate 1.41–1.43 ms). That is not the
kernels: in the alternating rounds the new build's staged ZGEMV `'N'` at
2048² was 1.01–1.03× as fast as the old one's. Mapping A for each call
varies in cost, and at 2048² the GPU's margin is small (1.22× and 1.28×
in `../2026-10-19-317bb91-gemv/`). The 1024 × 4096 and 4096 × 1024 calls,
which have as many elements, leave the row with it. The row still stops
at 12288² (16384² 1.06× in the first run).
