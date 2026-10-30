# Register prefetch in the GPU GEMM kernels, 2026-10-23/24

Commit 15ef7b9 gives every tiled GEMM kernel a register-prefetch instance
(`*_pf`). It loads K-tile k+1 into registers while it computes tile k. A
rule, `use_prefetch` in `src/apple_bottom.m`, picks it. This directory holds:
- the measurements that set the rule;
- the bit checks;
- the timings on Quantum ESPRESSO-like shapes, both as kernel GPU time and
  through the BLAS ABI.

| | |
|---|---|
| Machine | MacBook Pro (Mac14,6), Apple M2 Max, 38-core GPU, 64 GB, macOS 27.0.1 (26A434) |
| Library | 15ef7b9 on release/v1.5.0 e3f09ca. `ab_build_info()`: `1.4.0 (15ef7b91d00f)`, a clean tree. Baseline: e3f09ca built from `git archive`, reporting `1.4.0 (e3f09ca-base)`. |
| Kernel timings | [`pfprobe.m`](pfprobe.m), described below. |
| Power, lock | Every run held an exclusive machine lock, so no other benchmark job ran. See the next paragraph for which runs were on AC and which on battery. |

**Kernel timings.** [`pfprobe.m`](pfprobe.m) extracts the MSL from
`src/apple_bottom.m` and compiles it with `MTLMathModeSafe`, as the library
does.
- **What it times:** each kernel's GPU time, `GPUEndTime − GPUStartTime` of
  one command buffer.
- **Order:** the plain and `_pf` instances run round-robin, the order
  rotating each round, after a 1.5 s warm-up.
- **Products per command buffer:** `1x` is one product. `3x` is three
  products with distinct operands in one concurrent encoder, like Gauss-3M's.
- **Columns:** each line gives every kernel's mean ms. For each pair x/y it
  gives mean(x)/mean(y), then the median of the per-round ratios. Above 1
  means y is faster.
- **Routing figure:** the ratio of means. Small grids can time bimodally
  (how the GPU spreads them over its cores), so medians can mislead.
- **Kernel names:**
  - `sq`/`sqp`: `dd_dgemm`/`_pf`.
  - `ab`/`abp`: `dd_dgemm_ab`/`_pf`.
  - `nw`/`nwp`, `nws`/`nwsp`: the narrow 32×32 kernels, plain and scaled.
  - `ts`/`tsp`: tall-skinny.
  - `sy`/`syp`: DSYRK.
- **Grid columns:** `t64` and `t32` are the threadgroups of the 64×64 and
  32×32 tilings.
- **Build and run lines:** in the file's header.

**Power.**
- **Calibration, on AC (2026-10-23 03:57–04:16):** `kernel_sq_*`,
  `kernel_ts*`, `kernel_scaled`, `kernel_rect`.
- **Final runs, on battery (2026-10-24 00:52–02:05):** bits, tests,
  harness, `kernel_qe`, `kernel_gap*`, `kernel_syrk`, traces, `e2e`.
- **Clock:** the `MHz` column is the GPU's mean active clock for that line.
  The battery runs' lines run at 1382–1398 MHz, the clock of the AC runs,
  with three exceptions:
  - the gap runs, which lower the clock on purpose;
  - two concurrent lines of `kernel_qe`, at 1307 and 1358 MHz;
  - the unchanged 18277 × 150 × 150 control, at 508–563 MHz there, so its
    numbers come from `kernel_ts_big`, on AC.

## Bits

The `_pf` instances store every tile element in the same threadgroup slot as
the plain kernels and share their inner loop, so each output gets the same
`dd_fma` sequence and renormalizations. Checked four ways:

- **[`bits.txt`](bits.txt):** `pfprobe bits`, every `_pf` kernel against its
  plain twin. 17 shapes, from 1 × 1 × 1 to 4000 × 171 × 64, with K = 1, 3,
  16, 17, 64, 129, 513, 1144, 1500 and 18433 and non-power-of-two M and N. Three kinds of data: exponents in
  [−4, 4], in [−20, 20], and the latter with ±0, ±Inf, NaN and 2^±100
  mixed in. Each shape runs alone and as three concurrent products.
  1071 comparisons, 0 differing elements.
- **`make test` and `make ci-local`:** PASS. `ci-local` ran 176 of 176
  tests, `make test`'s 158 among them. The golden bits are 32/32 equal to
  1.4.0.
- **`test_prefetch_same_bits` in `tests/test_correctness.c`:** every `_pf`
  kernel through the public API, with `AB_PREFETCH=1` (every dispatch
  prefetches) against `AB_PREFETCH=0`. It requires bitwise-equal results and
  one prefetch dispatch per product.
- **The 510-case harness of the v1.5 kernel work** (a local harness, not
  published in this repository): every GPU GEMM entry point (ABMatrix,
  batched, batch, async, DSYRK, ZHERK, DTRSM, MPIR, BLAS ABI, device API),
  with the same mix of data, hashing raw DD buffers and FP64 results.
  - Its hashes for this build equal e3f09ca's with `AB_PREFETCH` unset, `0`
    and `1`.
  - The one case it flags in every run, base included, is the guards'
    intended Accelerate fallback for NaN data
    (`zgemm_blas/NC/300x300x100/…/N`).

## The rule

Kernel GPU time, plain ÷ `_pf`, by grid (ratio of means). Long, medium and
short K are 18432, 4608 and 1024, except tall-skinny: 4608, 1024 and 160.

| kernel | grid (threadgroups) | long K | medium K | short K | file |
|---|---|---:|---:|---:|---|
| 64×64, alone | 16–25 | 1.21–1.26 | 1.15–1.18 | 1.14–1.17 | `kernel_sq_1x` |
| | 36–49 | 1.09–1.13 | 1.07–1.08 | 1.04–1.05 | |
| | 60–64 | 1.08–1.11 | 1.06–1.09 | 1.05–1.06 | `kernel_sq_1x`, `kernel_rect` |
| | 70–96 | 1.00–1.02 | 0.99–1.00 | 0.97 | |
| | 100–128 | 1.00–1.08 | 1.01–1.08 | 0.98–1.06 | |
| | 144–289 | 1.00–1.03 | 1.00–1.02 | 0.98–0.99 | |
| 64×64, three concurrent | 3 × 16–49 | 1.00–1.05 | 0.98–1.04 | 0.97–1.02 | `kernel_sq_3x` |
| | 3 × 64 | 0.87–0.91 | 0.87–0.92 | 0.86 | |
| | 3 × 81–289 | 0.98–1.01 | 0.97–1.04 | 0.97–1.01 | |
| narrow, alone | 25 (150², 160²) | 1.43 | | | `kernel_qe` |
| | 64–1156 | 1.01–1.13 | 1.00–1.10 | 0.97–1.10 | `kernel_sq_1x` |
| narrow, three concurrent | 3 × 64–1156 | 1.01–1.08 | 1.01–1.07 | 1.01–1.07 | `kernel_sq_3x` |
| tall-skinny, alone | 24–48 | 1.20–1.47 | 1.17–1.36 | 1.15–1.32 | `kernel_ts` |
| | 72–224 | 1.00–1.16 | 1.01–1.16 | 0.98–1.11 | |
| | 720–1430, K = 150–300 (QE's rotations) | | | 0.99–1.01 | `kernel_ts_big` |
| tall-skinny, three concurrent | 3 × 24–1430 | 0.99–1.09 | | | `kernel_ts`, `kernel_ts_big` |
| DSYRK | N ≤ 512 (≤ 8×8 grid) | 1.08–1.18 | 1.06–1.16 | 1.03–1.11 | `kernel_syrk` |
| | N = 640–1280 | 0.92–1.05 | 0.97–1.06 | 0.92–1.02 | |

What the rule does with these numbers:
- **64×64 kernels and DSYRK:** prefetch up to 64 threadgroups, and the 64×64
  kernels only in a serial encoder. Concurrent products share the GPU, and
  three of 64 lose 9–14%. The narrow tiles take every C of at most 480², so
  a 64×64 grid always has at least 57 threadgroups.
- **Tall-skinny:** up to 512 threadgroups, alone or concurrent.
- **Narrow:** every grid.
- **Why larger grids keep the plain kernels:** the registers cost occupancy.
  `maxTotalThreadsPerThreadgroup` falls 832 → 512 for 64×64, 896 → 576 for
  tall-skinny, 768 → 512 for DSYRK and 1024 → 896 for narrow.
- **Scaled kernels:** `ab`/`abp` track `sq`/`sqp` within 2% (`kernel_scaled`).

## Quantum ESPRESSO-like shapes: kernel GPU time

From [`kernel_qe.txt`](kernel_qe.txt), one product per command buffer for
`dgemm_` and Gauss-3M's three concurrent products for `zgemm_`. Each row
compares the kernel each routine used before with the one it uses now:

| M × N × K | `dgemm_` kernel | before → after ms | × | `zgemm_` products | before → after ms | × |
|---|---|---:|---:|---|---:|---:|
| 160 × 160 × 18432 | narrow | 4.86 → 3.40 | 1.43 | narrow | 8.20 → 7.55 | 1.09 |
| 150 × 150 × 18277 (si64 overlap) | narrow | 4.81 → 3.39 | 1.42 | narrow | 8.54 → 8.07 | 1.06 |
| 256 × 256 × 18432 | narrow | 6.00 → 5.29 | 1.13 | narrow | 16.00 → 15.34 | 1.04 |
| 384 × 384 × 18432 | narrow | 13.03 → 12.47 | 1.04 | narrow | 34.31 → 30.99 | 1.11 |
| 500 × 500 × 18277 | 64×64, 64 | 19.01 → 17.57 | 1.08 | 64×64, concurrent | unchanged | |
| 512 × 512 × 18432 | 64×64, 64 | 19.08 → 17.60 | 1.08 | 64×64, concurrent | unchanged | |
| 512 × 512 × 4608 | 64×64, 64 | 4.71 → 4.42 | 1.07 | 64×64, concurrent | unchanged | |
| 640 × 640 × 9216 | 64×64, 100 | unchanged | | 64×64, concurrent | unchanged | |
| 4096 × 64 × 4608 | tall-skinny, 128 | 5.81 → 4.92 | 1.18 | tall-skinny | 14.29 → 13.63 | 1.05 |
| 18277 × 150 × 150 (rotation) | tall-skinny, 1430 | unchanged | | tall-skinny | unchanged | |

## Through the BLAS ABI

[`trace_dgemm.txt`](trace_dgemm.txt) splits each `dgemm_` call with
`AB_TRACE`: 69 calls per mode, prefetch off (`AB_PREFETCH=0`) and on,
alternating, on battery. The kernel stage is the command buffer's GPU time.

| `dgemm_` | kernel off → on ms | × | wall off → on ms | × |
|---|---:|---:|---:|---:|
| 150 × 150 × 18277 | 5.18 → 3.65 | 1.42 | 15.66 → 14.32 | 1.09 |
| 256 × 256 × 18432 | 9.61 → 9.92 | 0.97 | 58.38 → 58.87 | 0.99 |
| 320 × 320 × 4608 | 2.36 → 2.51 | 0.94 | 9.90 → 9.90 | 1.00 |
| 384 × 384 × 18432 | 13.31 → 12.92 | 1.03 | 77.97 → 77.68 | 1.00 |
| 384 × 384 × 4608 | 3.34 → 3.29 | 1.01 | 12.85 → 12.72 | 1.01 |
| 448 × 448 × 4608 | 4.04 → 3.93 | 1.03 | 14.89 → 14.87 | 1.00 |
| 512 × 512 × 18432 | 19.37 → 18.09 | 1.07 | 127.50 → 125.06 | 1.02 |
| 512 × 512 × 4608 | 6.07 → 6.10 | 1.00 | 24.58 → 24.42 | 1.01 |

- **Host work dominates these calls.** Packing and converting the FP64
  operands take most of the wall time (the `pack` column).
- **The GPU clock stays low.** The GPU idles between kernels, so a kernel
  runs slower here than in the probe: 9.6 ms at 256 × 256 × 18432 against
  6.0 ms.
- **Prefetch at that clock:** it keeps its gain on the smallest grid (1.42×)
  and is within ±6% on the mid-size ones.

[`e2e_summary.txt`](e2e_summary.txt) gives five rounds of
`bench_rect_dgemm` / `bench_rect_zgemm` over
[`shapes/qe_e2e.txt`](shapes/qe_e2e.txt) (5 calls per shape, the
end-to-end wall time per call):
- **Stages:** each round runs three, the order rotating: the e3f09ca build,
  this branch with `AB_PREFETCH=0`, and this branch as routed.
- **Prefetch off ÷ on**, as the ratio of the median wall times per call,
  with the median [min–max] of the paired per-round ratios, which
  `e2e_summary.txt` also prints, after it:
  - `dgemm_` 160 × 160 × 18432: 1.03 (34.84 → 33.77 ms); paired 1.10
    [1.03–1.13].
  - `dgemm_` 150 × 150 × 18277: 1.11 (15.56 → 14.05 ms); paired 1.10
    [0.85–1.15].
  - `dgemm_` 512 × 512 × 18432: 1.04; paired 1.04.
  - `zgemm_` 160 × 160 × 18432: 1.03; paired 1.04 [1.01–1.06].
  - `zgemm_` 150 × 150 × 18277: 1.03; paired 1.02.
  - Every other shape: 0.98–1.03 (paired 0.99–1.04), within the rounds'
    spread. That includes the shapes whose kernels did not change.
  - The two measures disagree most at 160 × 160 × 18432, where the times
    spread widely from round to round; quote the ratio of medians.
- **Baseline ÷ prefetch off** (paired): 0.96–1.04. Both run the plain
  kernels, so this spread is the noise of these battery runs.

## When the GPU clock is low

`kernel_gap{0,5,20}.txt` idle the GPU 0, 5 or 20 ms before each timed run,
like a caller that spends that long on the host between calls. The GPU
clock falls accordingly: 1395–1398 MHz at 0 ms, 704–1378 MHz at 5 ms,
634–1364 MHz at 20 ms.

| plain ÷ `_pf` (ratio of means) | gap 0 | gap 5 ms | gap 20 ms |
|---|---:|---:|---:|
| 64×64 512 × 512 × 18432 | 1.09 | 1.09 | 1.08 |
| 64×64 512 × 512 × 4608 | 1.07 | 1.05–1.08 | 1.02–1.03 (median 0.96) |
| narrow 160 × 160 × 4608 | 1.30–1.32 | 1.23–1.34 | 1.17–1.26 |
| narrow 384 × 384 × 4608 | 1.02–1.03 | 1.04–1.05 | 0.89–0.94 |
| tall-skinny 2000 × 40 × 4608 | 1.21 | 1.27 | 1.22 |

- **Long kernels and the smallest grids:** keep the gain at every gap.
- **Short mid-size kernels:** after a long idle spell they are within about
  ±10%.
- **Why the rule ignores this:** it follows the full-clock measurements,
  because it cannot know the clock. `AB_PREFETCH=0` turns it off for a
  workload where it does not pay.

## Files

| file | |
|---|---|
| `pfprobe.m` | The probe. Build and run lines are in its header; `-DPFPROBE_CLOCK` with an `ior.h` adds the MHz column. `ior.h` is an IOReport wrapper that is not in this repository; without it the column reads `nan`. |
| `shapes/*.txt` | The probe's shape lists (`qe_e2e.txt` is the rect benchmarks' config). |
| `kernel_sq_1x.txt`, `kernel_sq_3x.txt` | 64×64 and narrow, squares 256–1088 step 32, K = 18432 / 4608 / 1024 (halved where a command buffer would exceed 32 GF). |
| `kernel_rect.txt` | 64×64 and narrow, non-square grids of 60–128 threadgroups. |
| `kernel_ts.txt`, `kernel_ts_big.txt` | Tall-skinny, 24–1430 threadgroups, K = 150–4608. |
| `kernel_scaled.txt` | The scaled kernels (alpha, beta epilogue). |
| `kernel_syrk.txt` | DSYRK, N = 128–1280. |
| `kernel_qe.txt` | The QE-like shapes, every kernel, 1x and 3x. |
| `kernel_gap*.txt` | The idle-gap runs. |
| `bits.txt` | `pfprobe bits`. |
| `trace_dgemm.txt`, `trace_summary.py` | The `AB_TRACE` split and the script that made it. |
| `e2e/`, `e2e_summary.txt`, `e2e_summary.py` | The rect benchmarks' CSVs per stage and round, their summary and the script. |

## Not covered

- **Other GPUs:** only this 38-core M2 Max was measured. The limits (64 and
  512 threadgroups) probably scale with the core count and register file.
- **The narrow/64×64 boundary** (`use_narrow_tiles`, C ≤ 480²) is not part
  of this change. In `kernel_sq_1x.txt`:
  - The narrow kernels with prefetch beat the plain 64×64 kernel by
    1.04–1.42× at most squares from 544² to 1024².
  - They lose by 2–12% at 832², 896², 1056² and 1088².
  - At 512² they are 10% slower than 64×64 with prefetch.
  - A better boundary needs a model of padding and per-core load.
- **Golden-file cases on the 64×64 and tall-skinny prefetch paths:** the
  golden file must come from the 1.4.0 library, which this machine no
  longer has. `test_prefetch_same_bits` compares the paths with each other
  instead.
