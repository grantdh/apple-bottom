# Narrow (32×32) GEMM tiles: which grids they win, 2026-10-24/25

Commit 2825d2a changes `use_narrow_tiles` in `src/apple_bottom.m`. That rule
picks the 32×32 kernels (`dd_dgemm_narrow`, `dd_dgemm_ab_narrow`) over the
64×64 and tall-skinny 128×16 ones.
- **Why now:** the old boundary was calibrated before every kernel had a
  register-prefetch instance (15ef7b9).
- **Bits:** every tiling computes each output with the same `dd_fma`
  sequence and renormalizations, so the change is speed only (see Bits).

This directory holds:
- the grid sweeps behind the rule;
- the new-shape validation, alone, concurrent and at a low GPU clock;
- the Quantum ESPRESSO-like shapes before and after;
- the bit checks.

| | |
|---|---|
| Machine | MacBook Pro (Mac14,6), Apple M2 Max, 38-core GPU, 64 GB, macOS 27.0.1 (26A434) |
| Library | 2825d2a on `perf/gemm-prefetch` dce55b5 (PR #8). Base: dce55b5 built from `git archive`, reporting `1.4.0 (dce55b5-base)`. |
| Probe | [`nrprobe.m`](nrprobe.m). It is [`pfprobe.m`](../2026-10-24-15ef7b9-prefetch/pfprobe.m) with two additions: a one-row grid sweep, and up to four products in one concurrent encoder. It compiles the library's MSL with `MTLMathModeSafe`, as the library does. It times each kernel's GPU time with rotating rounds after a 1.5 s warm-up. The `MHz` column is the mean active GPU clock. |
| Lock, power | Every run held machlock's exclusive bench lock. Sweeps: 2026-10-24 22:43–22:47 on AC. Validation: 2026-10-25 06:19–06:23 on AC. Tests, abbits and QE shapes: 10:29–10:34 on battery; QE lines at 1255–1398 MHz except three 18277-row rotation lines at 518–647 MHz (their route is unchanged). Final tests and abbits on 2825d2a: 13:32–13:33 on battery. |

## The rule

| | dce55b5 | 2825d2a |
|---|---|---|
| narrow vs 64×64, alone (serial encoder) | C ≤ 480² | narrow grid ≤ 1024 threadgroups; ≤ 239 where 64×64 prefetches (≤ 64 threadgroups) |
| narrow vs 64×64, concurrent encoder (Gauss-3M, 4M) | C ≤ 480² | narrow grid ≤ 441 |
| narrow vs tall-skinny (M ≥ 4N, scaled dispatch) | C ≤ 480² and (32 columns pad N no more than 16, or ≤ 16 tall-skinny threadgroups) | narrow grid within the limits above, and ⌈Gn/33⌉ ≤ 2·⌈Gts/33⌉ |

Gn, Gts and G64 are the threadgroup counts of the narrow, tall-skinny and
64×64 grids. The rule takes the encoder to see whether it is concurrent; the
encode helpers keep their signatures. `AB_PREFETCH` changes which instance
runs, never the tiling.

## The cost model: GPU steps

**Sweep.** `nrprobe sweep` runs each kernel on one-row grids:
- shape: M = tile rows, N = G × tile columns, K = 1024;
- G = 1–400 (64×64), 1–1300 (narrow), 1–800 (tall-skinny), every value;
- alone (`sweep_*_1x.txt`), as three concurrent products
  (`sweep_*_3x.txt`, `*_3x_big.txt`) and as four (`sweep_4x.txt`).
- `narrow_rule.py steps` lists where each sweep steps up.

**Alone on the GPU, every kernel steps up at the same G:** 34, 67, 100,
141, 174–182, 207–215, 240, 305, 338, 371, 404, 437, 470, 503.
- The steps are about 33 threadgroups apart, with one 65 wide at 240–304.
- Within a step the time is flat. The boundaries are the same whatever the
  kernel or its residency (`maxTotalThreadsPerThreadgroup`/256 is 2 or 3).
- This explains the earlier calibration's plateaus: 64×64 grids of 144 and
  169 threadgroups (768², 832²) take the same time, and of 256 and 289
  (1024², 1088²) nearly so.

Time per step at K = 1024, ms (G inside each step):

| step | G | 64×64 | 64×64 prefetch | narrow prefetch | tall-skinny prefetch |
|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 0.696 | 0.588 | 0.194 | 0.365 |
| 2 | 50 | 1.047 | 0.968 | 0.298 | 0.553 |
| 3 | 80 | 1.446 | 1.450 | 0.414 | 0.845 |
| 4 | 120 | 1.958 | 1.877 | 0.563 | 1.070 |
| 6 | 190 | 2.800 | 2.778 | 0.807 | 1.582 |
| 8 | 280 | 3.698 | 3.686 | 1.068 | 2.096 |
| 11 | 390 | 5.032 | 5.029 | 1.453 | 2.858 |

**Per step, beyond the first, a narrow step costs:**
- 0.127 ms against 0.44 for 64×64, so 0.29×;
- half a tall-skinny step (0.255).
- Per output, narrow is ~16% dearer than 64×64 (1.17× at full occupancy).
- So the narrow tiles win while a grid takes few steps. Two
  consequences:
  - Against tall-skinny the break-even falls at twice the steps: there
    narrow is at most 2% slower, and 23% faster when tall-skinny takes one
    step.
  - Against 64×64 prefetch (two steps up to 64 threadgroups) the narrow
    grid may take seven steps: up to 239 threadgroups.

**Concurrent products step differently.**
- **Three concurrent products** (G per product): steps at 10, 14, 31, 40, 55,
  70, 88, 103, 120, 134, 149, 158.
  - At large G they pack about 45 threadgroups per step against 33 alone.
  - Small grids are erratic.
- **Four products:** steps at 10, 12, 14, 21, 23, 33, 41, 48, 54.
- One product seems to spread over about 33 of the 38 cores per step;
  concurrent ones reach the rest.

**Fit.** `narrow_rule.py predict` scales the sweeps' times by K/1024 to
predict each 2-D measurement (log ratio measured/predicted):

| data | K ≥ 1024: mean, sd | K = 150–160 |
|---|---|---|
| 2026-10-24 prefetch calibration (squares, rectangles, tall-skinny; 1x, 3x) | −1.6% to +2.9%, sd 0.7–3.3% | +5% to +9% (fixed cost) |
| this validation | −3.5% to +1.0%, sd 2.0–3.7% | +2% to +15% |

The lookup alone picks within 0.3% (mean) of the faster kernel on the 252
prefetch-era lines. The rule replaces it with a few comparisons:
- **Squares and rectangles:** thresholds. A step-ratio rule misjudges large
  64×64 grids, whose real steps drift from multiples of 33.
- **Tall-skinny:** the step ratio, which no threshold on Gts matches.
- **Limits:** 1024 and 441 are where narrow stops winning, alone and
  concurrent.
- **Robustness:** on the measured lines, a step size of 32–34 gives the same
  routes. 30 or 36 would misroute a few tall-skinny shapes by up to 29%.

## Validation on new shapes

`narrow_rule.py score` takes the kernels each rule routes to and gives old
time ÷ new time (ratio of means, 15 rotated rounds). Regret is the extra
time over the faster of the two kernels.

| file | lines | rerouted | old ÷ new, geomean (min–max) | regret old: mean / max | regret new: mean / max |
|---|---:|---:|---|---|---|
| [`kernel_sq_rect.txt`](kernel_sq_rect.txt): squares 560²–2048², rectangles, K = 1024–18432, 1x and 3x | 80 | 31 | 1.033 (0.992–1.462) | 3.6% / 46% | 0.2% / 5.2% |
| [`kernel_ts.txt`](kernel_ts.txt): tall-skinny, N = 8–300, 1x and 3x | 74 | 38 | 1.050 (0.917–1.336) | 6.1% / 34% | 0.8% / 14% |
| [`kernel_np4.txt`](kernel_np4.txt): four concurrent products | 7 | 4 | 0.996 (0.919–1.077) | 1.2% / 7.7% | 1.7% / 8.8% |
| [`kernel_gap20.txt`](kernel_gap20.txt): 20 ms idle before each run (GPU at 700–1370 MHz) | 10 | 7 | 1.058 (1.000–1.282) | 6.1% / 28% | 0.0% / 0.1% |
| [`kernel_qe.txt`](kernel_qe.txt): QE-like shapes, below | 39 | 15 | 1.032 (0.978–1.411) | 4.2% / 41% | 0.7% / 14% |
| prefetch-era `kernel_sq_1x.txt` | 81 | 48 | 1.068 (0.925–1.424) | 7.7% / 42% | 0.3% / 8.1% |
| prefetch-era `kernel_sq_3x.txt` | 81 | 18 | 1.013 (0.987–1.179) | 1.6% / 18% | 0.2% / 3.9% |
| prefetch-era `kernel_rect.txt` | 30 | 24 | 1.158 (0.987–1.460) | 17% / 46% | 0.1% / 1.4% |
| prefetch-era `kernel_ts.txt` | 60 | 36 | 1.015 (0.884–1.259) | 2.5% / 26% | 0.9% / 13% |

All 482 lines: 1.043×. The 228 rerouted lines run 1.093× faster: 194
faster, 34 slower.

**Where the new rule loses:**
- **Concurrent tall-skinny products with tiny grids,** where the
  three-product staircase is erratic: 2000×40×160 0.88×, 1500×40×4608 0.92×,
  1024×48×4608 as four products 0.92×, 2000×100 0.95–0.98×.
- **64×64 grids that fill their last step:** 832×832 0.92–0.96× and
  896×896 0.96–0.98×.
- **The rest:** within 3%.

**Where it gains most:**
- **Squares and rectangles whose 64×64 grid has just passed a step:**
  - 600×600 1.46×, 608×608 1.41–1.46×;
  - 448×640 and 384×768 1.34–1.37×;
  - 544×544 1.32–1.36×.
- **Small tall-skinny grids whose N the 32-column tiles pad more:**
  - 1000×80 1.34×, 1500×40 1.29× alone;
  - 1024×40 and 1024×48 1.25×;
  - 1200×100 1.23× alone and 1.20× concurrent.

**Low clock.** With `--gap 20` the GPU runs at 700–1370 MHz, like a
host-bound BLAS-ABI caller. The new rule picks the faster kernel on all 10
shapes, within 0.1%.

## Quantum ESPRESSO-like shapes: kernel GPU time

From [`kernel_qe.txt`](kernel_qe.txt). `dgemm_` runs one product; `zgemm_`
runs Gauss-3M's three concurrent products. Lines whose route did not change
are left out.

| M × N × K | products | old → new kernel | old → new ms | × |
|---|---|---|---:|---:|
| 600 × 600 × 9140 | 1 | 64×64 → narrow | 17.00 → 12.05 | 1.41 |
| 600 × 600 × 9140 | 3 | 64×64 → narrow | 38.08 → 35.64 | 1.07 |
| 640 × 640 × 9216 | 1 | 64×64 → narrow | 17.95 → 13.58 | 1.32 |
| 640 × 640 × 9216 | 3 | 64×64 → narrow | 39.69 → 38.61 | 1.03 |
| 800 × 800 × 9140 | 1 | 64×64 → narrow | 21.55 → 20.31 | 1.06 |
| 1000 × 1000 × 4570 | 1 | 64×64 → narrow | 16.93 → 16.89 | 1.00 |
| 500 × 500 × 18277 (si64) | 3 | 64×64 → narrow | 52.77 → 50.96 | 1.04 |
| 512 × 512 × 18432 | 3 | 64×64 → narrow | 54.90 → 52.39 | 1.05 |
| 512 × 512 × 4608 | 3 | 64×64 → narrow | 12.83 → 12.50 | 1.03 |
| 4096 × 64 × 4608 | 1 | tall-skinny → narrow | 4.88 → 4.82 | 1.01 |
| 4096 × 64 × 4608 | 3 | tall-skinny → narrow | 13.30 → 12.44 | 1.07 |
| 1024 × 40 × 4608 | 1 | tall-skinny → narrow | 1.66 → 1.32 | 1.26 |
| 1024 × 40 × 4608 | 3 | tall-skinny → narrow | 3.91 → 3.63 | 1.08 |
| 2000 × 100 × 4608 | 1 | tall-skinny → narrow | 4.82 → 4.85 | 0.99 |
| 2000 × 100 × 4608 | 3 | tall-skinny → narrow | 11.98 → 12.24 | 0.98 |

Unchanged routes:
- 150²–384² × 18k overlaps stay narrow.
- 500² and 512² alone keep 64×64 with prefetch.
- The 18277 × 150 × 150 and 18277 × 300 × 300 rotations stay tall-skinny.
- Davidson's 200 × 40 × 18277 and PAW's 36 × 64 × 73062 stay narrow.

The probe skipped 800 × 800 × 9140 as three products (over 32 GF per command
buffer). These are kernel times. Through the BLAS ABI, packing and conversion
take most of a call, and the GPU idles between calls. The prefetch notes show
how such gains shrink end to end; the gap-20 run shows the rule's choices
still hold at that clock.

## Bits

- **Kernels:** unchanged by this commit, so the prefetch notes'
  `pfprobe bits` still applies. Over 1071 comparisons, the narrow,
  tall-skinny and 64×64 kernels and their prefetch instances agree bitwise,
  alone and as three concurrent products.
- **Library, the v1.5 harness (`abbits.m`, 510 cases over every GPU GEMM
  entry point, `AB_MODE=gpu AB_MIN_GPU_DIM=0`):**
  - Hashes are identical between the dce55b5 base and this change.
  - Each build gave the same hashes on two runs.
  - Its cases include shapes whose route changed: 1000×777×150 and
    1024×1024×64 move to narrow, and 4096×64×64 through `ab_dgemm_scaled`
    moves from tall-skinny to narrow.
  - The one case the harness flags in every run, base included, is the
    guards' intended Accelerate fallback for NaN data
    (`zgemm_blas/NC/300x300x100/ld+0/a(1,0)b(1,0)/N`).
  - Hash list: [`abbits_2825d2a.txt`](abbits_2825d2a.txt).
- **`make clean && make test` and `make ci-local`:** green twice.
  - First on the working tree before the commit (10:29, battery).
  - Then at 13:32 (battery) on 2825d2a plus the comment and doc edits
    committed with this directory, the only source they change (build
    identity `2825d2a-dirty`), together with the abbits runs above.
  - Each time: golden bits 32/32 equal to 1.4.0, `test_blas_abi`
    1069/1069, `test_correctness` 79/79, `test_precision` 16/16, and
    `make ci-local` 176/176.
- **Test changes,** so that each kernel stays covered:
  - The prefetch-bits cases use 1100×1000 for the 64×64 kernels (far from
    the limits) and 4000×40 for tall-skinny.
  - The K-P0-1 grids gain 8192×256, tall-skinny through `ab_dgemm_scaled`.
    The native ABI suite's 2048×128 becomes 8192×256.
  - `test_precision`'s `dd_dgemm_ab` case becomes 1100×1000 and its
    tall-skinny ZGEMM 4000×40.

## Not covered

- **Other GPUs.**
  - Only this 38-core M2 Max was measured.
  - The step size (33) and the limits will differ with the core count. The
    code comment says to re-calibrate them with the prefetch limits.
  - Why one dispatch fills about 33 cores per step rather than 38 is not
    known.
- **Concurrent small tall-skinny grids.** The staircase there is
  irregular: steps at 10 and 14 threadgroups per product. The rule takes the
  alone-on-the-GPU step and loses up to 12% on a few such shapes.
- **End-to-end BLAS-ABI timing** was not re-measured for this change.
- **Per-kernel coverage** in the tests is still worked out by hand from the
  rule; no test hook reports which kernel a dispatch used.

## Files

| file | |
|---|---|
| `nrprobe.m` | The probe (build and run lines in its header; `-DPFPROBE_CLOCK` with an `ior.h` for the MHz column). |
| `narrow_rule.py` | Both rules, the scoring, the step listing and the sweep-based prediction. |
| `sweep_*.txt` | One-row grid sweeps, K = 1024: `64`/`nw`/`ts` families, `1x`, `3x` (and `_big` beyond the first range), `4x`. |
| `kernel_sq_rect.txt`, `kernel_ts.txt`, `kernel_np4.txt`, `kernel_gap20.txt`, `kernel_qe.txt` | Validation runs; shape lists in `shapes/`. |
| `abbits_2825d2a.txt` | The 510-case harness hashes of 2825d2a, equal to dce55b5's. |

Reproduce the tables: `python3 narrow_rule.py score kernel_*.txt ../2026-10-24-15ef7b9-prefetch/kernel_{sq_1x,sq_3x,rect,ts}.txt`; `python3 narrow_rule.py steps`; `python3 narrow_rule.py predict <files>`.
