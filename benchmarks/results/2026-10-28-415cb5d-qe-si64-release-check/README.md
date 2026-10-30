# Quantum ESPRESSO si64, the 1.5.0 release check, 2026-10-26 to 2026-10-28

The check that the 1.5.0 pull requests (#5–#10) left for after their merges:
inputs A and B at 1 rank and at 8 ranks on a fresh QE develop build, in
`AB_MODE=cpu`, `auto` and `gpu`, with `release/v1.5.0` at 415cb5d. The same QE
objects relinked to 1.4.0 are the control, so the two libraries are compared on
one QE build. Inputs and scripts: [`benchmarks/qe/`](../../qe/).

**Since this check.** #11 (66c99a2, the device-buffer API's per-row and
per-column choice between Gauss-3M and 4M) merged after it. In the BLAS ABI
it moved the 3M/4M rule and the conversion's chunking into shared helpers
without changing them, and the golden bits held (`test_qe_golden`), so it
does not change these results. The calls this check found leaving the staged
path for the host path (about half of the GPU-routed calls) were fixed afterwards:
[`2026-10-29-351d56c-staged-refusals`](../2026-10-29-351d56c-staged-refusals/)
measures the fix on this QE build.

| | |
|---|---|
| Machine | MacBook Pro (Mac14,6), M2 Max, 38-core GPU, 64 GB, macOS 27.0.1 (26A434). Every reported run was on AC power from start to end ([`power.txt`](power.txt)). |
| QE | develop 86c214974 (2026-10-20, prints `v.8.0dev`): a fresh clone with no source change. Every Fortran object is `-O3 -fopenmp` (gfortran 15.2.0); Open MPI 5.0.9, OpenBLAS 0.3.31, ScaLAPACK 2.2.2, `-lfftw3_omp`. `make.inc` equals the 2026-10-18 control's ([QE builds](../../qe/README.md#qe-builds)) apart from paths and the apple-bottom lines. One OpenMP thread per rank in every run. |
| Libraries | **1.5.0** = 415cb5d (`release/v1.5.0` with #5–#10). The library prints `apple-bottom 1.4.0 (415cb5d4d888)`: the version string is bumped at release time, so the hash identifies the build. **1.4.0** = c5a1fa6, `apple-bottom 1.4.0 (c5a1fa6a8166)`. Both were built from clean worktrees and loaded from frozen copies; `lib_loaded` in `results.csv` names each copy's SHA-256 prefix. |
| Binaries | One `pw.x` per library, linked from the same objects. Only `pw.x` differs between the two builds (SHA-256 of every `.o`, `.a`, `.mod` and `.x` in the tree). Each binary reaches its library by `LC_RPATH`, and `zgemm_`/`dgemm_` bind to it (checked with `DYLD_PRINT_BINDINGS`). |
| Layouts | 1 × 1, and 8 × 1 with `-nk 4`, as in the 1.4.0 release check. With 8 ranks, `run_campaign.sh` caps each rank's OpenBLAS and Accelerate at 1 thread. |
| Plan | One block per input × layout × library ([`plans/`](plans/)). For 1.5.0: `cpu,gpu` × 3, mode order rotated, then `auto` × 1. For 1.4.0: `cpu,gpu` × 2. |
| Locking | Each block ran `run_campaign.sh` inside one exclusive machine lock, taken once `load1 < 5` and after waiting for AC power (part 4 also checked the power inside the lock; see the notes). A run's `load1` in `results.csv` is read at its start, so after a block's first run it includes the block's own previous run. |
| Date | 2026-10-26 20:16 to 2026-10-28 08:48 EDT (the A 8 × 1 1.4.0 block on 10-28; see the notes). |

## Results

From [`comparison.md`](comparison.md), [`tables.md`](tables.md),
[`analyze.txt`](analyze.txt), [`compare.txt`](compare.txt) and
[`staged-calls.txt`](staged-calls.txt). Times are medians, with ranges in brackets.

**Accuracy.** In all 44 runs, the following equal those of the block's `cpu` run:
* the total energy;
* the energy of every SCF iteration;
* every printed band energy (2,000 for A, 640 for B).

The SCF iteration counts match too: 7 for A, 9 for B. Both libraries give the
same values. The 1 × 1 and 8 × 1 totals differ by 1e-8 Ry, from the order of the
MPI reductions.

**Run time** (s):

| Layout | Input | 1.5.0 `cpu` | 1.5.0 `auto` | 1.5.0 `gpu` | `gpu` / `cpu` | 1.4.0 `cpu` | 1.4.0 `gpu` | `gpu` / `cpu` | 1.5.0 `gpu` / 1.4.0 `gpu` |
|---|---|---|---|---|---:|---|---|---:|---:|
| 1 × 1 | A | 306.4 [305.1–306.9] | 304.4 | 359.9 [358.8–360.4] | 1.17 | 304.9 [304.7–305.0] | 397.9 [397.6–398.3] | 1.31 | 0.90 |
| 1 × 1 | B | 184.6 [184.5–184.8] | 184.0 | 221.6 [221.5–221.8] | 1.20 | 184.7 [184.4–184.9] | 264.6 [264.4–264.8] | 1.43 | 0.84 |
| 8 × 1, `-nk 4` | A | 104.7 [104.3–105.3] | 103.5 | 124.9 [123.3–126.9] | 1.19 | 107.9 [105.9–109.9] | 114.4 [111.9–117.0] | 1.06 | 1.09 |
| 8 × 1, `-nk 4` | B | 52.6 [52.6–53.8] | 52.9 | 61.8 [61.6–62.8] | 1.17 | 54.0 [53.7–54.2] | 67.6 [66.7–68.5] | 1.25 | 0.91 |

- **`gpu` mode is still slower than Accelerate.** It takes 1.17–1.20× the `cpu` time in
  all four configurations.
- **1.5.0 vs 1.4.0, `gpu` mode** (same QE build):
  - **B 1 × 1:** 1.5.0 is 1.19× faster: 221.6 s against 264.6 s, with `cpu` mode at 184.6 s.
  - **A 1 × 1:** 1.11× faster (359.9 s against 397.9 s).
  - **B 8 × 1:** 1.09× faster (61.8 s against 67.6 s).
  - **A 8 × 1: 9% slower** (124.9 s against 114.4 s). Its 1.4.0 block was taken a day and a
    half after the 1.5.0 block, and its `cpu` median is 3% higher. Within each block,
    `gpu`/`cpu` is 1.19 for 1.5.0 against 1.06 for 1.4.0, so the slowdown holds within
    each block as well. The first 1.4.0 block for A 8 × 1 ran on battery and gave the
    same picture: `gpu` 116.7 s, `cpu` 105.2 s ([supporting runs](#supporting-runs-not-in-the-tables)).
- **`auto` sends nothing to the GPU**, as in 1.4.0: the calibrated routing table did
  not change. Its wall time is within −1.1% to +0.6% of `cpu`.

**Where the GPU time goes** (ZGEMM calls the GPU computed, s, summed over ranks;
the full table, DGEMM included, is in [`comparison.md`](comparison.md)):

| Layout | Input | Library | GPU calls | host stages¹ | `kernel`² | GPU-call time | Accelerate, same calls³ |
|---|---|---|---:|---:|---:|---:|---:|
| 1 × 1 | A | 1.5.0 | 944 | 31.9 | 72.9 | 104.9 | 52.3 |
| 1 × 1 | A | 1.4.0 | 951 | 87.9 | 55.1 | 143.0 | 53.7 |
| 1 × 1 | B | 1.5.0 | 1088 | 24.9 | 31.7 | 56.5 | 25.0 |
| 1 × 1 | B | 1.4.0 | 1090 | 68.1 | 32.0 | 100.0 | 25.7 |
| 8 × 1 | A | 1.5.0 | 1790 | 55.8 | 329.7 | 385.8 | 287.5 |
| 8 × 1 | A | 1.4.0 | 1800 | 141.6 | 187.5 | 329.1 | 281.6 |
| 8 × 1 | B | 1.5.0 | 2177 | 48.0 | 139.6 | 188.4 | 120.4 |
| 8 × 1 | B | 1.4.0 | 2180 | 158.7 | 78.2 | 236.9 | 135.4 |

¹ alloc, pack, upload, download, epilogue and other. ² Command-buffer GPU time. For
1.5.0 it includes the GPU's FP64 ↔ double-float conversion and epilogue, which 1.4.0
did on the host. ³ The `cpu` runs' time for these calls: the routine's total minus
what `gpu` mode left on Accelerate. With fallbacks (below), the second term also holds
the guard's check, so the 1.5.0 estimate is slightly low. For 1.5.0's DGEMMs it turns
negative, and `tables.md` prints those negative values; read them as unavailable.

- **One rank:** the staged path (#9) cut the host stages to about a third: from
  68.1 s to 24.9 s at B. The GPU kernel time, which now includes the conversion, did
  not change at B (31.7 s against 32.0 s) and rose at A (72.9 s against 55.1 s). The
  GPU calls still take 2.0–2.3× as long as Accelerate needs for the same calls
  (1.4.0: 2.7–3.9×).
- **Eight ranks share one GPU.** Their command buffers overlap, so a span can include
  other ranks' work, and the conversion that 1.4.0 ran on the eight CPU cores now
  joins the GPU queue. At A, `kernel` rose from 187.5 s to 329.7 s, more than the host
  stages saved (141.6 → 55.8 s), and the GPU calls took longer than in 1.4.0. At B the
  savings win (236.9 → 188.4 s).
- BLAS takes 16–19% of each rank's time in `cpu` mode at 1 × 1, and 32–36% at 8 × 1.

**GPU share by time** (`gpu` mode; `AB_REPORT`, summed over ranks; median over
repetitions). The share of each routine's time spent in calls routed to the GPU, not
the share of calls:

| Layout | Input | Library | ZGEMM | DGEMM | GPU-routed calls, share of rank time |
|---|---|---|---:|---:|---:|
| 1 × 1 | A | 1.5.0 | 95.8% | 43.2% | 29.4% |
| 1 × 1 | A | 1.4.0 | 97.8% | 99.9% | 36.8% |
| 1 × 1 | B | 1.5.0 | 93.7% | 59.5% | 27.1% |
| 1 × 1 | B | 1.4.0 | 96.9% | 99.8% | 40.1% |
| 8 × 1 | A | 1.5.0 | 96.9% | 72.8% | 38.9% |
| 8 × 1 | A | 1.4.0 | 97.1% | 99.8% | 36.9% |
| 8 × 1 | B | 1.5.0 | 93.4% | 50.3% | 39.6% |
| 8 × 1 | B | 1.4.0 | 94.9% | 99.8% | 46.4% |

**Fallbacks.** 1.5.0's exponent-range guard (#7) sends some GPU-routed calls back to
Accelerate (`fallbacks range=` in `AB_REPORT`) when they hold values beyond
double-float's exponent range:
* the PAW augmentation DGEMMs of shape nh(nh+1)/2 × atoms × 2·ngm: 36 × 64 × 292,250
  for A and 171 × 64 × 287,330 for B at 1 rank, with K split over the ranks at 8.
  All of them fall back at 1 × 1 and at B 8 × 1. At A 8 × 1, 32 of the 64 do: the
  guard decides from the data, and each rank holds a different part.
* 2–10 ZGEMMs per run (shapes in `staged-calls.txt`).

That is 15, 12, 42 and 83 fallbacks per run (A 1 × 1, B 1 × 1, A 8 × 1, B 8 × 1).
1.4.0 ran these calls on the GPU, with the same printed energies; 1.5.0 refuses such
data by design. So the GPU computes 50% of the DGEMMs routed to it (75% at A 8 × 1).
These are the 2·ngm × nh(nh+1)/2 × atoms products, plus, at A 8 × 1, the 32
augmentation DGEMMs that passed the guard. By the signature below, all of them take
the host path.

**Staged-path coverage.** These counts are signature-based, read from the
`AB_PROFILE` per-call lines ([`staged-calls.txt`](staged-calls.txt)). A GPU call
whose line shows `epilogue=0.000` ran no host epilogue: the staged path with the GPU
epilogue (ZGEMM with alpha = 1 and beta = 0 or 1). The other calls ran a host
epilogue: the staged path's variant for other alpha and beta, or the host path. A
host epilogue under 0.5 µs would print 0.000 too.

| Input, layout | Calls without a host epilogue, of the GPU calls | Their share of the GPU calls' time |
|---|---|---:|
| B, 1 × 1 | **597 of 1,098** (598 and 599 in the other two runs) | 26% |
| A, 1 × 1 | 596 of 952 | 30% |
| B, 8 × 1 | 1,254–1,255 of 2,257 | 34–35% |
| A, 8 × 1 | 1,196–1,197 of 1,886 | 39% |

No DGEMM shows a zero epilogue.

**Versus the published 1.4.0 release check**
([2026-10-15](../2026-10-15-c5a1fa6-qe-si64-opt/), QE develop 5b7f8e1bc). Its
`gpu`/`cpu` ratios were 1.30, 1.40, 1.05 and 1.21 (A 1 × 1, B 1 × 1, A 8 × 1,
B 8 × 1). Here they are 1.17, 1.20, 1.19 and 1.17. The QE commit (792 commits
later), the day and the machine's state all differ, so only the same-QE control
above attributes a difference to the library. The control reproduces the published
1.4.0 ratios within 0.03 at 1 × 1 and A 8 × 1, and within 0.04 at B 8 × 1.

**Floating-point flags** ([`fpflags.txt`](fpflags.txt)). 43 runs report
`IEEE_INVALID_FLAG`, `IEEE_DIVIDE_BY_ZERO` and `IEEE_UNDERFLOW_FLAG` at exit, as the
1.4.0 check did for A and B. One `cpu` run (1.5.0, A 1 × 1, cpu_r3) reports no
underflow. In `cpu` mode the GPU is not involved. A likely cause: with one rank,
Accelerate runs its own threads, and gfortran reports only the main thread's flags.

## Power state and what it confounds

- **The 44 reported runs** were on AC power from start to end. Each run's power was
  read inside the lock at its start, and a 20-second power log covers every `pw.x`
  run (`power.txt`).
- **A 8 × 1, 1.4.0: first attempt on battery.** The first 1.4.0 block (2026-10-26
  23:33) passed the AC check, waited three hours for the lock, then ran on battery.
  It is not in the tables. It was rerun on AC on 2026-10-28 with the power also
  checked inside the lock. Its numbers agree with the rerun: `gpu` 116.7 s against
  114.4 s.
- **The A 8 × 1 comparison is the most exposed to conditions.** Its 1.5.0 and 1.4.0
  blocks ran 36 hours apart, and their `cpu` medians differ by 3%. The 9% `gpu`
  difference also shows in the within-block ratios.
- **Part 1 (2026-10-25/26, one lock per run, below): two runs touched battery.**
  B 8 × 1 1.4.0 gpu_r2 ran entirely on battery, and B 8 × 1 1.5.0 cpu_r2's power
  changed at its end. Part 1 recorded the power before the AC wait and the lock
  wait, so its `power` column can say "Battery Power" for runs that ran on AC.
  `power.txt` has the state over each run.

## How the runs were taken, and the supporting runs

The check took four passes because other sessions' jobs held the machine lock for
hours at a time.

1. **Part 1** (2026-10-25/26) took one lock per run. In 22 hours it finished 15
   runs: B 8 × 1 for both libraries, and part of A 8 × 1 for 1.5.0. From then on, by
   the maintainer's decision, a block takes one lock.
2. **Part 2** (2026-10-26/27) ran A 8 × 1, B 1 × 1 and A 1 × 1 for both libraries.
3. **Part 3** (2026-10-27) ran both B 8 × 1 blocks again, so every reported block
   ran in one lock on AC power.
4. **Part 4** (2026-10-28) ran the A 8 × 1 1.4.0 block again on AC power.

One part-1 run has no row anywhere: A 8 × 1, 1.5.0, `cpu`, rep 3. Its driver was
stopped while it waited for the lock, then the run took the lock on its own (100.6 s,
same energy).

### Supporting runs (not in the tables)

[`supporting/`](supporting/) holds these runs' `results.csv`, reports, power and plans.
Their `pw.x` outputs are not included. They agree with the reported runs: every
energy is identical, and the `gpu`/`cpu` medians are 1.14 for 1.5.0 at B 8 × 1, 1.33
for 1.4.0 at B 8 × 1, and 1.15 for 1.5.0 at A 8 × 1.

| Input | Layout | Library | `AB_MODE` | Run | pw.x start (EDT) | Power over the run | Wall (s) |
|---|---|---|---|---|---|---|---:|
| B | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `cpu` | cpu_r1 | 10-25 11:15 | AC Power | 56.1 |
| B | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `gpu` | gpu_r1 | 10-25 11:28 | AC Power | 63.9 |
| B | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `gpu` | gpu_r2 | 10-25 11:48 | AC Power | 61.9 |
| B | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `cpu` | cpu_r2 | 10-25 12:36 | AC Power, then Battery Power | 55.1 |
| B | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `cpu` | cpu_r3 | 10-25 13:30 | AC Power | 58.9 |
| B | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `gpu` | gpu_r3 | 10-25 14:00 | AC Power | 64.5 |
| B | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `auto` | auto_r1 | 10-25 18:33 | AC Power | 54.1 |
| B | 8 × 1, `-nk 4` | 1.4.0 (`c5a1fa6a8166`) | `cpu` | cpu_r1 | 10-25 19:06 | AC Power | 54.5 |
| B | 8 × 1, `-nk 4` | 1.4.0 (`c5a1fa6a8166`) | `gpu` | gpu_r1 | 10-25 21:15 | AC Power | 69.8 |
| B | 8 × 1, `-nk 4` | 1.4.0 (`c5a1fa6a8166`) | `gpu` | gpu_r2 | 10-25 22:49 | Battery Power | 73.6 |
| B | 8 × 1, `-nk 4` | 1.4.0 (`c5a1fa6a8166`) | `cpu` | cpu_r2 | 10-26 01:09 | AC Power | 53.5 |
| A | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `cpu` | cpu_r1 | 10-26 03:41 | AC Power | 98.4 |
| A | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `gpu` | gpu_r1 | 10-26 03:44 | AC Power | 116.7 |
| A | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `gpu` | gpu_r2 | 10-26 04:04 | AC Power | 114.7 |
| A | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `cpu` | cpu_r2 | 10-26 08:46 | AC Power | 102.3 |
| A | 8 × 1, `-nk 4` | 1.4.0 (`c5a1fa6a8166`) | `cpu` | cpu_r1 | 10-26 23:33 | Battery Power | 106.1 |
| A | 8 × 1, `-nk 4` | 1.4.0 (`c5a1fa6a8166`) | `gpu` | gpu_r1 | 10-26 23:34 | Battery Power | 114.8 |
| A | 8 × 1, `-nk 4` | 1.4.0 (`c5a1fa6a8166`) | `gpu` | gpu_r2 | 10-26 23:36 | Battery Power | 118.5 |
| A | 8 × 1, `-nk 4` | 1.4.0 (`c5a1fa6a8166`) | `cpu` | cpu_r2 | 10-26 23:38 | Battery Power | 104.2 |

The last four are part 2's battery block. All the others are part 1.

## Reported runs

| Input | Layout | Library | `AB_MODE` | Run | pw.x start (EDT) | Power over the run | Wall (s) |
|---|---|---|---|---|---|---|---:|
| A | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `cpu` | cpu_r1 | 10-26 20:16 | AC Power | 104.3 |
| A | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `gpu` | gpu_r1 | 10-26 20:17 | AC Power | 124.9 |
| A | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `gpu` | gpu_r2 | 10-26 20:20 | AC Power | 126.9 |
| A | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `cpu` | cpu_r2 | 10-26 20:22 | AC Power | 105.3 |
| A | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `cpu` | cpu_r3 | 10-26 20:23 | AC Power | 104.7 |
| A | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `gpu` | gpu_r3 | 10-26 20:25 | AC Power | 123.3 |
| A | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `auto` | auto_r1 | 10-26 20:27 | AC Power | 103.5 |
| B | 1 × 1 | 1.5.0 (`415cb5d4d888`) | `cpu` | cpu_r1 | 10-27 03:19 | AC Power | 184.5 |
| B | 1 × 1 | 1.5.0 (`415cb5d4d888`) | `gpu` | gpu_r1 | 10-27 03:22 | AC Power | 221.8 |
| B | 1 × 1 | 1.5.0 (`415cb5d4d888`) | `gpu` | gpu_r2 | 10-27 03:26 | AC Power | 221.6 |
| B | 1 × 1 | 1.5.0 (`415cb5d4d888`) | `cpu` | cpu_r2 | 10-27 03:30 | AC Power | 184.6 |
| B | 1 × 1 | 1.5.0 (`415cb5d4d888`) | `cpu` | cpu_r3 | 10-27 03:33 | AC Power | 184.8 |
| B | 1 × 1 | 1.5.0 (`415cb5d4d888`) | `gpu` | gpu_r3 | 10-27 03:36 | AC Power | 221.5 |
| B | 1 × 1 | 1.5.0 (`415cb5d4d888`) | `auto` | auto_r1 | 10-27 03:40 | AC Power | 184.0 |
| B | 1 × 1 | 1.4.0 (`c5a1fa6a8166`) | `cpu` | cpu_r1 | 10-27 07:08 | AC Power | 184.9 |
| B | 1 × 1 | 1.4.0 (`c5a1fa6a8166`) | `gpu` | gpu_r1 | 10-27 07:11 | AC Power | 264.8 |
| B | 1 × 1 | 1.4.0 (`c5a1fa6a8166`) | `gpu` | gpu_r2 | 10-27 07:16 | AC Power | 264.4 |
| B | 1 × 1 | 1.4.0 (`c5a1fa6a8166`) | `cpu` | cpu_r2 | 10-27 07:20 | AC Power | 184.4 |
| A | 1 × 1 | 1.5.0 (`415cb5d4d888`) | `cpu` | cpu_r1 | 10-27 09:01 | AC Power | 306.4 |
| A | 1 × 1 | 1.5.0 (`415cb5d4d888`) | `gpu` | gpu_r1 | 10-27 09:06 | AC Power | 359.9 |
| A | 1 × 1 | 1.5.0 (`415cb5d4d888`) | `gpu` | gpu_r2 | 10-27 09:12 | AC Power | 358.8 |
| A | 1 × 1 | 1.5.0 (`415cb5d4d888`) | `cpu` | cpu_r2 | 10-27 09:18 | AC Power | 305.1 |
| A | 1 × 1 | 1.5.0 (`415cb5d4d888`) | `cpu` | cpu_r3 | 10-27 09:23 | AC Power | 306.9 |
| A | 1 × 1 | 1.5.0 (`415cb5d4d888`) | `gpu` | gpu_r3 | 10-27 09:28 | AC Power | 360.4 |
| A | 1 × 1 | 1.5.0 (`415cb5d4d888`) | `auto` | auto_r1 | 10-27 09:34 | AC Power | 304.4 |
| A | 1 × 1 | 1.4.0 (`c5a1fa6a8166`) | `cpu` | cpu_r1 | 10-27 10:11 | AC Power | 305.0 |
| A | 1 × 1 | 1.4.0 (`c5a1fa6a8166`) | `gpu` | gpu_r1 | 10-27 10:16 | AC Power | 398.3 |
| A | 1 × 1 | 1.4.0 (`c5a1fa6a8166`) | `gpu` | gpu_r2 | 10-27 10:22 | AC Power | 397.6 |
| A | 1 × 1 | 1.4.0 (`c5a1fa6a8166`) | `cpu` | cpu_r2 | 10-27 10:29 | AC Power | 304.7 |
| B | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `cpu` | cpu_r1 | 10-27 14:53 | AC Power | 52.6 |
| B | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `gpu` | gpu_r1 | 10-27 14:53 | AC Power | 61.6 |
| B | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `gpu` | gpu_r2 | 10-27 14:55 | AC Power | 62.8 |
| B | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `cpu` | cpu_r2 | 10-27 14:56 | AC Power | 52.6 |
| B | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `cpu` | cpu_r3 | 10-27 14:56 | AC Power | 53.8 |
| B | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `gpu` | gpu_r3 | 10-27 14:57 | AC Power | 61.8 |
| B | 8 × 1, `-nk 4` | 1.5.0 (`415cb5d4d888`) | `auto` | auto_r1 | 10-27 14:58 | AC Power | 52.9 |
| B | 8 × 1, `-nk 4` | 1.4.0 (`c5a1fa6a8166`) | `cpu` | cpu_r1 | 10-27 18:56 | AC Power | 53.7 |
| B | 8 × 1, `-nk 4` | 1.4.0 (`c5a1fa6a8166`) | `gpu` | gpu_r1 | 10-27 18:57 | AC Power | 68.5 |
| B | 8 × 1, `-nk 4` | 1.4.0 (`c5a1fa6a8166`) | `gpu` | gpu_r2 | 10-27 18:58 | AC Power | 66.7 |
| B | 8 × 1, `-nk 4` | 1.4.0 (`c5a1fa6a8166`) | `cpu` | cpu_r2 | 10-27 18:59 | AC Power | 54.2 |
| A | 8 × 1, `-nk 4` | 1.4.0 (`c5a1fa6a8166`) | `cpu` | cpu_r1 | 10-28 08:40 | AC Power | 109.9 |
| A | 8 × 1, `-nk 4` | 1.4.0 (`c5a1fa6a8166`) | `gpu` | gpu_r1 | 10-28 08:42 | AC Power | 117.0 |
| A | 8 × 1, `-nk 4` | 1.4.0 (`c5a1fa6a8166`) | `gpu` | gpu_r2 | 10-28 08:44 | AC Power | 111.9 |
| A | 8 × 1, `-nk 4` | 1.4.0 (`c5a1fa6a8166`) | `cpu` | cpu_r2 | 10-28 08:46 | AC Power | 105.9 |

## Files

| File | |
|---|---|
| `results.csv` | One row per reported run, as `run_campaign.sh` writes it. Home and frozen-library paths are shortened. The block's lock was taken before `run_campaign.sh` started, so `start`/`end` bracket the run itself. In `supporting/results.csv`, part 1's `start`/`end` also include the waits for AC power and for the lock. |
| `outputs/` | Each run's `pw.x` output. |
| `reports/` | Each run's stderr, keeping apple-bottom's lines (`AB_REPORT`, the `AB_PROFILE` tables, fallback lines) and gfortran's IEEE note. Dyld output and per-call `AB_PROFILE` lines are dropped. With 8 ranks, the table lines of different ranks can interleave; the `AB_REPORT` lines stay whole. |
| `analyze.txt`, `compare.txt`, `tables.md` | Output of `benchmarks/qe/analyze.py`, `compare.py` and `tables.py` (one section per label). |
| `comparison.md` | The tables above in full: run time, GPU share by time, stages per routine and library, energies. |
| `staged-calls.txt` | The signature-based count per `gpu` run, with each run's fallbacks by shape. |
| `power.txt` | Per run: the start and end of `pw.x`, the power at its start, and every power state logged while it ran. |
| `fpflags.txt` | The IEEE flags reported as signalling in each run, as the union over its ranks. |
| `plans/` | The plan of each block. |
| `supporting/` | The runs not in the tables (above), without their `pw.x` outputs. |
