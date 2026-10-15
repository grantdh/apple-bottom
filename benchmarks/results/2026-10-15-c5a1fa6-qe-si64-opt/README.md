# Quantum ESPRESSO si64 on an optimized QE build, 2026-10-15

The 1.4.0 release check in a real code. It uses the same three si64 inputs as
[VAL-1](../../../tests/validation/VAL001_QE_Si64.md), but QE is compiled
`-O3`. VAL-1's QE build was `-O0`, which made the non-GEMM part of these runs
1.8–11× slower and the GEMMs a smaller share of the run. Each input ran in
`AB_MODE=cpu`, `auto` and `gpu`, at one rank and at the fastest CPU layout.
Inputs and scripts: [`benchmarks/qe/`](../../qe/).

| | |
|---|---|
| Machine | MacBook Pro (Mac14,6), M2 Max, 38-core GPU, 64 GB, macOS 27.0.1 (26A434), AC power, `caffeinate -is` |
| QE | develop 5b7f8e1bc (reports 7.5) with the cegterg/regterg patches. Every Fortran object is `-O3 -fopenmp` (gfortran 15.2.0), with Open MPI 5.0.9, Homebrew OpenBLAS, ScaLAPACK and `-lfftw3_omp`, and apple-bottom linked as in [INTEGRATION](../../../docs/INTEGRATION.md). One OpenMP thread per rank in every run; see the notes. |
| Library | `c5a1fa6` (1.4.0; the library code is that of `9c9e4a8`), built from a clean worktree. It reports `apple-bottom 1.4.0 (c5a1fa6a8166)`. Every run loaded it from a frozen copy next to a copy of pw.x (`<frozen>` in `results.csv` and `reports/`). |
| Layouts | `1.4.0-np1`: 1 rank × 1 thread. `1.4.0-np8nk4`: 8 ranks × 1 thread, `-nk 4` (k-point pools), for A and B. `1.4.0-np8`: 8 ranks × 1 thread for C (Γ only, R&G). The multi-rank layouts are the fastest CPU layouts found in a separate study of an optimized build of this QE. |
| Plan | [`plan.txt`](plan.txt); mode order rotated per repetition. The last line is a rerun (see the notes). |
| Date | 2026-10-14 22:55 to 2026-10-15 00:49 local time (EDT). Each run held an exclusive machine lock, taken once `load1 < 5`, so no other job that takes the lock overlapped a run. Between the main block (to 00:20) and the rerun (from 00:38) the lock served eight runs of a separate GPU experiment. |

## Results

From [`tables.md`](tables.md), [`analyze.txt`](analyze.txt) and
[`compare.txt`](compare.txt).

**Accuracy.** In every layout and mode, the total energy, the energy of every
SCF iteration and every printed band energy are identical to the `cpu` run
(2,000 band energies for A, 640 for B, 160 for C). So are the SCF and Davidson
iteration counts. In `gpu` mode the GPU took the calls that make up 88–99% of
each routine's time in `cpu` mode, as VAL-1 counts it (93.9–99.9% of its time
in `gpu` mode), with no fallbacks.

**Routing.** `AB_MODE=auto` sent no call to the GPU in any run, as the
calibrated table says for these shapes. Its median wall time is within 1.7% of
the `cpu` median in every configuration (−1.6% to +1.4%).

**Floating-point flags.** At exit gfortran reports which IEEE flags are
signalling ([`fpflags.txt`](fpflags.txt)). In every run that set is the same
in all three modes (the union over ranks): `IEEE_UNDERFLOW_FLAG` for C, and
`IEEE_UNDERFLOW_FLAG`, `IEEE_INVALID_FLAG` and `IEEE_DIVIDE_BY_ZERO` for A and
B. The `cpu` runs raise them too, so they come from QE or the CPU libraries it
calls, not from the GPU path. In VAL-1, every run that reached the GPU added
`IEEE_OVERFLOW_FLAG`, and none without GPU calls had it. `ab_init()` no longer
leaks it.

**Run time** (s, median [min–max]):

| Layout | Input | `cpu` | `auto` | `gpu` | `gpu` / `cpu` |
|---|---|---|---|---|---:|
| 1 × 1 | A | 306.4 [305.5–307.2] | 302.6 | 396.9 [395.7–398.1] | 1.30 |
| 1 × 1 | B | 188.3 [186.0–190.5] | 187.2 | 264.0 [262.9–265.2] | 1.40 |
| 1 × 1 | C | 68.1 [67.5–68.7] | 68.7 | 77.7 [77.7–77.7] | 1.14 |
| 8 × 1, `-nk 4` | A | 97.1 [96.4–97.9] | 96.7 [95.8–105.6] | 102.4 [102.3–102.4] | 1.05 |
| 8 × 1, `-nk 4` | B | 53.6 [53.4–53.8] | 54.4 | 65.1 [64.8–65.4] | 1.21 |
| 8 × 1 | C | 13.9 [13.7–14.2] | 13.7 | 16.2 [16.2–16.2] | 1.16 |

VAL-1 (`-O0`, 1 × 8) took 506.5–510.1 s for A, 1,120.1–1,121.7 s for B and
748.6 s for C in `cpu` mode.

**GPU kernel time ÷ Accelerate's time for the same calls.** The roadmap uses
this to decide whether BLAS-ABI performance work is worthwhile; the bar is
below 0.8.

| Layout | A, ZGEMM | B, ZGEMM | C, DGEMM | Other GPU stages ÷ kernel (A / B / C) |
|---|---:|---:|---:|---:|
| 1 × 1 | 1.00 | 1.13 | 1.72 | 1.6 / 2.1 / 2.0 |
| 8 × 1 | 0.78 | 0.65 | 0.69 | 0.71 / 1.9 / 2.1 |

With one rank, the GPU kernel is no faster than Accelerate, as in VAL-1. With
8 ranks, each rank's Accelerate has a single thread and the 8 ranks share the
matrix hardware, so the same calls take Accelerate longer. The kernel alone is
then below the 0.8 bar for A's and B's ZGEMMs and C's DGEMMs. It is not below
the bar for the few PAW DGEMMs of A and B (4.25× and 1.49×). The stages around
the kernel lose that margin: packing and converting operands, uploads, the
epilogue and per-call allocation. So `gpu` mode is 5–21% slower end to end.
Kernel time is the GPU time of each command buffer, summed over ranks. With 8
processes on one GPU the command buffers overlap (A's ZGEMMs: 183 s of kernel
time in a 102 s run), so a span can include other ranks' work. The 8 × 1
ratios therefore overstate the kernel's own cost, which makes "below 0.8" a
conservative reading.

## Files

| File | |
|---|---|
| `results.csv` | One row per run: label, input, ranks, threads, mode, repetition, start/end (Unix time; these include the wait for the machine lock), exit status, power, 1-minute load before the lock wait, energy, iterations, wall/CPU time, QE timers, the library it loaded (`<frozen>` is the frozen copy under Library), its run directory and the plan's arguments. |
| `reports/` | Each run's stderr, keeping the apple-bottom lines: `AB_REPORT`, the `AB_PROFILE` tables without the per-call lines, and gfortran's IEEE note. With 8 ranks the tables of different ranks can interleave; the `AB_REPORT` lines stay whole. |
| `analyze.txt`, `compare.txt`, `tables.md` | Output of `benchmarks/qe/analyze.py`, `compare.py` and `tables.py` (one section per layout). |
| `fpflags.txt` | The IEEE flags reported as signalling in each run, as the union over its ranks. |

## Notes

- **One thread per rank.** At OpenMP > 1, this build's FFTs run slower than
  `-O0`. QE asks FFTW for `omp_get_max_threads()` threads per plan and then
  runs one small 1D plan per stick, so the threaded FFTW (`fftw3_omp`, built
  on LLVM's libomp, while QE uses libgomp) spends its time in fork and join.
  With one thread per rank the FFTs are serial, as in an unthreaded build.
- **Thread caps.** With more than one rank, `run_campaign.sh` sets
  `OPENBLAS_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` to each rank's OpenMP
  threads (1 here). A single rank keeps the libraries' defaults, so in the 1 ×
  1 runs Accelerate may use several threads, as it did in VAL-1.
- **The rerun.** The first `auto` run of A at 8 × 1 took 105.6 s, against
  95.8–97.9 s for the other `cpu` and `auto` runs. Against the median of the
  four `cpu` runs, its extra time is in QE's FFT and other code outside BLAS
  (`fftw` +16%, `vloc_psi` +14%, `sum_band` +17%, `PAW_pot` +42%). The
  GEMM-heavy `calbec` was 12% faster, and the run sent no call to the GPU.
  Shortly after it, other processes were using about 1.6 cores. The plan's
  last line reran two `auto` and two `cpu` runs of that configuration; all are
  kept and the medians include them.
- **analyze.py fix.** With several ranks, another rank's unterminated output
  can precede an `AB_REPORT` line in the merged stderr. `analyze.py` matched
  only at the start of a line and missed 7 per-routine lines in 3 of the
  8-rank `gpu` runs (up to 4 of a run's 32). It now finds them anywhere in a
  line (commit `1688803`), so every rank is counted. That moved the 8 ×
  1 A ZGEMM ratio from 0.67 to 0.78. Single-rank results, such as VAL-1's, are
  not affected.
- macOS background services were not stopped.
