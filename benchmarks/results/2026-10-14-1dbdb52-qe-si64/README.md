# Quantum ESPRESSO si64 validation, 2026-10-14

The data behind [VAL-1](../../../tests/validation/VAL001_QE_Si64.md),
revision 2. Inputs and scripts: [`benchmarks/qe/`](../../qe/).

| | |
|---|---|
| Machine | MacBook Pro (Mac14,6), M2 Max, 38-core GPU, 64 GB, macOS 27.0.1 (26A434), AC power, `caffeinate -is` |
| QE | develop 5b7f8e1bc (reports 7.5), gfortran 15.2.0, Open MPI 5.0.9; QE's own Fortran compiled `-O0 -g -fbounds-check` except `cegterg`/`regterg` (`-O3 -fopenmp`); `np = 1`, `OMP_NUM_THREADS=8` |
| Libraries | `1dbdb52` (the 1.4.0 candidate; reports `1.3.1-dev (1dbdb52abeec)`, the version string before the 1.4.0 bump) and `eaad87e` (main before 1.4.0), each built from a clean worktree |
| Plan | [`plan.txt`](plan.txt): `<label> <input> <np> <omp> <repetitions> <modes>`; mode order rotated per repetition |
| Date | 01:14–06:06 local time; the baseline, B and C lines ran from 02:36 (the first launch stopped after line 1) |

## Files

| File | |
|---|---|
| `results.csv` | one row per run: label (`libtag`), input, np, OpenMP threads, mode, repetition, start/end (Unix time), exit status, power source, 1-minute load at start, final energy, SCF iterations, wall and CPU time, QE timers, and the libapplebottom.dylib the run loaded (`<campaign>` = the campaign directory) |
| `reports/` | each run's stderr without dyld lines and without the per-call `AB_PROFILE` lines: the library path it loaded, `AB_REPORT` and the `AB_PROFILE` tables (`eaad87e` predates `AB_REPORT` and the time columns) |
| `analyze.txt` | `benchmarks/qe/analyze.py` output: per run, per configuration, and GPU kernel time ÷ Accelerate time for the same calls |
| `compare.txt` | `benchmarks/qe/compare.py` output: every run against the `cpu` run of its input |
| `tables.md` | `benchmarks/qe/tables.py` output, the tables of VAL-1 |

## Notes

- `AB_PROFILE=1` was set in every run, so each GPU call also printed a
  stage line; those lines are left out of `reports/`.
- macOS background services (Spotlight indexing, media analysis, a WebKit
  content process) were not stopped. The spread over repetitions is in
  `tables.md`: at most 0.7% of wall time per configuration.
- In every run in which the GPU path ran (candidate `gpu`, baseline
  `auto`), the gfortran runtime reports `IEEE_OVERFLOW_FLAG` signalling at
  exit; runs without GPU calls do not. Energies and band energies are
  identical in all runs. The flag comes from `MTLCreateSystemDefaultDevice()`; see VAL-1 §6, item 4 (fixed after these runs).
