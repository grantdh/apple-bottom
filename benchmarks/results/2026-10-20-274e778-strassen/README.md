# Strassen-Winograd speed and accuracy, 2026-10-20

`build/bench_strassen` (`benchmarks/bench_strassen.c`), library built at
274e778 (`ab_build_info()`: `1.4.0 (274e778d5065)`, a clean tree), each group
under `machlock.py bench`. Summary tables: [docs/STRASSEN.md](../../../docs/STRASSEN.md).

Machine: MacBook Pro (Mac14,6), Apple M2 Max, 38-core GPU, 8P+4E CPU, 64 GB,
macOS 27.0.1 (26A434), Apple clang 17.0.0, SDK 26.0. The accuracy run, whose
results do not depend on it, was on battery. The speed and energy runs were
on AC power, two passes of the speed groups 20:48–20:54 and 20:59–21:03:

- `squares_*`, `zgemm_*`: both passes.
- `big_*`: the second pass only. The first, `big_204900.csv`, began 4 s after
  machlock logged a switch to battery power and ended on AC; its 8192
  classical median (1997 ms, best 1569 ms) is an outlier and is not used.
- `rect_*`, `abi_*`, `energy_*`: one pass each.

**What is timed.** The native API with A, B and C resident in GPU memory
(uploaded once; no conversion or transfer in the timings), unless the row's
mode is `abi`. Each method warms up for 0.5 s, then runs 5 times (3 for ZGEMM
8192). The classical DD GEMM (`ab_dgemm_scaled`, `ab_zgemm`) and Strassen
(`ab_dgemm_strassen`, `ab_zgemm_strassen`) alternate, back to back; then
Accelerate (`cblas_dgemm` / `cblas_zgemm`, FP64, all cores) runs on its own.
`strassen-first` is the first Strassen call of the shape, which grows the
workspace (and, the first time in the process, compiles its kernel). Columns:

| column | |
|---|---|
| `wall_med_ms`, `wall_min_ms` | median and best wall time per call |
| `gpu_med_ms` | median command-buffer GPU time (`GPUEndTime − GPUStartTime`) |
| `temps_med_ms` | median host time creating temporaries (classical ZGEMM's four per call; Strassen's workspace growth) |
| `submit_med_ms`, `wake_med_ms` | median time from commit to GPU start, and from GPU end to the waiting thread |
| `gflops` | 2MNK (8MNK for ZGEMM) per median call: Strassen's rate is effective, Accelerate's and the classical ZGEMM's count operations Gauss-3M does not perform |

`abi` rows time `ab_dgemm_blas` / `ab_zgemm_blas` end to end (column-major
FP64 in and out: packing, FP64 ⇄ DD conversion, transfer, epilogue), with
`AB_MODE=gpu` and `AB_STRASSEN` unset or 1; the method column is the path
the call took, `+strassen` where `AB_STRASSEN` applied.

**Accuracy.** `accuracy_*.csv`: the errors of the classical DD GEMM,
Strassen and Accelerate against Dot2 references (`tests/test_dot2.h`) of all
of C, computed beforehand by `bench_strassen reference` (CPU only; 5–53 s
per matrix). Data kinds and columns are described at the top of
`benchmarks/bench_strassen.c`.

| File | |
|---|---|
| `accuracy_*.csv` | 6 shapes × 3 kinds of data × 3 methods |
| `squares_*.csv`, `big_*.csv` | square DGEMM, 2048–4096 and 6144–8192 |
| `rect_*.csv` | rectangular, odd and long-K DGEMM shapes |
| `zgemm_*.csv` | square ZGEMM, 2048–8192 |
| `abi_*.csv` | the BLAS ABI end to end |
| `energy_*.txt` | per 10 s window: GPU energy (IOReport "GPU Energy") and SMC power (PDTR DC input, PSTR system total) |
| `storage_mode_experiment.txt` | the same Strassen calls with the workspace in shared storage (a one-off build) |
| `*.err` | each run's stderr (build identity, shapes) |

**Energy.** `energy_*.txt` comes from a no-sudo energy probe that calls the
library: `ab_dgemm_scaled` or `ab_dgemm_strassen` on resident 4096² and
8192² matrices in a loop, then `cblas_dgemm`, 30 s per GPU phase (20 s for
Accelerate) in 10 s windows after a 3 s warm-up. It reads private IOReport
and SMC interfaces, so it is kept outside this repository. The
battery was recharging during the run (PDTR stayed at ~137 W, 121 W even at
idle), so docs/STRASSEN.md uses PSTR above the idle phase instead.
