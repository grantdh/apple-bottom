# 1.6.0 before its tag, 2026-11-06

Commit 8b48826 on `release/v1.6.0`, built once and frozen before any run,
against v1.5.1 frozen the same way. This directory holds:
- Strassen-Winograd's speed and accuracy on 1.6.0;
- the data the range guard lets through;
- the narrow-tile rule's effect on whole calls, in paired runs;
- 1.6.0 against 1.5.1: result bits, exports, Netlib, LAPACK and Quantum ESPRESSO.

A later commit that changes only documentation or source comments leaves these
results standing; any other change does not.

| | |
|---|---|
| Machine | MacBook Pro (Mac14,6), Apple M2 Max, 38-core GPU, 64 GB, macOS 27.0.1 |
| Library | 8b48826 (`v1.5.1-42-g8b48826`), `make clean && make -j8 lib` in a fresh clone with 0 modified tracked files, Apple clang 17.0.0. Reports `apple-bottom 1.6.0 (8b4882691793)`. |
| Baseline | v1.5.1 = 93b1f2d, built the same way |
| Lock, power | Timing runs held machlock's exclusive bench lock, on AC from a 140 W adapter. `conditions.log` records adapter, power source and load at the start of each Strassen group and each narrow-tile shape. For Quantum ESPRESSO, `qe-release.jsonl` and `qe-gpu.jsonl` record the adapter's watts and `load1` at the start of each block, and a block ran only on AC. Bit and conformance runs took the shared lock. |

## Files

| path | content |
|---|---|
| `strassen/` | `bench_strassen` CSV, one per group and start time; every run's banner read `apple-bottom 1.6.0 (8b4882691793) on Apple M2 Max`. `docs/STRASSEN.md` takes its tables from these. `zgemm8kpool` ran with `AB_STAGING_MB=4000`; `big_173814` is the first pass that document leaves out. |
| `guard-gaps/` | `probe_guard_gaps` output: `default.csv`, `forced_gpu.csv` (`AB_MODE=gpu AB_MIN_GPU_DIM=0`), `cpu.csv` (`AB_MODE=cpu`) |
| `narrow/` | First block of forced-GPU call times, 1.5.1 then 1.6.0, 10 pairs per shape |
| `narrow2/` | Second block, 15 pairs per shape, order alternating, with `load1` per run |
| `narrow-summary.txt` | Paired ratios and bootstrap intervals of both blocks |
| `comparison/compare.txt` | Output hashes, pair by pair of runs |
| `comparison/vs-1eab4e9.txt` | The same labels against the 2026-11-04 comparison's runs of 1eab4e9 |
| `comparison/ref-data.txt`, `ref-data-classes.txt` | Errors against a compensated reference, and IEEE mismatches against Netlib, for the `data` and `dev` suites |
| `comparison/netlib-summary.txt`, `lapack-summary.txt` | Netlib's BLAS test programs and LAPACK's test suite, per routine and program |
| `comparison/si64-compare.txt`, `qe-release.jsonl`, `qe-release-summary.txt`, `qe-gpu.jsonl` | Quantum ESPRESSO si64 |
| `comparison/exports-summary.txt` | Exported symbols and linked libraries |
| `comparison-1eab4e9/` | Summaries from the 2026-11-04 comparison, whose results 8b48826 reproduces bit for bit outside the cases below: `perf-analysis.txt` and `perf-followups.txt` (call times) |

The bit comparison, reference and conformance runs use a harness kept outside
this repository; this README describes each run. `probe_guard_gaps` and
`bench_strassen` are in `benchmarks/`.

## Result bits against 1.5.1

| Runs | GEMM cases | Different |
|---|---:|---:|
| Default routing | 5363 | 0 |
| `AB_MODE=gpu AB_MIN_GPU_DIM=0` | 5363 | 68 |
| `AB_MODE=cpu` | 1669 | 0 |
| Forced GPU, 4096³ and 4097 × 4095 × 4096 | 72 | 0 |
| Native API, printed results (144 GEMM lines, 39 of `ab_dsyrk`, `ab_zherk`, `ab_dtrsm`, `ab_dgesv_mpir`, `ab_matrix_add` and `ab_matrix_sub`) | 183 lines | 0 |

- **The 68** are `zgemm_` calls with NaN or Inf in C and beta ≠ 0, which now run
  on Accelerate (e7e01ae). In the `data` suite under forced GPU, elements where
  Netlib gives a finite value and the result is NaN fall from 28 to 0, and elements
  where Netlib gives NaN and the result is finite rise from 0 to 40, the same 40 as
  under default routing: Accelerate keeps the other part of a non-finite element
  (`ref-data.txt`; at 1eab4e9 the `data` forced-GPU ZGEMM row read nan2fin 0, fin2nan 28).
- **The corpus** covers every TRANS pair, scalars and padded leading dimensions
  through `dgemm_`, `zgemm_` and the device API. `zgemm_` runs 22 data classes:
  `unif`, `zeros`, `sub`, `spread`, `cancel`, `dec25`, `opp300`, `mix16`, `mix40`,
  `mix300`, `tiny1042`, `tinyc98`, `tinyc100`, `tinyc130`, `timag80`, `timag100`,
  `timag130`, `nanA`, `infA`, `ninfB`, `nanC`, `infC`. `dgemm_` runs the same
  without the three `timag` classes. The device API runs `unif`, `mix16`, `spread`,
  `nanA` and, for ZGEMM, `timag100`. The native API has no leading dimension and
  uses its own data: uniform, rows scaled by 2^±16, everything times 2^-100 and
  imaginary parts times 2^-30; `ab_zgemm_ex` runs the nine TRANS pairs on the
  uniform data.
- **Repeat.** A second forced-GPU run gives the same hashes in all 3853 cases
  compared, and a second native run the same 183 lines.
- **BLAS-2 rows in `compare.txt` are not a version comparison.** 1.5.1 exports no
  `dgemv_`, `zgemv_` or `zhemv_`: its rows are Accelerate's legacy symbols, reached
  through the re-export, and 1.6.0's fallback is `cblas_*`.

Against 1eab4e9, the commit of the 2026-11-04 comparison, the only differences in
any label are the same 68 cases (`vs-1eab4e9.txt`), including the `AB_PRECISION=fast`
and `AB_STRASSEN=1` runs.

## `AB_PRECISION=fast`

Call times through the BLAS ABI with the GPU forced, at 1eab4e9 and not repeated at
8b48826, whose fast-mode bits are the same (`comparison/vs-1eab4e9.txt`): 1.05–1.08×
faster than the default for DGEMM and ZGEMM of N = 2048 and 4096
(`comparison-1eab4e9/perf-followups.txt`).

## Exports

Exported from the dylib: 145 → 237 symbols, none removed
(`ab_internal_prefetch_dispatches` is kept by 27fc189). From the archive: 188 →
295, two removed, both compiler-generated `___block_descriptor_*` names.
1.6.0 also links IOKit.

## Netlib and LAPACK

Netlib's test ratio has a limit of 16.

| Routine | Default routing, both | Forced GPU, 1.5.1 | Forced GPU, 1.6.0 |
|---|---|---:|---:|
| DGEMM | pass | 42.52 | 42.52 |
| ZGEMM | pass | 39.89 | 39.89 |
| DGEMV | pass | pass | 36.72 |
| ZHEMV | pass | pass | 27.79 |
| every other routine | pass | pass | pass |

LAPACK's test suite, tests counted across the 44 double and double-complex programs run
here (not its single-precision programs, `dstest` or `zctest`). Eight contribute none on
any library: `dbal` reads an input that stops it, and `dbak`, `dgbal`, `dgbak`,
`zbal`, `zbak`, `zgbal` and `zgbak` print a largest error but no pass or fail line:

| Run | Failed | Passed |
|---|---:|---:|
| Netlib's BLAS | 0 | 2,037,333 |
| 1.5.1, default routing | 2 | 2,037,331 |
| 1.6.0, default routing | 1 | 2,037,332 |
| 1.5.1, forced GPU | 3134 | 1,527,076 |
| 1.6.0, forced GPU | 4365 | 1,525,850 |
| 1.6.0, forced GPU, Netlib's `dgemv`, `zgemv`, `zhemv` linked first | 3134 | 1,527,076 |

- **Attribution.** In the last run each of the 44 programs prints the same results
  as under 1.5.1, once `AB_REPORT` and timing lines are removed (`dbal` differs only
  in the addresses of its backtrace). The 1231 additional counted failures come from
  BLAS-2 calls of every size reaching the GPU at double-float accuracy.
- **`dec` and `zec` under forced GPU.** On both libraries, and in the last run, these
  programs report test ratios above LAPACK's limit of 20: `Error in DTRSYL` (37.7) and
  `Error in DTRSEN` (24.3, or 24.0 on 1.6.0) in `dec`, and `Error in ZTRSYL` (49.9) in
  `zec`. They print these in place of the line saying all tests passed, which is the
  line that gives their total, and they print no count of failed tests. Only their
  error-exit tests (35 and 33) reach the Passed column. So `dec`'s 501,251 tests and
  `zec`'s 5,966 are missing from the forced-GPU rows, and the Failed column does not
  include these two programs' failures.
- **Default routing.** The 1 failure is in `znep`.
- **Illegal arguments.** With apple-bottom linked first, the error-exit parts of
  `dblat2` and `zblat2` end with status 255, as those of `dblat3` and `zblat3`
  already did.

## Quantum ESPRESSO

QE develop c7587f9, si64 inputs A (500 bands) and B (160 bands). QE was not linked
again for this run. The 1.5.1 binary is the 2026-11-04 comparison's. The 1.6.0
binary is that comparison's 1eab4e9 binary with its rpath pointed at the frozen
8b48826 library. The exports of those two 1.6.0 libraries differ by one symbol,
`ab_internal_prefetch_dispatches`, which `pw.x` does not import.
- **Default routing:** 8 ranks, `tprnfor`. The 1.5.1 outputs are the 2026-11-04
  comparison's, not rerun. Total energies, SCF iteration energies, all bands and all
  forces are identical at printed precision between the two libraries
  (`si64-compare.txt`).
- **Routing in those runs:** `AB_REPORT` gives `gpu_calls=0` for `zgemm` and `dgemm`
  on every rank. ZGEMV: 16 of 32 calls on the GPU for input B, 0 of 28 for input A.
  Each rank that makes such a call starts Metal first.
- **Time at default routing, at 1eab4e9:** 0.998–1.000× 1.5.1's at 1 and 8 ranks, 10
  pairs per configuration, with every 95% interval spanning 1
  (`comparison-1eab4e9/perf-analysis.txt`).
- **`AB_MODE=gpu`, the release check's mode** (`qe-release.jsonl`, `qe-release-summary.txt`):
  1.5.1, 1.6.0 and `AB_MODE=cpu` on 1.6.0, two blocks of the three runs per configuration
  in opposite order, without `tprnfor`:

  | Input, ranks | 1.5.1 gpu | 1.6.0 gpu | 1.6.0 cpu | 1.6.0 gpu ÷ 1.6.0 cpu | 1.6.0 gpu ÷ 1.5.1 gpu |
  |---|---:|---:|---:|---:|---:|
  | B, 8 | 54.9, 55.1 s | 55.7, 55.6 s | 53.9, 53.5 s | 1.03, 1.04 | 1.015, 1.009 |
  | A, 8 | 102.3, 104.4 s | 102.9, 103.3 s | 95.2, 93.8 s | 1.08, 1.10 | 1.006, 0.989 |
  | B, 1 | 192.8, 192.6 s | 193.1, 193.4 s | 183.2, 183.1 s | 1.05, 1.06 | 1.001, 1.004 |

  Within each configuration all six runs give the same total energy and SCF iterations.
- **`AB_MODE=gpu AB_MIN_GPU_DIM=0`** (`qe-gpu.jsonl`), which also sends calls below the
  floor of 32 to the GPU: three pairs of 1.5.1 and 1.6.0 per configuration. 1.6.0 takes
  1.002–1.019× 1.5.1's time on A at 8 ranks, 0.989–1.010× on B at 8 ranks and
  0.999–1.004× on B at 1 rank, with equal energies. These runs take 107–109 s, 63–64 s
  and 209–211 s, longer than with the floor.

## Strassen-Winograd

Run under the bench lock with `bench_strassen speed`, `abi` and `accuracy`; see
`docs/STRASSEN.md` for the tables and how to run them. Accuracy references are
computed once on the CPU (`bench_strassen reference`).

## The guard's gaps

`probe_guard_gaps`, three runs. Under default routing the large T,T ZGEMM rows and the
ZGEMV and ZHEMV rows other than `tiny` take the BLAS ABI's GPU path, and the other BLAS
rows run on Accelerate. The two `dev_zgemm` rows run on the GPU in all three runs, because
the device API does not read `AB_MODE` and has no fallback. `docs/vv/LIMITATIONS.md` ("What the guard
misses") states the triggers and reasons. Beyond the probe, the worst imaginary part
on `spread` data in the `data` suite is 887 u², at 257 × 129 × 300 'N','C' under
forced GPU, on a scale that adds FP64 rounding terms to |Ar||Bi| + |Ai||Br|, and in the
`dev` suite 166 u² for `ab_dev_zgemm` 'C','N' 544 × 481 × 600 (`ref-data-classes.txt`).

## Narrow tiles in whole calls

`abbench`: forced GPU through `dgemm_` and `zgemm_`, the median of the calls in about
1 s after a warm-up of about 1 s, each side in its own process.

| Call | 1.5.1 | 1.6.0 | 1.6.0 ÷ 1.5.1, median pair [95% CI] | Block |
|---|---:|---:|---:|---|
| DGEMM 544² 'N','N' | 1.19 ms | 1.01 ms | 0.850 [0.847, 0.858] | first |
| DGEMM 832² 'N','N' | 2.65 ms | 2.79 ms | 1.053 [1.051, 1.055] | first |
| DGEMM 600 × 600 × 9140 'T','N' | 16.00 ms | 14.94 ms | 0.934 [0.933, 0.935] | second |
| ZGEMM 1024 × 40 × 4608 'N','N' | 6.69 ms | 6.41 ms | 0.960 [0.948, 0.961] | second |
| ZGEMM 500 × 500 × 9140 'C','N' | 29.08 ms | 28.61 ms | 0.999 [0.980, 1.002] | second |

- **ZGEMM 500 × 500 × 9140.** Single pairs range over 0.96–1.00 in the second block
  and 0.98–1.06 in the first. Process medians group at a few levels (1.6.0: 28.5 ms in
  15 of 25 processes, 29.1 ms in 7, and 28.07, 30.49 and 30.93 ms once each; 1.5.1:
  28.6 ms in 3, 29.1 ms in 19, 29.6 ms in 3). `abbench` records each process's median
  and minimum, not each call, so these runs do not show whether a process stays on one
  level. The 2026-11-04 comparison measured 1.057 [1.019, 1.076] in one block of 10
  pairs; these 25 pairs do not confirm a slowdown.
- **Outside load.** The last five pairs of ZGEMM 1024 × 40 × 4608 in the second block
  ran as `load1` rose from 1.49 to 8.92 and took about twice as long on both sides.
  The first block logged load only at the start of each shape. Its ZGEMM 1024 × 40 ×
  4608 started at `load1` 19.48 and took about twice as long in the first five pairs
  and on the 1.5.1 side of the sixth. Its DGEMM 600 × 600 × 9140 started at 4.20 and
  took 1.3–1.4× as long in the last three pairs. The table uses the second block for
  both.
