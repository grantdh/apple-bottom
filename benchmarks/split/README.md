# One GEMM on Accelerate and the GPU at once

`bench_split` splits one DGEMM or ZGEMM between Accelerate (the CPU's matrix units) and apple-bottom's
GPU path, and times the split against each engine alone. Results from an M2 Max are in
[`benchmarks/results/2026-10-28-415cb5d-hybrid-split`](../results/2026-10-28-415cb5d-hybrid-split/README.md).

The split is made here, in user code. The fixed block also times the library's own split of the
same call through its seam (`SEAM`, below), which `AB_SPLIT` uses inside `dgemm_` and `zgemm_`.

## What it does

- **CPU slice:** `cblas_dgemm`/`cblas_zgemm` from Accelerate on its part of the call, on the calling
  thread, with `VECLIB_MAXIMUM_THREADS` threads.
- **GPU slice:** `ab_dgemm_blas`/`ab_zgemm_blas` with `AB_MODE=gpu` on the rest, from a second thread.
  - This is the public BLAS ABI's staged path: conversion to double-float and the epilogue run on the
    GPU, and the buffers are pooled.
  - A, B and C are registered once with `ab_register_host_memory`, so the GPU reads them in place.
    Registration is setup and is not timed. `SPLIT_NOREG=1` skips it, and the staged path then copies
    the operands.
- **Axes:**
  - `N` splits the columns of C, `M` its rows.
  - `K` splits the sum: the GPU writes its partial product into a separate array, which is added to C
    after both slices finish, C = fl(fl(βC₀ + P_cpu) + P_gpu).
- **Layouts:** `np` processes (1 to 8) run every repeat in lockstep, with a barrier before each
  repeat. The results use 1 process × 8 Accelerate threads, and 4 processes × 2 threads (the layout of
  Quantum ESPRESSO runs with 4 ranks). The `qe8x1` suite runs 8 processes × 1 thread, the layout of
  pw.x at 8 ranks.
- **pw.x's shapes at 8x1:** `A8_UN512` ('N','N' 9139×500×512, β = 1, N axis), `A8_RK512` ('C','N'
  512×500×9139) and `A8_RK1000` ('C','N' 500×1000×9176, ldc 1000), both K axis, from input A;
  `B8_UM1152` ('N','N' 8927×160×1152, β = 1, M axis) and `B8_RK1152` ('C','N' 1152×160×8988, K axis)
  from input B. Shapes, leading dimensions and scalars are one rank's calls in 1.5.1's `AB_TRACE` of
  pw.x at 8 ranks with `-nk 4`; each rank's M (or K) differs a little (8918-9176).
- **One case:**
  1. a warm-up;
  2. a sweep over the GPU share f, 3 rounds with every configuration interleaved;
  3. a fixed block at the sweep's best f: 3 interleaved rounds of Accelerate alone, the GPU alone and
     the split, and the four arms below. Its medians are the reported numbers, so choosing f does not
     bias them;
  4. bit checks: ≥ 12 split repeats compared with `memcmp`, both forced completion orders, hashes
     across processes, each part against the engine that computed it;
  5. errors against a compensated Dot2 reference (`tests/test_dot2.h`) on 16 sampled columns.
- **Arms** (the fixed block, at the split's GPU part g, each with as many repeats per round as the
  split; `SPLIT_ARMS=0` leaves them out):
  - `SEAM`: the library's split of the same call through its seam (`ab_internal_dgemm_split` /
    `ab_internal_zgemm_split` in `src/split_slice.h`): staged path, free completion order, a thread
    per call, and on K a partial product allocated per call. Each call's outcome and timeline
    (`ab_internal_last_split`) go to `raw.tsv`.
  - `S<g>b`: a second user-code split, identical to `S<g>`, so the user split can be timed against
    itself (G3's null run).
  - `HEAD`: the CPU slice alone, on the calling thread.
  - `TAIL`: the GPU slice alone, on the GPU thread, with the caller waiting and doing nothing else.

  With the arms, each round of the fixed block runs:
  1. an idle gap (`SPLIT_GAP_S`, default 1 s);
  2. the CPU-only configs, Accelerate and `HEAD` (their order alternating by round), so neither
     follows a config that used the GPU;
  3. the GPU alone and `TAIL` (alternating likewise);
  4. one untimed call of `S<g>`, then the trio `SEAM`, `S<g>`, `S<g>b` repeat by repeat, in round
     r mod 6's order of a Williams design: over every 6 rounds each of the three runs first equally
     often and comes right before and right after each of the others equally often, none follows
     Accelerate, the GPU, `HEAD` or `TAIL`, and the calls paired by (round, repeat) sit within one
     repeat of the trio. Use a multiple of 6 rounds; the report says when the count is not one.

  Before the block the split runs untimed for `SPLIT_WARM_S` seconds (default 2 with the arms, 0
  without), so the block does not start cool. `raw.tsv`'s `prev` column names each call's
  predecessor in its process (`idle` after the gap). Without the arms the order is as before:
  Accelerate, the GPU and the split, rotated and reversed by round. Bits: `SEAM`'s repeats against its
  first and its C against the user split's; `S<g>b`'s against the user split's first; on N and M the
  part of C that `HEAD` and `TAIL` write against the user split's (their first call after the user
  split's first, so from round 1).
- **Fixed share:** `SPLIT_G=<g>` skips the sweep and runs the fixed block at GPU part g. `SPLIT_F=<f>`
  does the same at the GPU part the library would cut at share f (`ab_split_cut` in `src/split_slice.h`,
  with the `AB_MIN_GPU_DIM` floor, default 32); a share that does not cut the axis is refused.
- **Rounds:** `SPLIT_ROUNDS=<n>` (default 3) sets the fixed block's rounds, and `SPLIT_REPS=<n>` gives
  every config n repeats per round instead of the case's own (3 or 2 for Accelerate and the
  GPU, 4 for the split and the arms; SUMMARY's `reps=` gives them). The program refuses counts whose
  records would not fit (`MAXREC`, 2048 timed calls per process) or whose round would exceed `MAXSTEP`
  (512) calls. Each process's watchdog (600 s, 2400 s for the 8192²
  cases) restarts for each round, so more rounds do not trip it.
- **Data:** `data 0` is uniform in [−1, 1). `data 1` is QE-like: the rows of the plane-wave arrays decay
  over 9 decades. That is synthetic data chosen to exercise the exponent-range guard, not data from QE.
- **Power:**
  - The parent process samples SMC keys at 20 Hz: PDTR (DC input, the wall side on AC power), PSTR,
    PPBR (battery) and the GPU rails.
  - Sustained runs also read IOReport's GPU energy and clock residency.
  - **Outside load** (run mode): process 0 takes a snapshot at each round boundary of the fixed block,
    between timed calls: the host's CPU ticks, this run's own CPU time (its processes', by
    `proc_pidinfo`) and, unless `SPLIT_SAMPLER=0`, IOReport's GPU state residency. The CPU used by
    everything else is the busy share times the CPUs minus this run's own. Root's processes cannot be
    read one by one without root, so `kernel_task`'s work counts as outside. With the arms each round
    has a GPU-idle stretch, the idle gap and the CPU-only configs, when none of this run's processes
    uses the GPU: over it `.rounds.tsv` and the report give other processes' CPU in cores
    (`fcpu_idle`, without this run's GPU driver work) and the GPU's busy fraction (`gpu_busy_idle`,
    someone else's GPU work). Over the whole round they give other processes' CPU too (`fcpu_round`,
    which includes this run's GPU driver work in `kernel_task`). `SUMMARY` has the largest of each
    (`fcpu_max`, `gpu_idle_busy_max`, `fcpu_round_max`). The AB_SPLIT pilot flagged 1.5 cores (by `ps`,
    without `kernel_task`) and saw outside GPU work as busy 0.34-0.36 against 0.07-0.20 without it.
  - Both are read without root through interfaces Apple does not document. Where they are unavailable,
    the power figures read `nan`.
  - `powerlog` (`benchmarks/powerlog.c`, `make build/powerlog`) streams the same SMC keys through any
    command, with IOReport's GPU and CPU clock-state residency by state name, its energy channels and
    each agent's DRAM bytes (the memory controller's counters: the CPU clusters, the GPU, DRAM), as TSV
    on the `AB_TRACE` clock, without root; `powerlog --list` shows which channels a machine reports.
    To run `bench_split` under `powerlog`, set `SPLIT_SAMPLER=0`: `bench_split` then samples nothing
    itself, and its power fields read `nan`, so two samplers never run at once.
- **Sustained mode** (`bench_split sustain <case> <outdir> <g> [secs] [idle] [steady_after]`): runs of
  back-to-back calls, 80 s each after 30 s of idle:
  1. Accelerate alone (a control);
  2. in a shuffled order (`SPLIT_SEED=<n>` repeats one; the report prints the seed): the split at GPU
     part g, the GPU alone, the CPU slice alone (`HEAD`) and the GPU slice alone (`TAIL`);
  3. the split at the share that balances the split run's steady state;
  4. Accelerate alone again (the control at the end).

  C is not reset between calls, so the GPU does not wait for a copy of C between them. With β = 0 every
  call writes the same C, and every 5th is compared with the run's first; with β = 1 C accumulates and
  is not compared. Every call is kept, however many (the records grow as needed). `SPLIT_NOREG=1`
  copies the operands, as in the run mode. It logs every call's throughput, the wall power and the GPU
  clock over time, and reports the slices alone against the split: (HEAD + TAIL) ÷ split, and how much
  longer each slice takes in the split than alone. `SUSTAIN`'s first fields keep their order (split,
  GPU alone, Accelerate alone at the start, rebalanced); the rest are added after them.

## Running it

```bash
make build/bench_split
```

```bash
benchmarks/split/run.sh /path/to/out sq1x8 qe4x2
```

- **Suites:** `run.sh` with no suite runs all of them but `smoke` and the G3 suites, about 50 minutes.
  Its header lists them.
- **Environment:** every case runs with `AB_MODE=gpu AB_MIN_GPU_DIM=0 AB_STAGING_MB=8192 AB_REPORT=1`
  and an `AB_TRACE` file per process.
- **G3 suites** (`g3-1x8-s8192`, `g3-1x8-s2048`, `g3-4x2-s8192`, `g3-4x2-s2048`, or `g3` for all four;
  not in the default run): the six squares and the seven QE-shaped cases of the 2026-10-28 results with
  copied operands (`SPLIT_NOREG=1`), at 1×8 and at 4×2, with `AB_STAGING_MB` 8192 and 2048, so that the
  library's seam and the user split meet the staged path's copies and a small pool. Each runs 6 rounds
  of 2 repeats (`SPLIT_ROUNDS=6 SPLIT_REPS=2`): the trio's order is balanced over the 6 rounds, and
  rounds are the unit of the by-round interval in `g3.py`. They run with `AB_TRACE` and `AB_REPORT`
  off: those profile the user split's GPU slices (`ab_[dz]gemm_blas` takes the profiler's lock and
  writes a trace line per call) but not the seam, which the library does not profile, so with them on
  the comparison would lean towards the seam. `raw.tsv` has the seam's timeline either way. `ZGEMM
  8192²` is not run at 4×2 (see Memory). `g3.py` reads the results.
- **Machine:** run on an otherwise idle Mac on AC power, from an adapter of at least 90 W
  (`SPLIT_MIN_W`), with the battery at 95% or more (`SPLIT_MIN_BATT`) and neither charging nor finishing
  its charge, as the AB_SPLIT pilot required. The script checks all three at the start and before each
  case, waiting up to 900 s (`SPLIT_BATT_WAIT_S`) for the battery, and stops with exit status 75 if a
  check fails. `bench_split` prints the adapter's watts on its `start:` and `end:` lines (and its
  sustained mode checks them before every run); a case whose report does not start and end on AC power
  from such an adapter is moved aside, and running the script again resumes where it stopped.
  `SPLIT_DRY=1` makes the checks, without waiting, and prints each case's command without running it.
  `power.log` has the power source, the battery's charge and state, the adapter's watts and the load
  averages before and after every case, and `raw.tsv` the load average after every call (`load1`).
- **Settings per suite:** a suite's own variables (`SPLIT_NOREG=1`, `AB_STAGING_MB=2048`, `AB_TRACE=`, ...)
  come last on the `env` line, so they override the common ones and the trace file.
- **Memory:** the 8192² cases needed about 15 GB in the 2026-10-28 runs; the arms keep one more copy of
  C (1 GiB at `ZGEMM 8192²`). An estimate from the code, per process:
  - Each process holds A, B, C, C₀, the partial P and four reference copies of C (the user split's, the
    GPU's, Accelerate's and the seam's): nine arrays of N² elements, 9 GiB at `ZGEMM 8192²` and 4.5 GiB
    at `DGEMM 8192²`. With copied operands the GPU's buffers add their double-float copies of A, B and C
    (3 GiB for ZGEMM, 1.5 GiB for DGEMM) and, for ZGEMM, the Gauss-3M temporaries (4 N² double-float
    values, 2 GiB), whatever `AB_STAGING_MB` keeps between calls.
  - At 4×2 that is about 4 × 14 = 56 GiB for `ZGEMM 8192²`, more than a 64 GB Mac can give without
    swapping, so the G3 suites leave it out at 4×2. `DGEMM 8192²` needs about 4 × 6 = 24 GiB and is run.

## Output, per case

| file | content |
|---|---|
| `<case>.np<N>.d<data>.out` | the report: sweep table, fixed block, bits, errors, power, then a `SUMMARY` line |
| `.raw.tsv` | every timed call: process, phase (0 sweep, 1 fixed block, 2 completion order), round, configuration (`cfg` index and `name`: `ACC`, `GPU`, `S<g>`, `S<g>b`, `SEAM`, `HEAD`, `TAIL`), GPU part g, repeat, times in s (`total`, the slices `cpu` and `gpu`, the caller's wait `join`, the GPU slice's start after the call's `wake`, the K axis's sum `kadd`, and the call's `t0`, `t1` on `mach_absolute_time`); for `SEAM` calls also the outcome (`KEPT`, `KEPT_4M`, `RANGE`, `GPU_FAILED`, ...; `-` on other rows), retries, operands in place, and the timeline in ns on the same clock (`at_*`: cut, thread created, CPU part start and end, GPU part start and end, join, end; `gpu_*`: first commit, `GPUStartTime`, `GPUEndTime`, last completion, encode, submit, wake and temporaries times, threadgroups, largest grid, dispatches, command buffers, kernel kinds, Metal error), 0 where a step did not happen |
| `.power.tsv` | the SMC samples |
| `.ior.tsv` | IOReport samples (power runs only) |
| `.rounds.tsv` | per round of the fixed block: its start, the end of its GPU-idle stretch and its end (`t0`, `idle_t1`, `t1`), other processes' CPU in cores over the stretch and the round (`fcpu_idle`, `fcpu_round`), the GPU's busy fraction over the stretch (`gpu_busy_idle`) |
| `.trace.<pid>` | `AB_TRACE`, one line per GPU-routed call |
| `sustain/<case>.sustain.*` | report and `SUSTAIN` line, per-call TSV (`run` is the run's id: 0 split, 1 GPU alone, 2 Accelerate alone, 3 rebalanced, 4 `HEAD`, 5 `TAIL`, 6 Accelerate again; `name` the config), run windows in the order run (`position`), SMC and IOReport samples |
| `power.log` | `pmset` and load before and after every case |

## Notes on the numbers

- **Per-call power:** `SUMMARY`'s `pdtr_*` fields average the SMC samples inside each timed call, with no
  lag trimming. SMC's PDTR lags the load by about a second, so they mean little for calls shorter than
  that. For wall power, use the `power4x2` suite (6 s runs), the sustained runs, or `power.py`.
- **GPU clock:** IOReport gives the time spent in each DVFS state. `GPU_MHZ` in `bench_split.c` maps the
  states to MHz with the M2 Max's table, so the MHz values are valid on an M2 Max only. Other chips need
  their own table (`ioreg`'s `voltage-states9`).
- **Error metric:** max |C − R| / (|op(A)||op(B)| + |βC₀|) over the sampled columns, with complex
  moduli, in u² = 2⁻⁴⁸. The "bound 8 u^2" in the report is the library tests' long-K bound, which they
  apply to each part's own scale (`tests/test_dot2.h`); it is quoted here as a reference.
- **Arms in `SUMMARY`:** `seam_ms`, `split2_ms`, `head_ms`, `tail_ms` (lockstep medians), `seam_ratio`
  and `null_ratio` (medians over the (round, repeat) pairs of `SEAM` ÷ `S<g>` and `S<g>b` ÷ `S<g>`),
  `pairs`, `seam_out` (outcome counts), `seam_rep_diff`, `seam_vs_split` (elements of C that differ
  from the user split's), `split2_rep_diff`, `head_vs_split` and `tail_vs_split` (-1: not compared).
- **Hashes:** `h_split`, `h_gpu` and `h_acc` in `SUMMARY` are this program's own, over the 64-bit words
  of C. Equal hashes across runs or machines mean equal bits. They cannot be compared with `AB_TRACE_HASH`.

## Analysis

| script | what it prints |
|---|---|
| `table.py <outdir>` | the results tables: medians [min–max] of the lockstep times, speedups, bits, errors, power state; with the arms, the seam and twin against the user split (paired medians), the slices alone and the seam's outcomes; sustained runs |
| `classify_trace.py <dir>` | every traced GPU call as staged path, host path or fallback, by shape, with median stage times |
| `power.py <dir>/<case>.np<N>.d<data> ...` | wall power by run of consecutive calls, trimming SMC's ~1 s lag |
| `g3.py <dir> ... [--tsv FILE]` | G3 per case and layout: `SEAM` ÷ `S<g>` and `S<g>b` ÷ `S<g>` paired by (round, repeat), each as the median ratio, a BCa bootstrap interval and the Hodges-Lehmann estimate with its exact Wilcoxon interval; the verdict (the upper bound of `SEAM` ÷ `S<g>` against max(1.03, the twin's upper bound), on the BCa bounds, the Hodges-Lehmann ones as a check); and the seam's timeline per call (init, thread start, CPU and GPU parts, the wait for the GPU, submit, GPU execution, wake) beside the user split's parts. BCa needs scipy (`/Library/Frameworks/Python.framework/Versions/3.13/bin/python3` on the M2 Max) |
