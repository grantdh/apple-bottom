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

  Each round runs, rotated and reversed as before, Accelerate, the GPU, `SEAM`, `S<g>`, `S<g>b`,
  `HEAD`, `TAIL`, so `SEAM` and `S<g>b` sit next to `S<g>` in every round. Bits: `SEAM`'s repeats
  against its first and its C against the user split's; `S<g>b`'s against the user split's first;
  on N and M the part of C that `HEAD` and `TAIL` write against the user split's (first call).
- **Fixed share:** `SPLIT_G=<g>` skips the sweep and runs the fixed block at GPU part g. `SPLIT_F=<f>`
  does the same at the GPU part the library would cut at share f (`ab_split_cut` in `src/split_slice.h`,
  with the `AB_MIN_GPU_DIM` floor, default 32); a share that does not cut the axis is refused.
- **Rounds:** `SPLIT_ROUNDS=<n>` (default 3) sets the fixed block's rounds. The program refuses a count
  whose records would not fit (`MAXREC`, 2048 timed calls per process).
- **Data:** `data 0` is uniform in [−1, 1). `data 1` is QE-like: the rows of the plane-wave arrays decay
  over 9 decades. That is synthetic data chosen to exercise the exponent-range guard, not data from QE.
- **Power:**
  - The parent process samples SMC keys at 20 Hz: PDTR (DC input, the wall side on AC power), PSTR,
    PPBR (battery) and the GPU rails.
  - Sustained runs also read IOReport's GPU energy and clock residency.
  - Both are read without root through interfaces Apple does not document. Where they are unavailable,
    the power figures read `nan`.
  - `powerlog` (`benchmarks/powerlog.c`, `make build/powerlog`) streams the same SMC keys through any
    command, with IOReport's GPU and CPU clock-state residency by state name, its energy channels and
    each agent's DRAM bytes (the memory controller's counters: the CPU clusters, the GPU, DRAM), as TSV
    on the `AB_TRACE` clock, without root; `powerlog --list` shows which channels a machine reports.
    `bench_split` still runs its own sampler, so do not run it under `powerlog` until it can be turned
    off.
- **Sustained mode** (`bench_split sustain <case> <outdir> <g>`): 80 s runs, back to back, after 30 s
  of idle each:
  1. the split at GPU part g;
  2. the GPU alone;
  3. Accelerate alone;
  4. the split at the share that balances the first run's steady state.

  It logs every call's throughput, the wall power and the GPU clock over time.

## Running it

```bash
make build/bench_split
```

```bash
benchmarks/split/run.sh /path/to/out sq1x8 qe4x2
```

- **Suites:** `run.sh` with no suite runs all of them, about 35 minutes. Its header lists them.
- **Environment:** every case runs with `AB_MODE=gpu AB_MIN_GPU_DIM=0 AB_STAGING_MB=8192 AB_REPORT=1`
  and an `AB_TRACE` file per process.
- **Machine:** run on an otherwise idle Mac on AC power. Each case checks `pmset` first, and on battery
  the script stops with exit status 75. A case that did not both start and end on AC power is moved
  aside, and running the script again resumes where it stopped.
- **Memory:** the 8192² cases need about 15 GB per process (16 GB with the arms, which keep one more
  copy of C).

## Output, per case

| file | content |
|---|---|
| `<case>.np<N>.d<data>.out` | the report: sweep table, fixed block, bits, errors, power, then a `SUMMARY` line |
| `.raw.tsv` | every timed call: process, phase (0 sweep, 1 fixed block, 2 completion order), round, configuration (`cfg` index and `name`: `ACC`, `GPU`, `S<g>`, `S<g>b`, `SEAM`, `HEAD`, `TAIL`), GPU part g, repeat, times in s (`total`, the slices `cpu` and `gpu`, the caller's wait `join`, the GPU slice's start after the call's `wake`, the K axis's sum `kadd`, and the call's `t0`, `t1` on `mach_absolute_time`); for `SEAM` calls also the outcome (`KEPT`, `KEPT_4M`, `RANGE`, `GPU_FAILED`, ...; `-` on other rows), retries, operands in place, and the timeline in ns on the same clock (`at_*`: cut, thread created, CPU part start and end, GPU part start and end, join, end; `gpu_*`: first commit, `GPUStartTime`, `GPUEndTime`, last completion, encode, submit, wake and temporaries times, threadgroups, largest grid, dispatches, command buffers, kernel kinds, Metal error), 0 where a step did not happen |
| `.power.tsv` | the SMC samples |
| `.ior.tsv` | IOReport samples (power runs only) |
| `.trace.<pid>` | `AB_TRACE`, one line per GPU-routed call |
| `sustain/<case>.sustain.*` | report and `SUSTAIN` line, per-call TSV, run windows, SMC and IOReport samples |
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
