# One GEMM on Accelerate and the GPU at once

`bench_split` splits one DGEMM or ZGEMM between Accelerate (the CPU's matrix units) and apple-bottom's
GPU path, and times the split against each engine alone. Results from an M2 Max are in
[`benchmarks/results/2026-10-28-415cb5d-hybrid-split`](../results/2026-10-28-415cb5d-hybrid-split/README.md).

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
- **Layouts:** `np` processes run every repeat in lockstep, with a barrier before each repeat. The
  results use 1 process × 8 Accelerate threads, and 4 processes × 2 threads (the layout of Quantum
  ESPRESSO runs with 4 ranks).
- **One case:**
  1. a warm-up;
  2. a sweep over the GPU share f, 3 rounds with every configuration interleaved;
  3. a fixed block at the sweep's best f: 3 interleaved rounds of Accelerate alone, the GPU alone and
     the split. Its medians are the reported numbers, so choosing f does not bias them;
  4. bit checks: ≥ 12 split repeats compared with `memcmp`, both forced completion orders, hashes
     across processes, each part against the engine that computed it;
  5. errors against a compensated Dot2 reference (`tests/test_dot2.h`) on 16 sampled columns.
- **Data:** `data 0` is uniform in [−1, 1). `data 1` is QE-like: the rows of the plane-wave arrays decay
  over 9 decades. That is synthetic data chosen to exercise the exponent-range guard, not data from QE.
- **Power:**
  - The parent process samples SMC keys at 20 Hz: PDTR (DC input, the wall side on AC power), PSTR,
    PPBR (battery) and the GPU rails.
  - Sustained runs also read IOReport's GPU energy and clock residency.
  - Both are read without root through interfaces Apple does not document. Where they are unavailable,
    the power figures read `nan`.
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
- **Memory:** the 8192² cases need about 15 GB.

## Output, per case

| file | content |
|---|---|
| `<case>.np<N>.d<data>.out` | the report: sweep table, fixed block, bits, errors, power, then a `SUMMARY` line |
| `.raw.tsv` | every timed call: process, phase (0 sweep, 1 fixed block, 2 completion order), round, configuration, GPU part g, repeat, times in s |
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
- **Hashes:** `h_split`, `h_gpu` and `h_acc` in `SUMMARY` are this program's own, over the 64-bit words
  of C. Equal hashes across runs or machines mean equal bits. They cannot be compared with `AB_TRACE_HASH`.

## Analysis

| script | what it prints |
|---|---|
| `table.py <outdir>` | the results tables: medians [min–max] of the lockstep times, speedups, bits, errors, power state, sustained runs |
| `classify_trace.py <dir>` | every traced GPU call as staged path, host path or fallback, by shape, with median stage times |
| `power.py <dir>/<case>.np<N>.d<data> ...` | wall power by run of consecutive calls, trimming SMC's ~1 s lag |
