# CPU cores in use: Accelerate alone, the GPU alone and the split, 2026-10-29

How many CPU cores does each engine keep busy while it computes one GEMM? This run measured the
process's CPU time over each timed call and divided it by the call's wall time:

| Case | Layout | Accelerate alone | GPU alone | Split |
|---|---|---:|---:|---:|
| DGEMM 2048², `'N','N'` | 1 process × 8 threads | 1.97 | 0.01 | 1.85 |
| ZGEMM 2048², `'N','N'` | 1 process × 8 threads | 1.98 | 0.00 | 1.71 |
| R-K ZGEMM 500×500×18277, `'C','N'`, K split | 4 processes × 2 threads, per process | 1.87–1.92 | 0.00 | 1.69–1.80 |
| | the 4 processes together | 7.57 | 0.01 | 6.97 |

- **Accelerate alone kept about two cores busy** per process, although `VECLIB_MAXIMUM_THREADS=8`
  allowed eight. This is consistent with Accelerate computing these GEMMs on the CPU's matrix units
  rather than on all eight cores. Which cores did the work was not recorded.
- **The GPU alone left the CPU idle.** The calling thread waits for the command buffer, and the
  operands are registered with `ab_register_host_memory`, so nothing is copied.
- **The split kept 1.7–1.85 cores busy** per process: Accelerate on its slice, plus almost nothing for
  the GPU's. The split's speed comes from adding the GPU to the CPU that Accelerate alone already
  uses, not from more CPU cores.

| | |
|---|---|
| Machine | MacBook Pro (Mac14,6), Apple M2 Max, 38-core GPU, 64 GB, macOS 27.0.1 (26A434) |
| Library | Built with `make lib build/bench_split` from `bench/split-demo` at c30a298, the pre-amend version of b56327e, with the same source; its library source is release/v1.5.0 at 882d2d8: the build of the [verification](../README.md#verification-of-the-committed-program) section. `ab_build_info()`: `1.4.0 (c30a298c1795)`, the version string not yet bumped. |
| Program | `benchmarks/bench_split.c` with [`bench_split_cores.patch`](bench_split_cores.patch) applied. The patch records `getrusage(RUSAGE_SELF)` (user + system time of every thread of the process) at the start and end of each timed call, sums both over every call of the fixed block for each mode, and prints one `CORES` line per process. The timed loops are unchanged. |
| Settings | `AB_MODE=gpu AB_MIN_GPU_DIM=0 AB_STAGING_MB=8192 AB_REPORT=1`, `VECLIB_MAXIMUM_THREADS` 8 (1 process) or 2 (4 processes), operands registered, data uniform in [−1, 1). The runner is [`cores.sh`](cores.sh). |
| Date and power | 2026-10-29, 21:23:08–21:23:44 EDT, on AC power from start to end ([`out/log.txt`](out/log.txt)). |
| Machine state | The run held the exclusive benchmark lock, so no other benchmark job ran. The machine was in interactive use: the 1-minute load average read 3.6–4.4 during the run. |
| Data | [`out/`](out): each case's report with its `CORES` lines, raw timings and power samples. |

The run's own speedups are close to the published ones (it is not used in any other table here):

| | over the GPU alone | over Accelerate alone | published, over the faster engine |
|---|---:|---:|---:|
| DGEMM 2048², 1 × 8 | 1.68× | 1.85× | 1.69× |
| ZGEMM 2048², 1 × 8 | 1.50× | 2.21× | 1.52× |
| R-K 500×500, 4 × 2 | 1.81× | 2.03× | 1.83× |

## What this does not show

- **One run per case**, each summed over the fixed block's calls: 9, 9 and 12 per process for
  Accelerate alone, the GPU alone and the split, over the block's 3 rounds.
- **Registered operands only.** With copied operands, the GPU slice's copies into the library's GPU
  buffers would add CPU time that was not measured here.
- **Which cores:** performance or efficiency cores, and how busy each was, were not recorded.
- **The probe's own cost:** two `getrusage` calls per timed call, microseconds against calls of 16 ms
  and more.

## Reproducing it

From a checkout of release/v1.5.0:

```bash
git apply benchmarks/results/2026-10-28-415cb5d-hybrid-split/cpu-use/bench_split_cores.patch
```

```bash
make lib build/bench_split
```

```bash
bash benchmarks/results/2026-10-28-415cb5d-hybrid-split/cpu-use/cores.sh "$PWD/build/bench_split" /tmp/cpu-use
```

Then restore the program with `git checkout benchmarks/bench_split.c`.
