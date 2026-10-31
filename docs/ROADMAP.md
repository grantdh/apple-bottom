# apple-bottom Roadmap

See [CHANGELOG.md](../CHANGELOG.md) for what has shipped.

## 1.5 — the BLAS ABI's work moves to the GPU (this release)

- **The staged path.** A `dgemm_`/`zgemm_` call routed to the GPU copies
  its operands into GPU buffers the library reuses, or reads memory
  registered with `ab_register_host_memory` in place. The GPU converts
  them to double-float, scales the lines the exponent-range guard scales,
  multiplies, converts back and applies alpha and beta, with the host
  path's bits.
- **Data outside FP32's exponent range** is scaled by powers of two, or
  sent to Accelerate, instead of giving silently wrong GPU results.
- **ZGEMM with imbalanced real and imaginary parts** runs as four real
  products; the BLAS ABI and the device-buffer API decide per row and
  column.
- **Kernels:** 32 × 32 tiles for small outputs, float4 tile loads, register
  prefetch on small grids, Gauss-3M's three products run concurrently, and
  the native ZGEMMs keep their temporaries between calls. Every result
  keeps its bits.
- **Accounting:** a per-call trace (`AB_TRACE`), the `staged` line of
  `AB_REPORT`, and a replay of a traced Quantum ESPRESSO run.

On Quantum ESPRESSO si64 on an M2 Max, forcing the GPU now takes
1.01–1.06× the time on Accelerate, against 1.06–1.43× for 1.4.0 on the same
QE build, with the same energies
([release check](../benchmarks/results/2026-10-28-415cb5d-qe-si64-release-check/),
[staged-path fix](../benchmarks/results/2026-10-29-351d56c-staged-refusals/)).
The routing table is still 1.4.0's, so `AB_MODE=auto` keeps those calls on
Accelerate.

## Next

- **Calibrate the staged path.** `make calibrate` on 1.5.0 measures it;
  the built-in routing table still comes from 1.4.0's host path.
- **Split one call between Accelerate and the GPU inside the library**,
  planned for 1.6.0. `bench_split` does it in user code
  ([split benchmark](../benchmarks/results/2026-10-28-415cb5d-hybrid-split/));
  `dgemm_` and `zgemm_` do not split calls in 1.5.0.
- **Split-K** for long-K, small-output shapes, so that more than a few
  dozen threadgroups run (the `kernel` candidate below).

## 1.4 — correct and measured

The BLAS ABI behaves like reference BLAS, `AB_MODE=auto` follows measured
end-to-end timings instead of fixed FLOP thresholds, `AB_REPORT=1` shows
where the BLAS time went, and the documentation states only what was
measured. On an M2 Max that routing keeps almost every BLAS-ABI call on
Accelerate, which is the honest default today.

### What 1.4 planned for the shapes real codes send through the BLAS ABI

*Written for 1.4 and kept as written. 1.5's staged path took up the `pack`
and `upload` stages: it packs and converts on the GPU and reads registered
operands in place. The `kernel` candidate, split-K, is still open.*

Quantum ESPRESSO's Davidson ZGEMMs have a long K and a small output
(e.g. 320 × 320 × 18,000). Through the BLAS ABI they are 0.14–0.76× as
fast on the GPU as on Accelerate. Where that time goes, from `AB_REPORT`
on the si64 validation inputs ([VAL-1](../tests/validation/VAL001_QE_Si64.md)):
the GPU kernel alone takes 1.00× (input A, ZGEMM), 1.22× (B, ZGEMM) and
1.67× (C, all DGEMMs: Davidson and PAW) as long as Accelerate needs for the
same calls, and the other stages (packing and converting operands,
transfers, per-call allocation, the epilogue) add another 1.6–2.1× the
kernel time.

Performance work on these shapes starts only if the GPU kernel time alone
is below 0.8× of Accelerate's time for the same calls. Above that, even a
GPU path with no other cost would be at most 1.25× faster, a margin that
per-call conversion and transfers would not leave.
On the M2 Max that ratio is 1.00–1.72 with one rank (VAL-1 and the
[1.4.0 release check](../benchmarks/results/2026-10-15-c5a1fa6-qe-si64-opt/)), so
for single-process runs this work is **not planned for M2-class GPUs**.
With 8 ranks it is 0.65–0.78 for A's and B's ZGEMMs and C's DGEMMs, below
the bar: each rank's Accelerate runs one thread and the ranks share the
matrix hardware. With 8 ranks the other stages, at 0.7–2.1× the kernel
time, still make `gpu` mode 5–21% slower. Whether to work on that case is a
decision for 1.5; the candidates below would be measured there first. The candidates stay open for GPUs on which `make
calibrate` and an `AB_REPORT` show the kernel clearly ahead of Accelerate.

Candidates, each aimed at a measured stage:

- `kernel`: split-K for long-K, small-output shapes, so that more than a
  few dozen threadgroups run.
- `pack`: skip the repack of transposed operands where swapping the
  operands gives the same product; convert in the copy that packs.
- `upload`: keep device copies of operands that repeat across calls.

## Planned

- `AB_VERIFY`: optional randomized check (Freivalds) of GPU results.
- Thread-safe contexts for concurrent callers.
- Calibration tables for more chips (M1, M3, M4 families). `make calibrate`
  results from other machines are welcome as pull requests.
- Homebrew formula.

## Later

- LU factorization (DGETRF/DGETRS) built on the mixed-precision solver.
- True asynchronous ZGEMM with overlapping command buffers.
