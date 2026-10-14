# apple-bottom Roadmap

See [CHANGELOG.md](../CHANGELOG.md) for what has shipped.

## 1.4 — correct and measured (this release)

The BLAS ABI behaves like reference BLAS, `AB_MODE=auto` follows measured
end-to-end timings instead of fixed FLOP thresholds, `AB_REPORT=1` shows
where the BLAS time went, and the documentation states only what was
measured. On an M2 Max that routing keeps almost every BLAS-ABI call on
Accelerate, which is the honest default today.

## Next — the shapes real codes send through the BLAS ABI

Quantum ESPRESSO's Davidson ZGEMMs have a long K and a small output
(e.g. 320 × 320 × 18,000). Through the BLAS ABI they are 0.14–0.76× as
fast on the GPU as on Accelerate. Where that time goes, from `AB_REPORT`
on the si64 validation inputs ([VAL-1](../tests/validation/VAL001_QE_Si64.md)):
the GPU kernel alone takes 0.95× (input A), 1.08× (B) and 1.57× (C, the
real Davidson path) as long as Accelerate needs for the same routine, and
the other stages (packing and converting operands, transfers, per-call
allocation, the epilogue) add another 1.6–2.1× the kernel time.

Performance work on these shapes starts only if the GPU kernel time alone
is below 0.8× of Accelerate's time for the same calls. Above that, even a
GPU path with no other cost would be at most 1.25× faster, a margin that
per-call conversion and transfers would not leave.
On the M2 Max that ratio is 0.95–1.57, so this work is **not planned for
M2-class GPUs**. The candidates stay open for GPUs on which `make
calibrate` and an `AB_REPORT` show the kernel clearly ahead of Accelerate.

Candidates, each aimed at a measured stage:

- `kernel`: split-K for long-K, small-output shapes, so that more than a
  few dozen threadgroups run.
- `pack`: skip the repack of transposed operands where swapping the
  operands gives the same product; convert in the copy that packs.
- `upload`: keep device copies of operands that repeat across calls.
- `other`: reuse scratch buffers instead of allocating ZGEMM's four
  temporaries on every call.

## Planned

- `AB_VERIFY`: optional randomized check (Freivalds) of GPU results.
- Thread-safe contexts for concurrent callers.
- Calibration tables for more chips (M1, M3, M4 families). `make calibrate`
  results from other machines are welcome as pull requests.
- Homebrew formula.

## Later

- LU factorization (DGETRF/DGETRS) built on the mixed-precision solver.
- True asynchronous ZGEMM with overlapping command buffers.
