# REVIEW_04 Tranche A — end-to-end rect sweep (2026-04-24)

## Methodology callout — steady-state vs end-to-end

apple-bottom has two distinct BLAS entry points, and the performance regime
differs sharply between them:

| Entry point | What it times | Where it is used |
|---|---|---|
| `ab_dgemm(mA, mB, mC)` / `ab_zgemm(...)` — takes pre-allocated `ABMatrix` handles that have already been uploaded. | **Steady-state compute.** Metal kernel invocation only; no allocation, no FP64↔DD conversion, no col↔row repack, no transfer. | Internal paths where the caller manages handle lifetime. `bench_dgemm.c` / `bench_zgemm.c` measure this regime. |
| `ab_dgemm_blas(transA, transB, ..., double* A, int ldA, ...)` / `ab_zgemm_blas(...)` — takes raw column-major pointers. | **End-to-end per-call.** Allocation of 3 (DGEMM) or 6 (ZGEMM) `ABMatrix` objects, col→row repack, FP64→DD upload, kernel dispatch, download, col-major pack-back, destroy. **Everything on every call.** | The BLAS ABI path. Fortran host codes (QE, Yambo, …) reach us here via `dgemm_` / `zgemm_` — they hand us raw pointers; we have nowhere to cache GPU-side handles. |

This commit characterizes the **end-to-end** regime via the
`ab_*_blas` path. It is the regime Fortran host codes actually
experience when linked against apple-bottom. Steady-state numbers
are preserved in `benchmarks/results/2026-04-22-b9b0641/{dgemm,zgemm}.csv`
and pulled in below for comparison.

## Sweep provenance

- **Commit:** `271a0f0` — `bench: add rect bench + dispatch-path hook for REVIEW_04`
- **Runner:** `benchmarks/scripts/run_rect_sweep.sh`
- **Config files:** `benchmarks/configs/rect_{dgemm,zgemm}.txt`
- **Raw CSVs (12 files):** `benchmarks/results/2026-04-24-271a0f0-rect/`
  - `{dgemm,zgemm}_{cpu,gpu,auto}_sweep{1,2}.csv`
  - DGEMM: 25 shapes × 5 runs × 2 sweeps = 250 rows/CSV-pair
  - ZGEMM: 38 shapes × 5 runs × 2 sweeps = 380 rows/CSV-pair
- **CSV schema (11 columns):**
  `timestamp, mode, tag, M, N, K, run_idx, gflops, wall_s, frob_rel_err, dispatch_hint`
- **Verify regime:** every row reports Frobenius relative error against
  `cblas_{d,z}gemm`. Guard is dispatch-aware: CPU rows must show
  `frob = 0.0` exactly (bit-identical to cblas by construction); GPU
  rows must land in `(1e-18, 1e-12]` (DD-FP32×2 vs FP64 reference).
  `frob < 1e-18` on a GPU row would flag an aliasing/verify bug.
- **gpu stanza override:** the `AB_MODE=gpu` sweep prepends
  `AB_MIN_GPU_DIM=0` to force-GPU sub-32 shapes (pathology characterization).
  The `cpu` and `auto` stanzas preserve the default min_dim=32 floor.

## Verify outcomes — 1890 rows, 0 failures

| Class | Count | Meaning |
|---|---|---|
| CPU_BITID | 780 | cpu path, `frob = 0.0` exactly (cblas vs cblas, bit-identical — expected) |
| GPU_OK | 1110 | gpu path, `frob ∈ (1e-18, 1e-12)` — honest DD precision |
| GPU_TOO_CLOSE_BUG | 0 | — |
| GPU_FAIL_PRECISION | 0 | — |

Verify and dispatch-hint routing both clean across the full grid.

## Headline — end-to-end GFLOP/s (`ab_*_blas` path)

Mean across 10 samples (2 sweeps × 5 runs) per (mode, shape) cell.
**All numbers below are the end-to-end BLAS-ABI path, not steady-state.**

### DGEMM synthetic squares

| N | cpu GFLOP/s | gpu GFLOP/s | auto GFLOP/s | auto routes | gpu slowdown vs cpu |
|---|---:|---:|---:|---|---:|
| 128 | 262 | 1.6 | 281 | cpu | 166× |
| 256 | 358 | 3.9 | 356 | cpu | 92× |
| 512 | 649 | 5.2 | 5.2 | gpu | 124× |
| 1024 | 649 | 10.5 | 10.4 | gpu | 62× |
| 2048 | 580 | 20.2 | 20.1 | gpu | 29× |

### ZGEMM synthetic squares

| N | cpu GFLOP/s | gpu GFLOP/s | auto GFLOP/s | auto routes | gpu slowdown vs cpu |
|---|---:|---:|---:|---|---:|
| 128 | 276 | 5.8 | 276 | cpu | 48× |
| 256 | 320 | 6.5 | 6.9 | gpu | 49× |
| 512 | 704 | 11.5 | 12.9 | gpu | 61× |
| 1024 | 623 | 20.8 | 21.7 | gpu | 30× |
| 2048 | 604 | 39.8 | 39.7 | gpu | 15× |

**In the entire tested grid, GPU never wins via `ab_*_blas`.** The
dispatcher's auto-mode routes GPU for every shape with
`FLOPs ≥ threshold ∧ min(M,N,K) > 32`, and in exactly those shapes
GPU loses 15–124× to cpu (worst at DGEMM N=512, best at ZGEMM N=2048).
The auto-mode slowdown is monotone-improving with N (because
compute starts to amortize the ~1 s of unexplained overhead), but
it never crosses into a speedup.

### QE Davidson shapes (ZGEMM) — end-to-end

From committed QE benchmark outputs (`benchmarks/qe_yambo/`). Davidson
eigensolver fires these shapes as matched overlap/rotation pairs per
SCF iteration.

| Tag | (M, N, K) | cpu | gpu | auto | auto routes |
|---|---|---:|---:|---:|---|
| qe_si8_ovlp | (24, 24, 2969) | 212 | 0.9 | 209 | cpu (min_dim) |
| qe_si8_rot | (2969, 24, 24) | 143 | 1.2 | 148 | cpu (min_dim) |
| qe_si32_ovlp | (80, 80, 4573) | 573 | 2.9 | 2.8 | gpu (**wrong**) |
| qe_si32_rot | (4573, 80, 80) | 423 | 3.1 | 3.0 | gpu (**wrong**) |
| qe_si64_ovlp | (150, 150, 18277) | 432 | 5.1 | 5.0 | gpu (**wrong**) |
| qe_si64_rot | (18277, 150, 150) | 584 | 4.8 | 4.9 | gpu (**wrong**) |
| qe_si64_500b_ovlp | (500, 500, 18277) | 605 | 16.3 | 16.8 | gpu (**wrong**) |
| qe_si64_500b_rot | (18277, 500, 500) | 714 | 15.1 | 15.6 | gpu (**wrong**) |

### Yambo GW shapes (ZGEMM) — end-to-end

| Tag | (M, N, K) | FLOPs (z) | cpu | gpu | auto | auto routes |
|---|---|---:|---:|---:|---:|---|
| yambo_chi0_200 | (200, 200, 189) | 60.5 MF | 321 | 7.3 | 322 | cpu (under threshold) ✓ |
| yambo_chi0_300 | (300, 300, 189) | 136 MF | 330 | 8.4 | 8.3 | gpu (**wrong**) |
| yambo_chi0_500 | (500, 500, 189) | 378 MF | 636 | 7.0 | 7.3 | gpu (**wrong**) |
| yambo_chi0_inv_proxy | (200, 200, 200) | 64.0 MF | 328 | 7.2 | 304 | cpu (under threshold) ✓ |
| yambo_sigma_x | (25, 25, 19195) | 96.0 MF | 161 | 0.6 | 141 | cpu (min_dim) ✓ |

## Anchor-shape side-by-side: end-to-end vs steady-state

Steady-state numbers from `benchmarks/results/2026-04-22-b9b0641/{dgemm,zgemm}.csv`
(same hardware, same `libapplebottom.dylib` at the ancestor commit).

| Shape | cpu e2e | gpu e2e | cpu ss | gpu ss | gpu ratio (ss/e2e) |
|---|---:|---:|---:|---:|---:|
| DGEMM N=1024 | 649 | 10.5 | 553 | 514 | **49×** |
| DGEMM N=2048 | 580 | 20.2 | 563 | 654 | **32×** |
| ZGEMM N=1024 | 623 | 20.8 | 584 | 664 | **32×** |
| ZGEMM N=2048 | 604 | 39.8 | 610 | 813 | **20×** |

**CPU numbers are the same across regimes** (cpu path is just `cblas`,
no allocation/transfer distinction). The 20–49× GPU collapse
between steady-state and end-to-end is the entire R4-2 story.

## R4-2 reclassification

R4-2 splits into two findings with different scope and different fixes.

### R4-2a — dispatcher threshold tuning (end-to-end path). EMPIRICALLY GROUNDED.

The auto-mode FLOP thresholds (`DEFAULT_CROSSOVER_FLOPS_REAL = 50M`,
`DEFAULT_CROSSOVER_FLOPS = 100M`) assume the steady-state crossover.
For host codes reaching us through the BLAS ABI — **all Fortran
integrations, by construction** — the crossover does not exist within
the measured grid. Every auto-mode GPU-routed shape loses 15–200× to
cpu (15× at ZGEMM 2048³, 200× at the QE si32 Davidson shapes).

Concrete recommendation, empirically supported:
- **For the `ab_*_blas` end-to-end path, raise or disable FLOP thresholds** such
  that auto-mode routes cpu for all shapes up to at least N=2048.
  Practically this means `AB_CROSSOVER_FLOPS ≥ 100G` and
  `AB_CROSSOVER_FLOPS_REAL ≥ 100G` (effectively disabling GPU dispatch
  for Fortran host codes at any size represented in the grid).
- **For the direct `ab_dgemm`/`ab_zgemm` path**, the existing thresholds
  remain correct. Callers who manage their own `ABMatrix` lifetime
  still see the steady-state regime.
- The "right" answer is probably that `ab_*_blas` should refuse to
  dispatch GPU regardless of threshold, **until R4-2b is understood**.
  The current dispatcher assumes the two regimes share a crossover; they
  do not.

### R4-2b — unexplained end-to-end overhead. KNOWN-UNKNOWN.

The overhead delta between end-to-end and steady-state is larger than
allocation + transfer + repack + destroy can explain.

Back-of-envelope at ZGEMM 2048³:
- End-to-end wall: 68.7 GF / 40 GFLOP/s = **1.72 s per call**.
- Steady-state kernel: 68.7 GF / 813 GFLOP/s = **84 ms per call**.
- Overhead to explain: **≈ 1.64 s per call.**
- Upload budget: 3 matrices × 2048² × 16 B (FP64-packed-as-DD) ≈ 200 MB.
  At ~40 GB/s unified-memory bandwidth: **5–15 ms total**.
- Download budget (2048² × 16 B = 64 MB): **~2–5 ms**.
- Col↔row repack on CPU for 3 matrices of 2048² elements each:
  **~60–200 ms** depending on cache behavior (pure scalar loops,
  not vectorized).
- Alloc / free of 6 Metal buffers: **~50 ms**.
- **Total explainable overhead: ~250 ms at most.**
- **Remaining unexplained: ~1.4 s.**

Canary at ZGEMM 1024³ (`/tmp/canary_rect.c`, AB_MODE=gpu, one call,
reset_stats around it):

| Stat | Value |
|---|---|
| dispatch_hint | gpu |
| zgemm_count Δ | 1 (Metal path fired) |
| kernel_time_ms | 148.5 |
| upload_time_ms | 241.6 |
| wall (derived from 8.59 GF / 20.83 GFLOP/s) | ~412 ms |
| frob_rel_err | 3.60 × 10⁻¹⁴ |

So the **GPU kernel itself takes ~148 ms end-to-end at N=1024**, versus
~13 ms at steady-state (8.59 GF / 663 GFLOP/s). That is an
**~11× slowdown of the kernel alone**, on what should be the identical
Metal invocation. Transfer and allocation do not explain this — the
kernel is taking longer even when transfer is counted separately.

Hypotheses, ordered by how testable they are:

1. **Multiple kernel launches per `ab_*_blas` call.** The FP64→DD
   conversion may execute as its own Metal kernel (or as CPU-side
   scalar code running serially with the compute). The Gauss
   3-multiply ZGEMM decomposition fires 3 DGEMM-like kernels + 3
   epilogue kernels. Each launch has command-buffer-build overhead
   that the `kernel_time` stats accumulator may sum opaquely.
2. **FP64 → DD conversion on CPU, not GPU.** If format conversion
   happens in the upload path on CPU before the Metal upload, it
   counts against `upload_time_ms` and inflates that stat, while
   leaving kernel time comparable to steady-state. Counter-indicator:
   kernel time IS 11× inflated, so this alone doesn't fit.
3. **Per-call blocking device sync prevents pipelining.** Each
   `ab_*_blas` call ends with a download → `waitUntilCompleted` on
   the command buffer. The next call's command buffer has to be built
   from cold, GPU goes idle between, boost clock decays.
4. **Kernel-cold boost-clock issue.** The M2 Max GPU takes several
   hundred ms to ramp to sustained boost (see
   `docs/vv/VV_REPORT.md` §10, 7/240 samples ≥50% boost residency
   during `bench_paper`). Short kernels launched from idle never get
   the clock. Steady-state benchmarks run a tight loop that warms the
   clock; end-to-end calls do not.
5. **ARC / Objective-C object-lifetime costs.** `ABMatrix` create/destroy
   likely involves Metal command-queue state, autorelease pools, and
   reference-counting traffic that the steady-state benchmark amortizes
   to zero.

Concrete next step (investigation commit, not part of this tranche):
**instrument `ab_{d,z}gemm_blas` with per-stage timing** — alloc,
repack, upload, kernel dispatch, download, pack-back, destroy —
emitted to stderr under `AB_PROFILE=1`. Run on a single ZGEMM 2048³
call. Sum of stage times compared to wall clock identifies which
stage holds the unexplained 1.4 s. Only then should R4-2b be fixed.

## Out of scope / follow-up

- **Commit #2 (skill rectification).** Stays parked. The "GPU wins for
  N ≥ 1024 / 2048" claims in the `apple-bottom-qe` skill (and in
  `include/apple_bottom.h` L17-20) need both regimes labelled.
  Empirical recommendation for the skill edit: for host codes that
  reach us via BLAS ABI (QE, Yambo), GPU does not win in the measured
  grid — the "force GPU" case is a benchmark-only configuration.
  Skill edit deferred until R4-2b is at least hypothesis-tested.
- **Rainbow Connection manuscript audit.** Parked as separate todo:
  grep all GFLOP/s numbers in the manuscript, flag any that don't
  explicitly disclose which regime (steady-state persistent-handle
  vs end-to-end BLAS-ABI) they characterize. Not inside REVIEW_04
  scope but flagged while the data is fresh.
- **bench_rect_steady variant.** Option (2) from the classification
  round. A matched-shape steady-state bench (pre-allocated,
  pre-uploaded handles) over the same 38+25 shapes would let R4-2a
  and R4-2b be quantified on identical shapes rather than cross-bench.
  Not done in this tranche; commit deliberately narrow.

## Reproducibility

```bash
# Hardware: M2 Max, 38-core GPU, 96 GB unified memory
# OS: macOS 26.2, Xcode 26.0.1
# Commit: 271a0f0

cd ~/Dev/Claude/apple-bottom
make clean && make bench-rect
./benchmarks/scripts/run_rect_sweep.sh
# Outputs to benchmarks/results/<date>-<sha>-rect/
```

Wall time for the 12-CSV sweep: ~8 minutes on an idle system.
Longest stanza is `zgemm_gpu_sweep{1,2}` (N=2048 ZGEMM at ~1.7 s/call
× 7 runs × 2 sweeps ≈ 24 s per shape, × 38 shapes ≈ 15 min aggregate
when the 2048 shapes dominate).
