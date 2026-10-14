# Changelog

All notable changes to this project will be documented in this file.

## [1.3.1-dev] - Unreleased

_TBD following Si64 benchmark campaign._

### Added
- **`ab_build_info()`** returns the library version and the git commit it
  was built from, with `-dirty` if tracked files differed from it (the
  Makefile passes `AB_BUILD_SHA`; `unknown` outside a git checkout). Every
  `AB_PROFILE` and `AB_REPORT` report starts with it.
- **`AB_REPORT=1`**: a short summary on stderr at exit, once per process
  and tagged with the pid. Per routine: calls and wall time by where C was
  computed, the GPU calls' stage costs (init, alloc, pack, upload, kernel as
  command-buffer GPU time, download, epilogue, other) and fallbacks to cblas
  by reason (`no_device`, `too_large`, `gpu_failed`). See the environment
  variable table in `docs/INTEGRATION.md`.

### Fixed
- **Morton remap dropped output tiles on non-square power-of-two grids**
  (K-P0-1). The remap is now applied only when it is a bijection
  (gridW == gridH or gridW == 2·gridH). Before, shapes such as 512×256,
  1024×256 and tall-skinny 1024×64 / 2048×128 returned AB_OK with tiles of C
  unwritten, which also corrupted ZGEMM, the BLAS wrappers, MPIR and DTRSM.
- **`ab_dgemm_async` left kernel buffers 6/7 unbound** (K-P1-3), which
  aborts under Metal API validation. It now shares the dispatch helper with
  `ab_dgemm`, so the two give bitwise-identical results.
- **Fortran `CALL DGEMM` / `CALL ZGEMM` never reached apple-bottom**
  (H-P0-1). Under `ACCELERATE_NEW_LAPACK` the `dgemm_`/`zgemm_` definitions
  were emitted as `_dgemm$NEWLAPACK`/`_zgemm$NEWLAPACK`, so Fortran hosts
  linked with `-lapplebottom` (dylib or static) bound to Accelerate. Both
  libraries now export exactly `_dgemm_` and `_zgemm_` (see Changed for the
  `$NEWLAPACK` names). `make verify-symbols` checks the exact names (it
  accepted `_ab_dgemm_` before), and `make test-fortran` runs a gfortran
  probe.
- **BLAS-ABI calls ran on cblas unless the host called `ab_init()`**
  (H-P1-2). `ab_dgemm_blas`/`ab_zgemm_blas`, and through them
  `dgemm_`/`zgemm_`, now create the Metal context on first GPU use, so
  hosts such as QE reach the GPU without an init call.
- **Dispatch path and AB_PROFILE reported the routing decision, not the path
  taken** (H-P2-8). `ab_get_last_dispatch_path()` says `"gpu"` only
  once the GPU result is in C and `"cpu"` for every cblas run, including
  calls the Fortran bridge keeps on the CPU; `ab_use_gpu()` no longer sets
  it. The AB_PROFILE table counts a `dgemm_`/`zgemm_` call as GPU only if it
  ran there, and a GPU failure that falls back to cblas prints its reason
  under AB_PROFILE.
- **Lowercase TRANS flags and `'C'` for real data were misread** (K-P0-3,
  H-P1-1). `dgemm_`, `zgemm_`, `ab_dgemm_blas`, `ab_zgemm_blas` and the
  `ab_*gemm_` wrappers compared TRANS with upper-case letters only, and the
  CPU and GPU paths disagreed: `'n'` could run as a transpose or conjugate
  transpose, `'t'` as no transpose or a conjugate transpose, and DGEMM `'C'`
  as no transpose. TRANS is now case-insensitive as in reference BLAS,
  `'C'` means `'T'` for real data, and any other character is rejected by
  the cblas parameter check.
- **Rectangular `'T'`/`'C'` ZGEMM calls returned `beta*C` or read out of
  bounds** (H-P0-2). `ab_zgemm_blas` and `zgemm_` uploaded the transposed
  operands in their stored shape, so the GPU product failed or overran A
  and B; QE's `'C','N'` call in `cegterg` could crash. op(A) and op(B) are
  now packed as M×K and K×N.
- **A failed GPU step in a BLAS-ABI call could leave C wrong** (H-P1-3).
  Allocations and status results on the GPU path were not checked. Every
  step is now checked, and a failure before C is written reruns the whole
  call on cblas. `ab_dgemm`, `ab_dgemm_scaled` and the `ab_zgemm_ex`
  transposes now return `AB_ERROR_KERNEL_FAILED` when the command buffer
  fails.
- **BLAS-ABI calls with `alpha == 0` read A and B** (H-P1-7), so NaN or Inf
  in A or B reached C. They now take the reference BLAS quick return,
  C = beta·C on cblas, without reading A or B; with `beta == 0`, C is never
  read. Results for `alpha == 0` can differ in the last bit from the
  former GPU result.
- **Illegal BLAS arguments could read and write out of bounds on the GPU
  path.** Only the cblas path checked leading dimensions, so an lda, ldb or
  ldc below the operand's row count made `dgemm_`, `zgemm_`,
  `ab_*gemm_blas` and `ab_*gemm_` index past the caller's arrays; with lazy
  init this became the default for hosts such as QE. Calls with an invalid
  TRANS, a negative M, N or K, or a short leading dimension now always go
  to cblas, which names the bad parameter and stops the program as
  reference BLAS (`xerbla`) does, whatever `AB_MODE` says.
- **A `fork()`ed child crashed on its first GPU-routed BLAS call.** Metal
  does not survive `fork()`, and since lazy init every BLAS-ABI host that
  forks after a GPU call was exposed. The first `ab_init()` now registers a
  fork handler. In the child, `ab_init()` returns `AB_ERROR_NO_DEVICE`,
  `ab_is_initialized()` is false, native GEMMs return `AB_ERROR_NO_DEVICE`,
  and GPU-routed BLAS calls run on cblas (dispatch path `"cpu"`).
- **Parallel `ab_matrix_upload`/`ab_matrix_download` ran at ~125 MB/s**
  (H-P0-3). The FP64↔DD conversion used one GCD block per element, which
  slowed every BLAS-ABI call, DTRSM, MPIR and device-API transfer. It now
  converts in a few contiguous chunks, with bit-identical results. On an
  M2 Max at N=2048: upload 122 MB/s → 72 GB/s, download 123 MB/s → 59 GB/s,
  and an `ab_dgemm_blas` `'N','N'` call 862 ms → 94 ms.
- **A memory-pool matrix reused for a smaller shape kept its old element
  count** (K-P1-4), so `ab_matrix_upload`/`ab_matrix_download` read or
  wrote past the caller's buffer. Reuse now resets the count; the buffer
  capacity is tracked separately.
- **GPU calls leaked about 170 bytes each in programs without an
  autorelease pool**, such as Fortran and C hosts. Every entry point that
  creates Metal command buffers or encoders now drains its own pool, so
  memory stays flat.
- **Shaders lost DD precision before macOS 15** (K-P1-6). Without
  `MTLMathModeSafe` the library compiled them with fast math, which gives
  FP32-level errors. It now sets `fastMathEnabled = NO` there, which keeps
  full DD precision, and keeps `MTLMathModeSafe` on macOS 15 and later. The
  choice is made with a run-time selector check, so a static archive built
  for an older deployment target still links with gfortran (no compiler-rt
  `___isPlatformVersionAtLeast`).
- **`ab_dev_dgemm` / `ab_dev_zgemm` ignored their column-major contract**
  (K-P0-2, H-P0-4). Column-major buffers were read as row-major, so square
  inputs returned B·A, transpose flags were ignored or returned -3, and other
  shapes were wrong. Both now compute Cᵀ = op(B)ᵀ·op(A)ᵀ on the transposed
  operands and honour transA/transB, ld and offsets. Invalid flags, short
  leading dimensions and operands that run past their buffer return
  `AB_ERROR_INVALID_ARG`. `test_device_api` compared against the row-major
  API, which enforced the bug; it now checks against column-major cblas.
- **The Python bindings did not work** (H-P1-5). `ab.init()` and every
  status check raised `TypeError`, and matrices from
  `MemoryPool.get_matrix()` or `Session.get()` were freed twice, which
  crashed. Every handle is now freed exactly once. Borrowed matrices keep
  their pool or session alive and raise `InvalidArgumentError` once it is
  closed, or once their pool is `reset()`: the pool may hand the handle out
  again with another shape, and a stale wrapper's `download()` would
  overflow its buffer. numpy is optional, `AB_LIBRARY` selects the dylib,
  and `make test-python` runs the binding tests. The bindings remain
  experimental.
- **Matrices with more than 46,340 rows or columns never reached the GPU.**
  The limit was 46,340 per dimension, a square-matrix stand-in for 2³¹
  elements. In Quantum ESPRESSO Si64 (PAW, 2 MPI ranks) it rejected only
  the PAW DGEMMs in `newq` (called from `newd`) and `addusdens`,
  171 × 64 × 2·ngm_l and 2·ngm_l × 171 × 64: 171 = nh(nh+1)/2 projector
  pairs (nh = 18 per Si atom), 64 = the atoms of that species, and
  2·ngm_l = 71,834 (gamma point) or 143,666 (k-points) = the rank's
  dense-grid G-vectors, complex values passed as real pairs. Those
  BLAS-ABI calls always ran on cblas; the Davidson GEMMs (`regterg`,
  `cegterg`; long dimension ≈ 9,000 per rank) were never affected. This
  is a robustness fix, not a speed-up: the PAW DGEMMs are memory-bound
  and slower on the GPU, and calibrated `AB_MODE=auto` routing keeps them
  on Accelerate. The limit is now what actually overflows, for any shape:
  rows × cols ≤ `AB_MAX_MATRIX_ELEMENTS` (2³¹ − 1; the kernels index in
  32-bit unsigned arithmetic) and the DD buffer within the device's
  `maxBufferLength`. Every kernel was checked bit-exact on matrices past
  2²⁹ elements (byte offsets above 4 GiB).
  Beyond the limit, `ab_matrix_create` returns NULL; the new
  `ab_matrix_check_size()`, `ab_session_add` and `ab_dev_*gemm` return
  the new `AB_ERROR_TOO_LARGE` (`ab_dev_*gemm` returned
  `AB_ERROR_ALLOC_FAILED`); BLAS-ABI calls run on cblas, and `AB_PROFILE=1`
  prints why. In Python, `MAX_ELEMENTS` (2³¹ − 1) bounds rows × cols and
  `MAX_DIMENSION` is now 2³¹ − 1.

### Changed
- **The AB_PROFILE table separates call share from time share.** Its
  `gpu_pct` column was the share of calls that ran on the GPU, not of time,
  and was read as a time share. It is replaced by `gpu_calls`, `gpu_calls%`,
  `gpu_ms`, `cpu_ms`, `gpu_time%` and `fallbacks`, followed by tables of the
  GPU stage costs and of the fallbacks by reason. Every GPU call now prints
  a stage line, DGEMM included (it was ZGEMM only), with the stages named as
  in `AB_REPORT`. Direct `ab_dgemm_blas` / `ab_zgemm_blas` calls are now
  counted too (only `dgemm_`, `zgemm_` and `ab_*gemm_` were). A `fork()`ed
  child starts with empty counters, so it no longer reports the parent's
  calls as its own. In the `AB_PROFILE_JSON` file, `gpu_dispatch_count` and
  `amx_fallback_count` are now `gpu_calls` and `cpu_calls`, next to
  `gpu_ns`, `cpu_ns`, `gpu_stage_ns`, `fallbacks` and the top-level `build`.
- **`_dgemm$NEWLAPACK` / `_zgemm$NEWLAPACK` are no longer exported.** C code
  that calls `dgemm_`/`zgemm_` through the `ACCELERATE_NEW_LAPACK` headers
  now gets Accelerate, not apple-bottom; call `ab_dgemm_blas` /
  `ab_zgemm_blas` to use the GPU. Binaries already linked against the old
  dylib still resolve those names through the Accelerate re-export.
- **`ab_dev_dgemm` / `ab_dev_zgemm` now read column-major buffers as
  documented** (see Fixed). Code that swapped or transposed operands to work
  around the old row-major reading gets different results and must drop
  that workaround.
- **The precision tests fail when DD precision is lost** (K-P1-5).
  `test_precision` measures normwise and componentwise error against a
  compensated FP64 reference and also covers the scaled, tall-skinny and
  ZGEMM kernels (8 → 14 cases). The V-2 convergence study fails when the
  Frobenius error exceeds √N·u². A build without safe math now fails both.
- **`make ci-local` runs `test_device_api` and counts tests one way**
  (RH-P1-1). `TEST_TOTAL` counts tests (precision cases, `TESTS[]` entries
  and BLAS-ABI suites: 146), with device-API `CHECK`s and BLAS-ABI cases on
  their own lines. `test_convergence` takes its CSV path as `argv[1]`, so
  `make ci-local BUILD=...` reports the convergence data instead of
  `missing`.

## [1.3.0] - 2026-04-18

### Added (Phase 6)
- **Fused ZGEMM** — single command buffer with 4 encoders (was 8 separate round-trips)
  - Encoder boundaries enforce GPU barriers for the Gauss-trick dependency DAG
  - Eliminates ~350μs of per-call command buffer commit+wait overhead
  - Same precision (DD ~10⁻¹⁵) with lower dispatch latency
- **True async ZGEMM** (`ab_zgemm_async`) — non-blocking GPU execution
  - Metal completion handler cleans up temporaries when GPU finishes
  - Returns `ABFuture` that can be polled or waited on
  - Replaces previous synchronous wrapper implementation
- **Batched GEMM API** (`ABBatch`) — amortizes Metal overhead across many GEMMs
  - `ab_batch_create/dgemm/dgemm_scaled/zgemm/barrier/commit/wait/destroy`
  - Single command buffer for hundreds of GEMMs (QE fires 100s per SCF iteration)
  - `ab_batch_barrier()` inserts encoder boundary for dependent operations
  - Batched ZGEMM uses same fused 4-encoder pattern within the batch
  - 8 new correctness tests (batch DGEMM, multi-DGEMM, barrier, null safety,
    batch ZGEMM, committed reuse, fused ZGEMM precision, true async ZGEMM)
- **Performance regression CI** (`scripts/perf_regression.sh`)
  - Automated benchmark suite with baseline comparison
  - Configurable regression threshold (default 15%)
  - JSON baseline storage, markdown regression report
  - `--save` to capture baseline, `--ci` for non-zero exit on regression
  - Benchmarks: DGEMM (square + tall-skinny), ZGEMM, batched DGEMM

### Added (Tier 2)
- **Mixed-Precision Iterative Refinement** (`ab_dgesv_mpir`) — solves A*X=B via MPIR
  - FP32 LU factorization (Accelerate LAPACK sgetrf/sgetrs) for O(N³) bulk work
  - DD-DGEMM residual computation (10⁻¹⁵ fidelity) catches cancellation errors
  - Converges to DD precision in 1-3 iterations for well-conditioned matrices
  - Avoids explicit inversion (κ² error) and bespoke DD-DTRSM shaders (Paper §6)
  - 6 new correctness tests (basic, single-RHS, moderate-κ, null, dim-mismatch, non-square)
- **Broadened tall-skinny DGEMM dispatch** for domain-science matrices
  - Relaxed heuristic from `(M > 4*N) && (N <= 64)` to pure aspect-ratio `(M >= 4*N)`
  - Catches QE hot-path shapes like M=18277, N=150 that previously fell through to
    64×64 square kernel (~34% boundary waste → ~4% with 128×16 tiles) (Paper §5)

### Added
- **DTRSM**: Triangular solve (op(A) * X = alpha * B) — blocked forward/back-substitution with DGEMM panel updates
  - New enums: `ABSide`, `ABUplo`, `ABDiag` for BLAS-standard calling convention
  - GPU-efficient for N >= 1024
- **Python Bindings** (`python/` directory)
  - `ctypes` wrapper with Pythonic API: `ab.dgemm(A, B)` on numpy arrays
  - High-level classes: `Matrix`, `Session`, `MemoryPool`
  - Functional one-liner API: `dgemm()`, `zgemm()`, `dsyrk()`
  - Context managers and automatic cleanup
  - Full type hints (PEP 484 + PEP 561)
- **Benchmark Report Script** (`scripts/bench_report.sh`)
  - Automated benchmark suite with formatted markdown output
  - `make bench-report` target
  - System info collection, `--quick` mode
- **Competitor Comparison** (`docs/COMPARISON.md`)
  - Detailed technical comparison vs metal-float64, MLX, Accelerate, AppleNumericalComputing
  - Decision guide for when to use each approach
- **Blog Post Draft** (`docs/blog/fp64-on-apple-silicon.md`)
  - HN/Reddit-ready technical writeup
- **Dynamic Wilkinson-scaled error thresholds** in precision tests
  - Threshold: `C_safety × sqrt(K) × u_DD` instead of static `1e-14`
  - Mathematically correct per Higham (2002) §3.1 probabilistic analysis
  - Precision tests now include rectangular (K=4096) and tall-skinny shapes
  - Test count: 6 → 8 precision tests
- **GPU-native transpose DGEMM** in BLAS wrapper
  - `transA='T'` and `transB='T'` now route to GPU instead of CPU fallback
  - Upload-time reordering handles column-major transpose conversion
  - Improves GPU call ratio in QE workloads (was ~85%, now higher)
- **Block-wise compensated accumulation** in GEMM kernels
  - Periodic `twoSum` renormalization every 128 K-elements (8 tiles)
  - Final renormalization before epilogue alpha/beta scaling
  - Prevents accumulator drift for large-K dot products
  - Applied to all GEMM kernels: dd_dgemm, dd_dgemm_ab, dd_dgemm_ab_ts, dd_dsyrk
- **Morton Z-order threadgroup dispatch** for SLC cache locality
  - Space-filling curve remaps 2D threadgroup IDs to improve L2/SLC hit rate
  - Applied to all GPU kernels: dd_dgemm, dd_dgemm_ab, dd_dgemm_ab_ts, dd_dsyrk
  - Inline `morton_remap` function with 16-bit deinterleave + bounds checking
  - Fallback to identity mapping when coordinates exceed grid dimensions
- **DTRSM implementation** (blocked forward/back-substitution)
  - Left-side: blocked panel algorithm with NB=64
  - CPU (AMX) solves small triangular blocks, GPU DGEMM updates remaining panels
  - Supports upper/lower, no-trans/trans, unit/non-unit diagonal
  - Right-side falls back to cblas_dtrsm (AMX) for now
  - Alpha scaling support
  - 6 new correctness tests (lower, upper, trans, alpha, large N=256, null safety)
- **AMX heterogeneous dispatch** for small GEMMs
  - Dimension-aware heuristic: matrices with any dim ≤ 32 always route to AMX
  - Separate thresholds for real (50M FLOPs) vs complex (100M FLOPs) DGEMM
  - Skinny matrix penalty: higher threshold when any dimension < 64
  - Reduces GPU dispatch overhead for QE's many small ZGEMM calls

### Fixed
- **BUG-8: ARC leak in Metal-ref structs** — `ABMatrix_s`, `ABFuture_s`, `ABBatch_s`, and `ab_dev_buffer_s` now allocate via `new`/`delete` so ARC destructors release their `id<MTL…>` strong refs (was `calloc`/`free`, which does not run ARC destructors and leaked the buffers, command buffers, and encoders). Landed in ad993e1.

### Changed
- **README.md**: Complete rewrite as high-conversion landing page
  - One-line pitch above the fold with benchmark numbers
  - Performance comparison table (DD-GPU vs AMX-CPU)
  - Streamlined quickstart and API reference
  - Prior art comparison table
  - CI badge from GitHub Actions
- **Citation URLs**: Fixed TechnologyResidue → grantdh/apple-bottom in research/td-dgemm/README.md
- **Version**: Bumped to 1.3.0-dev

### Notes
- Test count: 50 → 70 (8 precision + 62 correctness)
- DTRSM right-side GPU acceleration planned for v1.4
- Rectangular matrix fix in progress (fix/rectangular-gemm branch)

## [1.2.0] - 2026-04-02

### Added
- **V&V Documentation Package** (NASA-STD-7009A compliant)
  - VV_REPORT.md: Master validation document with traceability matrix
  - PRECISION_ENVELOPE.md: Precision guarantees and validated envelope
  - VAL-1: Quantum ESPRESSO Si64 production validation
- **CI Pipeline** (GitHub Actions)
  - Automated compile-only checks on push/PR
  - Local validation script with convergence study
- **API Documentation**
  - Version alignment: 1.0.0 → 1.2.0 in all headers
  - API limits: AB_MAX_DIMENSION=46340, pool capacity 128, session capacity 64
  - Async ZGEMM implementation note (synchronous wrapper, true async planned for v2.0)
  - Precision scaling clarification: per-element ~10⁻¹⁵, DGEMM Frobenius ~N×10⁻¹⁵

### Fixed (Critical)
- **BUG-1/BUG-2**: Async DGEMM dimension packing and pipeline selection
- **BUG-3**: ab_dgemm_scaled alpha/beta truncation to float (precision loss)
- **BUG-4**: ab_matrix_scale alpha truncation to float (precision loss)
- **BUG-5**: dispatch_once_t reset undefined behavior (V&V audit flag)
- **BUG-6**: Memory pool overflow leak (returned unmanaged matrices)
- **BUG-7**: Missing pipeline creation error checks (crash on init failure)
- **ab_zherk**: Added error checking for ab_dgemm/ab_matrix_add/ab_matrix_sub with goto cleanup

### Changed
- **Integration Documentation**: Fixed INTEGRATION.md examples and cleanup
- **MTLMathModeSafe**: SDK compatibility (KVC pattern, no SDK dependency)
- **Build System**: Removed phantom CMake targets (cg_solver, zgemm_example, eigenvalue_solver)
- **Build System**: Commented out pkg-config section pending cmake/applebottom.pc.in creation
- **README**: Added precision scaling clarification (Frobenius error ~N×10⁻¹⁵)
- Test badge: 37 → 48 passing tests (6 precision + 42 correctness)

### Notes
- Validated baseline: Git tag v1.0.2-bugfix
- Production-validated for DFT, MD, FEM iterative solvers
- Version alignment across all components (headers, CMake, API)

## [1.0.2] - 2026-03-31

### Fixed (Critical)
- **BUG-1/BUG-2**: Async DGEMM dimension packing and pipeline selection (ship-blocker)
  - Fixed dimension packing: separate `setBytes` calls instead of array
  - Always use `dgemmPipeline` (async path doesn't support alpha/beta)
- **BUG-3**: `ab_dgemm_scaled` alpha/beta truncation to float (precision loss)
  - Convert alpha/beta to DD format on host before kernel dispatch
  - Metal kernel now accepts `constant DD&` instead of `constant float&`
- **BUG-4**: `ab_matrix_scale` alpha truncation to float (precision loss)
  - Convert alpha to DD format on host
- **BUG-5**: `dispatch_once_t` reset undefined behavior (V&V audit flag)
  - Replaced with `os_unfair_lock` and bool flag for reinit safety
- **BUG-6**: Memory pool overflow leak (returned unmanaged matrices)
  - Now returns NULL when pool is full instead of leaking
- **BUG-7**: Missing pipeline creation error checks (crash on init failure)
  - Added error checks for all 10 pipeline creations

### Added
- **V&V Documentation** (NASA-STD-7009A compliant)
  - `docs/vv/VV_REPORT.md`: Master validation report with traceability matrix
  - `docs/vv/PRECISION_ENVELOPE.md`: Precision guarantees and validated envelope
  - `tests/validation/VAL001_QE_Si64.md`: QE Si64 production validation
- **Verification Tests**
  - `tests/verification/test_convergence.c`: V-2 convergence study (N ∈ {64..4096})
  - 5 regression tests for BUG-1 through BUG-6
- **Test Coverage**: 42 tests (37 original + 5 regression)

### Changed
- Test badge: 37 → 42 passing tests
- Tagline: "FP64-class BLAS for scientific computing" (general-purpose)
- README: Added Verification & Validation section

### Validation Results
- **Frobenius error**: 6.5×10⁻¹⁵ to 5.1×10⁻¹⁴ (N ≤ 4096)
- **Max element error**: 3.4×10⁻¹¹ to 6.5×10⁻⁶ (documented for element-sensitive apps)
- **QE Si64 validation**: 11 decimal place agreement (-2990.44276157 Ry)
- **Performance**: 1.22× speedup vs 6-thread OpenBLAS on QE benchmark
- **Status**: VALIDATED for production DFT, MD, FEM iterative solvers

### Notes
- Validated baseline: Git tag `v1.0.2-bugfix` (SHA: d115498)
- All precision claims backed by empirical convergence data
- Known limitation: Rectangular matrices (aspect ratio >10:1) fail correctness

## [1.1.0] - 2026-03-23

### Added
- `ABTranspose` enum for matrix transpose operations (NO_TRANS, TRANS, CONJ_TRANS)
- `ab_zgemm_ex()`: Extended ZGEMM with transpose support for Quantum ESPRESSO compatibility
- GPU kernels for conjugate transpose and regular transpose operations
- Test coverage for conjugate transpose pattern (37 total tests)

### Changed
- `ab_zgemm()` now wraps `ab_zgemm_ex()` with NO_TRANS for both matrices (backward compatible)

### Deprecated
- `ab_zherk()`: Marked deprecated - CPU decomposition is 20x slower than AMX
  - Use `cblas_zherk()` from Accelerate framework instead
  - Will be removed in v2.0.0

### Technical Details
- New GPU kernels: `dd_transpose`, `dd_conj_transpose`
- Transpose pattern enables QE calbec optimization: C = A^H × B
- Maintains ~10⁻¹⁵ precision for all transpose variants

## [1.0.0] - 2026-03-22

### Added
- Initial release
- DGEMM: Double-precision general matrix multiply (618 GFLOP/s on M2 Max)
- ZGEMM: Complex double-precision matrix multiply (Gauss algorithm)
- DSYRK: Symmetric rank-k update
- ZHERK: Hermitian rank-k update
- Session API for named matrix management
- Thread-safe statistics tracking
- 22-test correctness suite
- Precision validation (~10⁻¹⁵ Frobenius error)

### Technical Details
- Double-float (FP32×2) emulation using Dekker/Knuth algorithms
- Tiled GEMM kernel: BM=BN=64, TM=TN=4, TK=16
- MTLMathModeSafe for precision guarantees
- Parallel FP64↔DD conversion for large matrices
