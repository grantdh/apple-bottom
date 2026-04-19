# Changelog

All notable changes to this project will be documented in this file.

## [1.3.1-dev] - Unreleased

_TBD following Si64 benchmark campaign._

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
- **BUG-8: ARC leak in Metal-ref structs** — `ABMatrix_s`, `ABFuture_s`, `ABBatch_s`, and `ab_dev_buffer_s` now allocate via `new`/`delete` so ARC destructors release their `id<MTL…>` strong refs (was `calloc`/`free`, which does not run ARC destructors and leaked the buffers, command buffers, and encoders). Landed in 55bc79a.

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
- Validated baseline: Git tag `v1.0.2-bugfix` (SHA: 700934f)
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
