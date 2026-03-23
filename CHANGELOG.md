# Changelog

All notable changes to this project will be documented in this file.

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
- Precision validation (~10⁻¹⁶ Frobenius error)

### Technical Details
- Double-float (FP32×2) emulation using Dekker/Knuth algorithms
- Tiled GEMM kernel: BM=BN=64, TM=TN=4, TK=16
- MTLMathModeSafe for precision guarantees
- Parallel FP64↔DD conversion for large matrices
