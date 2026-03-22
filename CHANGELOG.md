# Changelog

All notable changes to this project will be documented in this file.

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
