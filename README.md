# apple-bottom

[![Tests](https://img.shields.io/badge/tests-37%20passing-brightgreen)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%2014%2B-orange)](https://www.apple.com/macos/)
[![Metal](https://img.shields.io/badge/Metal-3.0+-red)](https://developer.apple.com/metal/)
[![Precision](https://img.shields.io/badge/precision-10⁻¹⁵-yellow)](#architecture)
[![QE Validated](https://img.shields.io/badge/QE-2.7×%20speedup-success)](#quantum-espresso-benchmark)

High-performance BLAS library for Apple Silicon GPU using Metal compute shaders. Implements FP64-class operations through double-float emulation with ~10⁻¹⁵ precision.

---

## Table of Contents

- [Status](#status-production-integration)
- [Quantum ESPRESSO Benchmark](#quantum-espresso-benchmark)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Performance](#performance-summary)
- [Architecture](#architecture)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

---

## Status: Production Integration

**Quantum ESPRESSO integration working** — 2.7× speedup over single-threaded OpenBLAS, 14% faster than 6-thread OpenBLAS on si64 benchmark.

```
Configuration          Wall Time    vs 1-thread    Energy
────────────────────────────────────────────────────────────
OpenBLAS (6 threads)      2:22         2.4×       -2990.44276157 Ry
OpenBLAS (1 thread)       5:43         1.0×       -2990.44276157 Ry
apple-bottom GPU          2:05         2.7×       -2990.44276157 Ry ✓
```

Integration via Fortran bridge with EXTERNAL declaration — minimal code changes, no module dependencies.

## Overview

Apple Silicon GPUs lack native FP64 support. This library uses double-float (FP32×2) arithmetic to achieve scientific computing precision while leveraging GPU parallelism. Validated in production with Quantum ESPRESSO.

## Quick Start

### Installation

```bash
git clone https://github.com/grantdh/apple-bottom.git
cd apple-bottom
make
make test
```

### C API Usage

```c
#include "apple_bottom.h"

int main() {
    ab_init();

    ABMatrix A = ab_matrix_create(2048, 2048);
    ABMatrix B = ab_matrix_create(2048, 2048);
    ABMatrix C = ab_matrix_create(2048, 2048);

    ab_matrix_upload(A, data_A, true);
    ab_matrix_upload(B, data_B, true);

    ab_dgemm(A, B, C);  // C = A × B

    ab_matrix_download(C, result, true);

    ab_matrix_destroy(A);
    ab_matrix_destroy(B);
    ab_matrix_destroy(C);
    ab_shutdown();
}
```

### Fortran Integration (Quantum ESPRESSO)

Drop-in replacement for BLAS routines via EXTERNAL declaration:

```fortran
! In your eigensolver:
IMPLICIT NONE
EXTERNAL :: ab_zgemm

! Replace CALL ZGEMM with:
CALL ab_zgemm('N', 'N', m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
```

The Fortran bridge automatically routes:
- **Small calls** (< 100M FLOPs) → OpenBLAS (zero overhead)
- **Large calls** (≥ 100M FLOPs) → GPU

**Integration Guides:**
- [Quantum ESPRESSO Integration](docs/qe-integration.md) — Step-by-step QE 7.4.1 integration
- [Fortran Integration Guide](docs/fortran-integration.md) — General Fortran BLAS integration
- [C API Integration](docs/INTEGRATION.md) — Native C API usage

## Quantum ESPRESSO Benchmark

**System:** Si64 (64-atom silicon crystal)
**Hardware:** M2 Max (38-core GPU, 64 GB RAM)
**Validation:** Total energy `-2990.44276157 Ry` (exact match to baseline)

### Performance Breakdown

| Routine | OpenBLAS 6T | OpenBLAS 1T | GPU | Speedup |
|---------|-------------|-------------|-----|---------|
| **Total** | 2:22 | 5:43 | **2:05** | **2.7×** |
| c_bands | 109s | 251s | 112s | 2.2× |
| cegterg | 107s | 248s | 110s | 2.3× |
| h_psi | 75.6s | 162s | 73.2s | 2.2× |
| calbec | 27.2s | 59.8s | 21.9s | 2.7× |

The GPU achieves single-threaded CPU performance with 14% less wall time than 6-thread OpenBLAS.

## API Reference

### Core Operations

```c
ABStatus ab_dgemm(ABMatrix A, ABMatrix B, ABMatrix C);
ABStatus ab_dgemm_scaled(double alpha, ABMatrix A, ABMatrix B,
                         double beta, ABMatrix C);
ABStatus ab_zgemm(ABMatrix Ar, ABMatrix Ai, ABMatrix Br, ABMatrix Bi,
                  ABMatrix Cr, ABMatrix Ci);
ABStatus ab_zgemm_ex(ABTranspose transA, ABTranspose transB,
                     ABMatrix Ar, ABMatrix Ai, ABMatrix Br, ABMatrix Bi,
                     ABMatrix Cr, ABMatrix Ci);
ABStatus ab_dsyrk(ABMatrix A, ABMatrix C);
```

### Matrix Management

```c
ABMatrix ab_matrix_create(int rows, int cols);
void ab_matrix_destroy(ABMatrix m);
ABStatus ab_matrix_upload(ABMatrix m, const double* data, bool parallel);
ABStatus ab_matrix_download(ABMatrix m, double* data, bool parallel);
ABStatus ab_matrix_zero(ABMatrix m);
ABStatus ab_matrix_copy(ABMatrix src, ABMatrix dst);
```

### Memory Pooling

Reduces allocation overhead in iterative codes:

```c
ABMemoryPool pool = ab_pool_create(0);

for (int i = 0; i < 100; i++) {
    ABMatrix tmp = ab_pool_get_matrix(pool, N, N);
    // ... use matrix ...
    ab_pool_reset(pool);  // Reuse without free
}

ab_pool_destroy(pool);
```

### Async Operations

Overlap CPU and GPU work:

```c
ABFuture f = ab_dgemm_async(A, B, C);
// Do CPU work while GPU computes
ab_future_wait(f);
ab_future_destroy(f);
```

## Performance Summary

**Production validation (Quantum ESPRESSO):** Best-case scenario showing the library's strengths.

**Square matrices (synthetic):** Performance varies 0.9-1.3× vs 6-thread OpenBLAS depending on size and conditions. The GPU shines in iterative workloads where upload/download overhead is amortized.

**Rectangular matrices:** Currently 0.8-1.0× vs OpenBLAS (known limitation being addressed in native API).

### When to Use apple-bottom

✅ **Iterative algorithms:** Davidson eigensolvers, SCF loops, Lanczos, etc.
✅ **Large matrices:** N ≥ 2048 (amortizes GPU overhead)
✅ **Repeated operations:** Multiple GEMM calls in a loop

⚠ **Variable performance:**
- Single large GEMM call: Overhead may dominate
- Small matrices (N < 2048): Use Accelerate (AMX) instead
- Rectangular matrices (M/N > 4): Known performance/correctness issues

### Synthetic Benchmarks

Square matrices on M2 Max (38-core GPU, 64 GB):

```
DGEMM (best case):
  2048 × 2048 × 2048:    1.10× vs single-threaded OpenBLAS
  4096 × 4096 × 4096:    1.12× vs single-threaded OpenBLAS

ZGEMM (best case):
  2048 × 2048 × 2048:    1.29× vs single-threaded OpenBLAS
  3072 × 3072 × 3072:    1.18× vs single-threaded OpenBLAS
```

**Note:** Performance vs multi-threaded OpenBLAS is typically 0.9-1.1× for square matrices. Real-world QE workload (2.7× speedup) demonstrates value in iterative contexts.

## Architecture

### Double-Float (DD) Emulation

Each FP64 value is represented as a pair of FP32 values `(hi, lo)` where:
- `hi` stores the high-order 24 bits (FP32 mantissa)
- `lo` stores the low-order ~24 bits (error correction)
- **Combined precision: ~10⁻¹⁵** (48-bit effective mantissa)
- **NOT full FP64:** True FP64 has 53-bit mantissa (~10⁻¹⁶)

This is sufficient for scientific computing where accumulated errors are typically << 10⁻¹⁵.

### Gauss 3-Multiply Algorithm

Matrix multiply uses Gauss's algorithm to reduce complex multiplies:
```
(a + bi)(c + di) = ac - bd + i[(a+b)(c+d) - ac - bd]
```
3 DD multiplies instead of 4 (25% reduction in compute).

### Fortran Bridge

```
Fortran caller (QE)
    ↓ EXTERNAL :: ab_zgemm
fortran_bridge.c: ab_zgemm_()
    ↓ dereference pointers, check threshold
    ├─ < 100M FLOPs → zgemm_() passthrough (OpenBLAS)
    └─ ≥ 100M FLOPs → ab_zgemm_blas() (GPU)
        ↓
blas_wrapper.c: split-complex conversion
    ↓
apple_bottom.m: Metal kernel dispatch
```

## Requirements

- macOS 14+ (Sonoma)
- Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools

## Limitations

### Performance Limitations

- **Small matrices** (N < 2048): AMX/Accelerate is faster due to GPU overhead (~100 μs per call)
- **Rectangular matrices** (M/N > 4): Currently 0.8-1.0× vs OpenBLAS
  - Known issue: Correctness failures for large rectangles (being investigated)
  - Use square matrices or wait for native API
- **Single GEMM calls**: Per-call overhead (upload/download) dominates
  - Best for iterative algorithms with many calls
- **ZHERK operation**: 20× slower than AMX, use `cblas_zherk` instead

### Precision Limitations

- **~10⁻¹⁵ relative error**, not full FP64 (10⁻¹⁶)
  - 48-bit effective mantissa vs 53-bit for FP64
  - Sufficient for scientific computing (validated with QE)
  - For true FP64, use Accelerate (AMX)

### System Limitations

- **Thread safety**: Matrix operations serialize via Metal command queue
- **Max dimension**: 46,340 × 46,340 (overflow protection)
- **macOS only**: Requires Metal framework (Apple Silicon M1/M2/M3/M4)

## Validation

Run the integration test suite:

```bash
./tests/test_qe_integration.sh
```

This validates:
- Library symbols (Fortran `_ab_zgemm_`, C `_ab_zgemm_blas`)
- QE patches (cegterg.f90, make.inc)
- Build configuration

For full QE validation (requires Quantum ESPRESSO):

```bash
cd ~/qe-test/benchmark
~/qe-test/q-e-qe-7.4.1/bin/pw.x < si64.in > si64.out 2>&1
grep '!' si64.out  # Should output: -2990.44276157 Ry
```

## Research

Ongoing research on triple-double (TD) emulation for correctly-rounded FP64 is documented in [`research/td-dgemm/`](research/td-dgemm/).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [`LICENSE`](LICENSE)

## Project Structure

```
apple-bottom/
├── src/
│   ├── apple_bottom.m          # Core Metal implementation
│   ├── blas_wrapper.c          # BLAS-compatible C API
│   └── fortran_bridge.c        # Fortran ABI bridge
├── include/
│   └── apple_bottom.h          # Public API header
├── tests/
│   ├── test_correctness.c      # Unit tests
│   └── test_qe_integration.sh  # QE validation
├── benchmarks/                 # Performance benchmarks
├── research/td-dgemm/          # Research prototypes
└── LESSONS_LEARNED.md          # Engineering insights
```
