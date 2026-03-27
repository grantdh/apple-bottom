# apple-bottom

[![Tests](https://img.shields.io/badge/tests-37%20passing-brightgreen)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%2014%2B-orange)](https://www.apple.com/macos/)

High-performance BLAS library for Apple Silicon GPU using Metal compute shaders. Implements FP64-class operations through double-float emulation with ~10⁻¹⁵ precision.

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

## Synthetic Benchmarks

Performance on M2 Max (38-core GPU, 64 GB):

```
DGEMM:
  Size    │  AMX GFLOP/s │  GPU GFLOP/s │  Speedup
──────────┼──────────────┼──────────────┼───────────
  1024    │       547    │       483    │   0.88x
  2048    │       533    │       585    │   1.10x
  4096    │       543    │       611    │   1.12x

ZGEMM:
  2048    │       563    │       726    │   1.29x
  3072    │       590    │       696    │   1.18x
```

**Note:** Real-world performance (QE benchmark) is the primary validation metric.

## Architecture

### Double-Float (DD) Emulation

Each FP64 value is represented as a pair of FP32 values `(hi, lo)` where:
- `hi` stores the high-order 24 bits
- `lo` stores the low-order ~24 bits
- Combined precision: ~10⁻¹⁵ (48-bit mantissa)

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

- **Small matrices** (N < 2048): AMX/Accelerate is faster due to GPU overhead
- **ZHERK operation**: 20× slower than AMX, use `cblas_zherk` instead
- **Thread safety**: Matrix operations serialize via Metal command queue
- **Max dimension**: 46,340 × 46,340 (overflow protection)

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
