# apple-bottom

[![Tests](https://img.shields.io/badge/tests-37%20passing-brightgreen)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%2014%2B-orange)](https://www.apple.com/macos/)

Double-precision BLAS library for Apple Silicon GPU using Metal compute shaders. Implements FP64-class operations through double-float emulation with ~10⁻¹⁵ precision.

## Overview

Apple Silicon GPUs lack native FP64 support. This library uses double-float (FP32×2) arithmetic to achieve scientific computing precision requirements while leveraging GPU parallelism.

| Operation | Performance vs AMX | Crossover Point | Precision |
|-----------|-------------------|-----------------|-----------|
| DGEMM | +12% faster | N ≥ 2048 | ~10⁻¹⁵ |
| ZGEMM | +32% faster | N ≥ 1024 | ~10⁻¹⁵ |
| DSYRK | +14% faster | N ≥ 3072 | ~10⁻¹⁵ |

## Installation

```bash
git clone https://github.com/grantdh/apple-bottom.git
cd apple-bottom
make
make test
```

## Usage

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

## Features

### Memory Pool

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

### Complex Arithmetic

```c
// Basic ZGEMM
ab_zgemm(Ar, Ai, Br, Bi, Cr, Ci);

// With transpose support
ab_zgemm_ex(AB_CONJ_TRANS, AB_NO_TRANS, Ar, Ai, Br, Bi, Cr, Ci);
```

## API

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

## Performance

Benchmarks on M2 Max (38-core GPU, 64 GB):

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

## Requirements

- macOS 14+ (Sonoma)
- Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools

## Limitations

- Small matrices (N < 2048): AMX/Accelerate is faster
- ZHERK operation: 20× slower than AMX, use `cblas_zherk` instead
- Thread safety: Matrix operations not thread-safe (Metal command queue serializes)
- Max dimension: 46,340 × 46,340 (overflow protection)

## License

MIT License - see [LICENSE](LICENSE)
