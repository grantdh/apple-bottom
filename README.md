# apple-bottom 🍑

[![Tests](https://img.shields.io/badge/tests-36%20passing-brightgreen)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![macOS](https://img.shields.io/badge/macOS-14%2B-orange)](https://www.apple.com/macos/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3%2FM4-red)](https://support.apple.com/en-us/116943)

**FP64-precision BLAS for Apple Silicon GPU** — double-float emulation achieving 10⁻¹⁶ precision with up to 1.3× speedup over AMX for large matrices.

## Why apple-bottom?

Apple Silicon GPUs only support FP32 natively. Scientific computing (DFT, quantum chemistry, molecular dynamics) requires FP64. This library uses **double-float arithmetic** to deliver FP64-class precision on the GPU.

| Operation | vs AMX | Crossover | Precision |
|-----------|--------|-----------|-----------|
| **DGEMM** | +12% faster | N ≥ 2048 | 10⁻¹⁶ |
| **ZGEMM** | +32% faster | N ≥ 1024 | 10⁻¹⁶ |
| **DSYRK** | +14% faster | N ≥ 3072 | 10⁻¹⁶ |
| ZHERK | Use AMX | — | — |

## Quick Start
```bash
git clone https://github.com/grantdh/apple-bottom.git
cd apple-bottom
make
make test        # 36 tests, all passing
make bench       # See performance on your machine
```

## Usage
```c
#include "apple_bottom.h"

int main() {
    ab_init();
    
    // Create matrices
    ABMatrix A = ab_matrix_create(2048, 2048);
    ABMatrix B = ab_matrix_create(2048, 2048);
    ABMatrix C = ab_matrix_create(2048, 2048);
    
    // Upload data
    ab_matrix_upload(A, my_data_A, true);
    ab_matrix_upload(B, my_data_B, true);
    
    // Compute C = A × B (FP64 precision on GPU!)
    ab_dgemm(A, B, C);
    
    // Download result
    ab_matrix_download(C, result, true);
    
    ab_matrix_destroy(A);
    ab_matrix_destroy(B);
    ab_matrix_destroy(C);
    ab_shutdown();
}
```

## Features for Scientific Computing

### Memory Pool (1.4-1.8× faster SCF loops)
```c
ABMemoryPool pool = ab_pool_create(0);

for (int scf = 0; scf < 100; scf++) {
    ABMatrix F = ab_pool_get_matrix(pool, N, N);  // Reuses allocation
    ABMatrix D = ab_pool_get_matrix(pool, N, N);
    ABMatrix FD = ab_pool_get_matrix(pool, N, N);
    
    // ... compute ...
    
    ab_pool_reset(pool);  // Mark all as available (no free!)
}

ab_pool_destroy(pool);
```

### Async API (18× speedup with CPU/GPU overlap)
```c
ABFuture f = ab_dgemm_async(A, B, C);

// Do CPU work while GPU computes
prepare_next_iteration();
compute_eigenvalues();

ab_future_wait(f);  // Block until GPU done
ab_future_destroy(f);
```

### Complex Arithmetic (ZGEMM)
```c
// C = A × B where A, B, C are complex matrices
// Stored as separate real/imaginary parts
ab_zgemm(Ar, Ai, Br, Bi, Cr, Ci);
```

## Performance (M2 Max)
```
DGEMM Benchmark:
  Size    │  AMX GFLOP/s │  GPU GFLOP/s │  Speedup
──────────┼──────────────┼──────────────┼───────────
  1024    │       547    │       483    │   0.88x
  2048    │       533    │       585    │   1.10x ✓
  4096    │       543    │       611    │   1.12x ✓

ZGEMM Benchmark:
  2048    │       563    │       726    │   1.29x ✓
  3072    │       590    │       696    │   1.18x ✓

Memory Pool (100 SCF iterations):
  256x256:  1.77x faster with pool
  1024x1024: 1.46x faster with pool

Async Overlap:
  Sync:  110.8ms | Async: 6.0ms | 18.6× speedup
```

## API Reference

### Core Operations
```c
ABStatus ab_dgemm(A, B, C);           // C = A × B
ABStatus ab_dgemm_scaled(α, A, B, β, C); // C = αAB + βC
ABStatus ab_zgemm(...);               // Complex GEMM
ABStatus ab_dsyrk(A, C);              // C = A × Aᵀ
ABStatus ab_zherk(...);               // Complex Hermitian rank-k
```

### Matrix Management
```c
ABMatrix ab_matrix_create(rows, cols);
void ab_matrix_destroy(m);
ABStatus ab_matrix_upload(m, data, parallel);
ABStatus ab_matrix_download(m, data, parallel);
ABStatus ab_matrix_zero(m);
ABStatus ab_matrix_copy(src, dst);
```

### Memory Pool
```c
ABMemoryPool ab_pool_create(size_hint);
ABMatrix ab_pool_get_matrix(pool, rows, cols);
void ab_pool_reset(pool);
void ab_pool_destroy(pool);
```

### Async Operations
```c
ABFuture ab_dgemm_async(A, B, C);
ABStatus ab_future_wait(f);
bool ab_future_is_ready(f);
void ab_future_destroy(f);
```

## Requirements

- macOS 14+ (Sonoma)
- Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools

## Thread Safety

- `ab_init()`/`ab_shutdown()`: Thread-safe (uses `dispatch_once`)
- Matrix operations: NOT thread-safe (Metal command queue serializes)
- Use separate pools for concurrent workloads

## Limitations

- **ZHERK**: Use `cblas_zherk` instead — GPU decomposition is 20× slower
- **Small matrices**: AMX wins below crossover points
- **Max dimension**: 46,340 × 46,340 (overflow protection)

## Citation

If you use apple-bottom in your research, please cite:
```bibtex
@software{apple_bottom,
  author = {Heileman, Grant},
  title = {apple-bottom: FP64 BLAS for Apple Silicon GPU},
  year = {2026},
  url = {https://github.com/grantdh/apple-bottom}
}
```

## License

MIT License — see [LICENSE](LICENSE)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**Built for quantum chemistry on Apple Silicon** 🧪💻
