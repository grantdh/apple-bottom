<p align="center">
  <h1 align="center">apple-bottom</h1>
  <p align="center"><strong>FP64-class BLAS on Apple Silicon GPU — 618 GFLOP/s, ~10⁻¹⁵ precision, zero CUDA dependency</strong></p>
</p>

<p align="center">
  <a href="https://github.com/grantdh/apple-bottom/actions"><img src="https://github.com/grantdh/apple-bottom/actions/workflows/vv-regression.yml/badge.svg" alt="CI"></a>
  <a href="tests/"><img src="https://img.shields.io/badge/tests-115%20passing-brightgreen" alt="Tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
  <a href="#requirements"><img src="https://img.shields.io/badge/platform-Apple%20Silicon-orange" alt="Platform"></a>
  <a href="#architecture"><img src="https://img.shields.io/badge/precision-~10⁻¹⁵-yellow" alt="Precision"></a>
</p>

---

Apple Silicon GPUs have no native FP64. `apple-bottom` fixes that — double-float (DD) arithmetic on Metal compute shaders gives you FP64-class matrix operations at GPU throughput. Production-validated with Quantum ESPRESSO (22% faster, 11 decimal places of agreement).

```bash
git clone https://github.com/grantdh/apple-bottom.git
cd apple-bottom
make && make test   # 115/115 tests pass
```

## Performance

Benchmarked on M2 Max (38-core GPU, 64 GB unified memory):

| Operation | GPU (apple-bottom) | CPU (Accelerate AMX) | Speedup | Precision |
|-----------|-------------------|---------------------|---------|-----------|
| DGEMM 2048² | 552 GFLOP/s | 468 GFLOP/s | **+18%** | ~10⁻¹⁵ |
| DGEMM 4096² | 643 GFLOP/s | 566 GFLOP/s | **+14%** | ~10⁻¹⁵ |
| ZGEMM (QE Si64) | — | — | **+22% wall** | 11 decimal places |
| ZGEMM (QE Si128) | — | — | **+12% wall** | exact energy match |

**Quantum ESPRESSO production benchmark (Si64, 64 atoms, DFT):**

```
Configuration          Wall Time    CPU Usage    Energy (Ry)
──────────────────────────────────────────────────────────────
OpenBLAS 6-thread         2:28        5.3×       -2990.44276157
apple-bottom GPU          2:01        3.4×       -2990.44276157  ✓
```

22% faster wall time, 47% less CPU usage, bit-identical energy.

### When to Use apple-bottom

**Use it for:** iterative solvers (SCF, CG, GMRES, Lanczos), large matrices (N ≥ 2048), repeated GEMM in loops — anywhere GPU upload/download overhead amortizes across iterations.

**Don't use it for:** single small GEMM calls (N < 1024), element-sensitive algorithms (pivoted LU, eigensolvers), or when IEEE 754 bit-exact FP64 is required.

## API

```c
#include "apple_bottom.h"

ab_init();

// Create and upload matrices
ABMatrix A = ab_matrix_create(2048, 2048);
ABMatrix B = ab_matrix_create(2048, 2048);
ABMatrix C = ab_matrix_create(2048, 2048);
ab_matrix_upload(A, data_A, true);
ab_matrix_upload(B, data_B, true);

// C = A × B  (FP64-class precision on GPU)
ab_dgemm(A, B, C);

// Download result
ab_matrix_download(C, result, true);

// Cleanup
ab_matrix_destroy(A);
ab_matrix_destroy(B);
ab_matrix_destroy(C);
ab_shutdown();
```

### Full API Surface

| Category | Functions |
|----------|-----------|
| **BLAS** | `ab_dgemm`, `ab_dgemm_scaled`, `ab_zgemm`, `ab_zgemm_ex`, `ab_dsyrk` |
| **Matrix** | `ab_matrix_create`, `ab_matrix_upload`, `ab_matrix_download`, `ab_matrix_zero`, `ab_matrix_copy` |
| **Element-wise** | `ab_matrix_add`, `ab_matrix_sub`, `ab_matrix_scale` |
| **Async** | `ab_dgemm_async`, `ab_zgemm_async`, `ab_future_wait` |
| **Pool** | `ab_pool_create`, `ab_pool_get_matrix`, `ab_pool_reset` (reduces allocation overhead in iterative codes) |
| **Session** | `ab_session_create`, `ab_session_dgemm`, `ab_session_zgemm` (named matrix management) |
| **Device** | `ab_dev_malloc`, `ab_dev_free`, `ab_dev_memcpy_h2d/d2h/d2d`, `ab_dev_dgemm`, `ab_dev_zgemm` (DevXlib-compatible buffer API) |

### Fortran Integration (Quantum ESPRESSO)

Drop-in replacement via EXTERNAL declaration — no module dependencies:

```fortran
EXTERNAL :: ab_zgemm
CALL ab_zgemm('N', 'C', m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
```

The Fortran bridge auto-routes small calls (< 100M FLOPs) to OpenBLAS and large calls to GPU. See [Integration Guide](docs/INTEGRATION.md) for C, Python, and Fortran setup.

## Architecture

### Double-Float (DD) Arithmetic

Each FP64 value is stored as two FP32 values `(hi, lo)` using Dekker/Knuth error-free transformations. This gives ~48-bit effective mantissa (~10⁻¹⁵ relative error) — not full IEEE 754 FP64 (53-bit, ~10⁻¹⁶), but sufficient for scientific computing where accumulated solver errors dominate.

Key implementation details: `MTLMathModeSafe` prevents the Metal compiler from reordering FMA operations that would destroy DD precision. Complex GEMM uses Gauss's 3-multiply algorithm (3 real GEMM instead of 4, 25% compute reduction). The GPU kernel uses 4×4 register blocking with BM=BN=64 tiling, optimized for Apple Silicon GPU occupancy.

### Precision Guarantees

Per-element DD precision is ~10⁻¹⁵. DGEMM Frobenius error scales as ~√N×10⁻¹⁵ (statistical cancellation in random matrices). For N=4096, empirical Frobenius error is ~5×10⁻¹⁴. Full analysis in [Precision Envelope](docs/vv/PRECISION_ENVELOPE.md).

### Fortran Bridge Architecture

```
Fortran (QE)  →  fortran_bridge.c  →  < 100M FLOPs? → cblas (OpenBLAS/Accelerate)
                                    →  ≥ 100M FLOPs? → blas_wrapper.c → Metal GPU
```

## Verification & Validation

Production-validated following NASA-STD-7009A methodology:

- **V-2 Convergence Study**: Frobenius error 6.5×10⁻¹⁵ to 5.1×10⁻¹⁴ for N ∈ {64, 128, ..., 4096}
- **VAL-1 Production**: Quantum ESPRESSO Si64 DFT — 11 decimal place agreement (-2990.44276157 Ry)
- **115/115 tests**: 9 precision + 66 correctness + 40 device-API tests (regression coverage for 7 critical bug fixes plus Week-2 device-buffer BLAS validation)

Documentation: [V&V Report](docs/vv/VV_REPORT.md) · [Precision Envelope](docs/vv/PRECISION_ENVELOPE.md) · [QE Validation](tests/validation/VAL001_QE_Si64.md)

## Requirements

- **macOS 14+** (Sonoma) with **Xcode 16+** SDK (for `MTLMathModeSafe`)
- **Apple Silicon** (M1, M2, M3, M4 — any variant)
- Older SDKs compile but achieve only ~10⁻⁸ precision (fast-math breaks DD arithmetic)

## Build Options

- `make DWTIMESDW3=1` — opt in to the tighter-bound DD multiplication variant (JMP 2017 Alg 12 / MR 2022 Thm 2.8, error bound < 4u² vs the default Alg 11's < 5u²). Trades one additional multiply, one FMA, and one add per `dd_mul`/`dd_fma`. The default algorithm's bound derivation lives in [`docs/vv/PRECISION_ENVELOPE.md`](docs/vv/PRECISION_ENVELOPE.md).

## Prior Art

| Project | Approach | Throughput | Status |
|---------|----------|------------|--------|
| [metal-float64](https://github.com/philipturner/metal-float64) (Turner) | Integer FP64 emulation via SoftFloat/LLVM | ~24 GFLOP/s | Archived 2024 |
| [AppleNumericalComputing](https://github.com/ShoYamanishi/AppleNumericalComputing) (Yamanishi) | Educational benchmarks, FP32 GPU | N/A | Research |
| [MLX](https://github.com/ml-explore/mlx) (Apple) | ML framework, FP64 on CPU only | N/A | Active, no GPU FP64 |
| Accelerate (Apple) | AMX hardware, IEEE FP64 | 536 GFLOP/s | CPU-only, thread-hostile |
| **apple-bottom** | **DD arithmetic on Metal FP32 ALUs** | **643 GFLOP/s** | **Active, production-validated** |

`apple-bottom` takes a fundamentally different approach from `metal-float64`: native FP32 ALU arithmetic with error-free transformations (Dekker 1971) instead of integer bit manipulation. This trades strict IEEE 754 compliance for ~25× higher throughput at the BLAS level, which is the right tradeoff for scientific iterative solvers where norm-averaged convergence dominates.

## Research

Triple-double (TD) emulation achieves faithfully-rounded FP64 (99.5% correctly rounded, max 1 ULP error) at 148 GFLOP/s — a new point on the precision-performance Pareto frontier. See [`research/td-dgemm/`](research/td-dgemm/).

## Limitations

- **Single GEMM calls**: ~100 μs GPU overhead dominates for one-shot operations
- **ZHERK**: deprecated (20× slower than CPU AMX) — use `cblas_zherk` instead
- **Thread safety**: Metal command queue serializes; use separate contexts for concurrency (planned)
- **Max dimension**: 46,340 × 46,340 (overflow protection)

## Project Structure

```
apple-bottom/
├── src/apple_bottom.m          # Core Metal GPU implementation (DD kernels)
├── src/device_api.m            # Device-buffer API for DevXlib backends
├── src/blas_wrapper.c          # BLAS-compatible C API
├── src/fortran_bridge.c        # Fortran ABI bridge for QE/VASP/CP2K
├── include/apple_bottom.h      # Public API header (v1.3.0-dev)
├── include/apple_bottom_device.h  # Device-buffer API header
├── tests/                      # 115 tests (precision + correctness + device-API)
├── benchmarks/                 # DGEMM, ZGEMM, DSYRK, pool, async benchmarks
├── examples/01_basic_dgemm/    # Runnable example
├── docs/                       # Integration guide + V&V documentation
└── research/td-dgemm/          # Triple-double faithfully-rounded FP64 research
```

## Ecosystem

apple-bottom powers GPU acceleration across a family of scientific computing tools:

| Project | What it does |
|---------|-------------|
| [Quantum-Espressivo](https://github.com/grantdh/Quantum-Espressivo) | Quantum ESPRESSO + Metal GPU acceleration (22% speedup, 11 decimal place agreement) |
| [YAMBOrghini](https://github.com/grantdh/YAMBOrghini) | Yambo GW/BSE + Metal GPU acceleration |
| [MEEPhistopheles](https://github.com/grantdh/MEEPhistopheles) | Metal FDTD kernels for electromagnetic simulation (3200+ Mcells/s) |
| [rainbow-connection](https://github.com/grantdh/rainbow-connection) | Multi-physics pipeline orchestrator (QE → Yambo → MEEP) |

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Bug reports and feature requests welcome via [Issues](https://github.com/grantdh/apple-bottom/issues).

## Citation

```bibtex
@software{heileman2026applebottom,
  author    = {Heileman, Grant David},
  title     = {apple-bottom: FP64-class BLAS for Apple Silicon GPU},
  year      = {2026},
  url       = {https://github.com/grantdh/apple-bottom},
  version   = {1.3.0-dev}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
