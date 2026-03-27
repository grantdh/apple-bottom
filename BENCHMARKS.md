# Performance Benchmarks and Analysis

## Overview

This document provides detailed performance and precision analysis for apple-bottom, including both production validation results and synthetic benchmarks.

---

## Precision Characteristics

### Double-Float Arithmetic

**Implementation:**
- 48-bit effective mantissa (two FP32 values per FP64)
- Relative error: ~10⁻¹⁵ in practice
- Comparison: Full FP64 has 53-bit mantissa (~10⁻¹⁶)

**Production Validation:**
```
Quantum ESPRESSO Si64 benchmark:
  Reference (FP64):      -2990.44276157 Ry
  apple-bottom (DD):     -2990.44276157 Ry
  Agreement: 11 decimal places
```

**Conclusion:** Precision is sufficient for scientific computing applications including density functional theory, molecular dynamics, and quantum chemistry.

---

## Production Workload Performance

### Quantum ESPRESSO Si64 Benchmark

**System:** 64-atom silicon crystal, DFT self-consistent field calculation
**Hardware:** M2 Max (38-core GPU, 64 GB RAM)

| Configuration | Wall Time | vs Single-Thread | Energy (Ry) |
|--------------|-----------|------------------|-------------|
| OpenBLAS (1 thread) | 5:43 | 1.0× | -2990.44276157 |
| OpenBLAS (6 threads) | 2:22 | 2.4× | -2990.44276157 |
| apple-bottom GPU | 2:05 | 2.7× | -2990.44276157 |

**Performance breakdown by routine:**

| Routine | OpenBLAS 1T | OpenBLAS 6T | GPU | Speedup vs 1T |
|---------|-------------|-------------|-----|---------------|
| c_bands | 251s | 109s | 112s | 2.2× |
| cegterg | 248s | 107s | 110s | 2.3× |
| h_psi | 162s | 75.6s | 73.2s | 2.2× |
| calbec | 59.8s | 27.2s | 21.9s | 2.7× |

**Key characteristics:**
- GPU outperforms single-threaded CPU by 2.7×
- GPU outperforms 6-thread CPU by 1.14×
- Per-call overhead amortized across iterative Davidson eigensolver
- Correctness validated: energy matches reference to 11 decimal places

---

## Synthetic Benchmarks

### Square Matrix Performance

**Test conditions:** M2 Max, square matrices (M = N = K)

#### DGEMM (Real Double Precision)

| Size | AMX GFLOP/s | GPU GFLOP/s | Speedup |
|------|-------------|-------------|---------|
| 1024 | 547 | 483 | 0.88× |
| 2048 | 533 | 585 | 1.10× |
| 4096 | 543 | 611 | 1.12× |

#### ZGEMM (Complex Double Precision)

| Size | AMX GFLOP/s | GPU GFLOP/s | Speedup |
|------|-------------|-------------|---------|
| 2048 | 563 | 726 | 1.29× |
| 3072 | 590 | 696 | 1.18× |

**Notes:**
- Best case: ZGEMM at 2048×2048 (1.29× speedup)
- Typical range: 0.9-1.3× vs multi-threaded AMX
- Single large operations may not amortize GPU overhead
- Iterative workloads show significantly better performance

### Rectangular Matrix Performance

**Test conditions:** Fixed total FLOPs, varying aspect ratios

| Dimensions | Aspect Ratio | GPU Time | BLAS Time | Speedup |
|------------|--------------|----------|-----------|---------|
| 2048 × 2048 × 2048 | 1:1 | 44.6ms | 35.8ms | 0.80× |
| 4096 × 1024 × 2048 | 4:1 | 29.7ms | 28.6ms | 0.96× |
| 8192 × 512 × 2048 | 16:1 | 30.6ms | 28.9ms | 0.94× |
| 18277 × 150 × 18277 | 121:1 (QE) | 257.3ms | 202.7ms | 0.79× |
| 512 × 8192 × 2048 | 1:16 | 30.1ms | 34.7ms | 1.15× |

**Observations:**
- Wide matrices (N >> M) perform better than tall matrices (M >> N)
- Aspect ratios > 10:1 show reduced performance
- Per-call overhead dominates for extreme aspect ratios
- QE production workload achieves better performance through amortization

---

## Performance Factors

### GPU Overhead Components

**Per-call overhead:**
- FP64 → DD conversion: ~5-10 μs (GCD parallel)
- Upload to GPU: ~100 μs + data transfer
- Kernel dispatch: ~50 μs
- Download from GPU: ~100 μs + data transfer
- DD → FP64 conversion: ~5-10 μs (GCD parallel)

**Data transfer rates:**
- PCIe bandwidth: ~30 GB/s (M2 Max)
- Example: 18277 × 150 complex matrix = 44 MB ≈ 1.5 ms transfer

**Threshold:** Operations below ~100M FLOPs route to OpenBLAS to avoid overhead.

### Tiling Strategy

**Current implementation:**
- Block size: BM = BN = 64
- Thread tile: TM = TN = 4
- K-dimension tile: TK = 16

**Implications:**
- Optimized for square matrices (M ≈ N)
- Underutilizes threadgroups for tall/skinny matrices
- Example: 18277 × 150 creates only 3 threadgroups in N dimension

**Future optimization:** Adaptive tiling based on aspect ratio (see tests/RECTANGULAR_MATRICES.md).

---

## Comparison: Production vs Synthetic

### Why QE Shows Better Performance

**Synthetic benchmarks (single GEMM call):**
- Overhead dominates for medium-sized matrices
- No amortization of upload/download costs
- Result: 0.8-1.3× performance

**QE production workload (iterative algorithm):**
- 12 ZGEMM calls per Davidson iteration
- Automatic routing: small calls → OpenBLAS, large calls → GPU
- Overhead amortized across iterations
- Result: 2.7× overall speedup

**Conclusion:** apple-bottom is optimized for iterative scientific computing workloads, not single large operations.

---

## Use Case Recommendations

### Optimal Use Cases

✓ **Iterative eigensolvers** (Davidson, Lanczos)
- Multiple large GEMM operations
- Overhead amortized across iterations
- Expected: 2-3× speedup

✓ **Self-consistent field loops** (DFT, Hartree-Fock)
- Repeated matrix operations
- Large problem sizes (N ≥ 2048)
- Expected: 1.5-2.5× speedup

✓ **Square matrices** (M ≈ N ≈ K)
- Good threadgroup utilization
- Balanced memory access
- Expected: 0.9-1.3× vs multi-threaded CPU

### Suboptimal Use Cases

⚠ **Single large GEMM call**
- Overhead not amortized
- Consider Accelerate/AMX instead

⚠ **Small matrices** (N < 2048)
- Below routing threshold
- Automatically uses OpenBLAS

⚠ **Rectangular matrices** (M/N > 4 or N/M > 4)
- Poor threadgroup utilization
- Known correctness issues (under investigation)
- Native API will address this limitation

---

## Precision Limitations

### Error Accumulation

**Theoretical:**
- Single DD multiply: ~10⁻¹⁵ error
- GEMM with K operations: K × 10⁻¹⁵ error (worst case)

**Observed:**
- Sub-linear error growth in practice
- Compensated summation helps
- QE validation: exact energy match despite K=18277

### When to Use Full FP64

Consider Accelerate (AMX) instead of apple-bottom for:
- Ill-conditioned problems (condition number > 10¹⁵)
- Applications requiring true 10⁻¹⁶ precision
- Single GEMM operations (no overhead amortization)
- Small matrices (N < 2048)

---

## Future Optimizations

### Native API (GPU-Resident Matrices)

**Current limitation:**
- Upload + download overhead per GEMM call
- QE: ~50s total transfer time per benchmark

**Proposed solution:**
- Keep matrices on GPU across iterations
- Upload once at start, download once at end
- Expected improvement: 40-50% reduction in wall time

**Status:** Planned (see PARALLEL_WORKFLOW.md)

### Adaptive Tiling

**Current limitation:**
- Fixed tile sizes (BM=BN=64) for all aspect ratios
- Underutilizes GPU for rectangular matrices

**Proposed solution:**
- Detect aspect ratio
- Adjust tile sizes dynamically (e.g., BM=128, BN=32 for tall matrices)
- Expected improvement: 10-20% for rectangular matrices

**Status:** Under consideration

---

## Methodology

### Test Environment

**Hardware:**
- Mac Studio M2 Max
- 38-core GPU
- 64 GB unified memory

**Software:**
- macOS 14.3 (Sonoma)
- Metal 3.0
- Xcode 15.2

### Benchmark Procedures

**Synthetic benchmarks:**
1. Warm-up runs (3 iterations)
2. Timed runs (10 iterations, median reported)
3. Comparison: cblas_dgemm/zgemm from Accelerate framework

**QE validation:**
1. Standard Si64 input deck
2. Three configurations: 1-thread, 6-thread, GPU
3. Energy convergence: 10⁻⁸ Ry threshold
4. Wall time via `/usr/bin/time`

---

## References

- Quantum ESPRESSO integration: `docs/qe-integration.md`
- Rectangular matrix analysis: `tests/RECTANGULAR_MATRICES.md`
- Test suite: `tests/test_rectangular.c`
- QE validation script: `tests/test_qe_integration.sh`
