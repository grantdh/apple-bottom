# Precision Envelope — apple-bottom v1.0.2

**Document Status**: Validated (2026-03-31)
**Validation Baseline**: Git tag `v1.0.2-bugfix` (SHA: 700934f)
**Test System**: Apple M2 Max, macOS 14.7, Metal 3

---

## Executive Summary

**apple-bottom** provides FP64-class precision through double-float (DD) arithmetic on Apple Silicon GPUs. This document defines the validated precision envelope—the operating conditions under which accuracy guarantees hold.

### Stated Guarantee

For matrix multiplication `C = A × B` where:
- Matrix dimensions: `64 ≤ N ≤ 4096` (square matrices, validated range)
- Matrix entries: `|A[i,j]|, |B[i,j]| ∈ [10⁻³⁰, 10³⁰]` (FP32 exponent range)
- Condition number: `κ(A), κ(B) < 10¹²`

apple-bottom guarantees:

```
‖C_gpu - C_ref‖_F / ‖C_ref‖_F  <  N · 2⁻⁴⁸
```

where `C_ref` is the IEEE FP64 reference result and `2⁻⁴⁸` is the DD mantissa precision.

**In practice**: Frobenius relative error for random matrices scales as `~N · 10⁻¹⁵`, but the theoretical `O(N·2⁻⁴⁸)` bound applies to worst-case inputs. See Section 2.3 for critical caveats on element-wise vs. norm-averaged errors.

---

## 1. Double-Float Arithmetic Foundation

### 1.1 Representation

Each FP64 value `x` is stored as two FP32 values `(hi, lo)`:

```
x ≈ hi + lo
```

where:
- `hi` = high-order 24 bits (FP32 conversion of `x`)
- `lo` = low-order ~24 bits (error correction term: `x - hi`)

**Effective precision**: ~48 bits of mantissa (vs. 53 bits for true FP64)

**Error bound**: Single DD operation introduces `≤ 2⁻⁴⁸` relative error

### 1.2 Error-Free Transformations

DD operations use Dekker/Knuth algorithms:

| Operation | Algorithm | FLOPs | Error Bound |
|-----------|-----------|-------|-------------|
| Addition | `twoSum` | 6 | `≤ ε_mach` |
| Multiplication | `twoProduct` (FMA) | 2 | `≤ ε_mach` |
| FMA | `dd_fma` | 8 | `≤ 2ε_mach` |

where `ε_mach = 2⁻²⁴` for FP32.

### 1.3 DGEMM Error Analysis

For `C = A × B` with dimensions `M × N` and `K`:

**Theoretical bound** (Higham, 2002):

```
‖ΔC‖_F ≤ γ_K · ‖A‖_F · ‖B‖_F
```

where `γ_K = K · ε / (1 - K · ε) ≈ K · ε` for `K · ε ≪ 1`, and `ε = 2⁻⁴⁸` for DD.

**For normalized matrices** (`‖A‖_F = ‖B‖_F = √N`):

```
relative_error ≈ K · 2⁻⁴⁸ = N · 2⁻⁴⁸
```

---

## 2. Empirical Validation

### 2.1 Convergence Study (V-2)

**Test Setup**:
- Matrix sizes: `N ∈ {64, 128, 256, 512, 1024, 2048, 4096}`
- Random matrices: `A[i,j], B[i,j] ~ Uniform(-1, 1)`, seed = 42
- Reference: `cblas_dgemm` (IEEE FP64, Accelerate framework)
- Metric: Frobenius relative error

**Results** (validated 2026-03-31):

| N    | Frobenius Error | Max Element Error | Status |
|------|-----------------|-------------------|--------|
| 64   | 6.54×10⁻¹⁵     | 3.36×10⁻¹¹       | ✓ PASS |
| 128  | 9.16×10⁻¹⁵     | 4.35×10⁻⁹        | ✓ PASS |
| 256  | 1.29×10⁻¹⁴     | 1.92×10⁻¹⁰       | ✓ PASS |
| 512  | 1.80×10⁻¹⁴     | 1.14×10⁻⁹        | ✓ PASS |
| 1024 | 2.55×10⁻¹⁴     | 1.30×10⁻⁸        | ✓ PASS |
| 2048 | 3.59×10⁻¹⁴     | 1.99×10⁻⁷        | ✓ PASS |
| 4096 | 5.09×10⁻¹⁴     | 6.54×10⁻⁶        | ✓ PASS |

**Convergence slope** (log-log regression on Frobenius error):
```
log(error) = 0.493 · log(N) - 31.7
R² = 1.0000
```

The observed `~√N` scaling (slope ≈ 0.5) is **expected for random matrices** due to statistical error cancellation—positive and negative rounding errors partially cancel in the Frobenius norm. Structured matrices (e.g., all-positive entries) will exhibit the theoretical **O(N)** worst-case growth. **Users should rely on the stated `N·2⁻⁴⁸` bound, not the empirically observed `√N` scaling.**

**Data source**: `tests/verification/test_convergence.c` → `build/convergence_data.csv`

### 2.2 Interpretation and Caveats

**⚠ CRITICAL CAVEAT**: The observed `~√N` scaling (slope = 0.493) reflects **statistical error cancellation in random matrices**. This is NOT a guarantee of better-than-theoretical performance. Structured or adversarial matrices (e.g., all-positive entries, or matrices designed to maximize rounding error accumulation) will exhibit the theoretical **O(N)** worst-case growth.

**The stated guarantee uses the O(N·2⁻⁴⁸) bound, not the empirical √N**. Random cancellation is a statistical artifact of the Uniform(-1,1) test distribution and should not be relied upon for production safety margins.

**For conservative error budgeting**: Use `error ≤ N · 10⁻¹⁴` as the design bound, not the observed `~10⁻¹⁴` values.

### 2.3 Element-Wise vs. Norm-Averaged Error

**⚠ CRITICAL WARNING**: Frobenius norm error is **not representative of individual element errors**.

At N=4096, the Frobenius error is `5.09×10⁻¹⁴` but the **maximum element error is `6.54×10⁻⁶`** — **8 orders of magnitude worse**.

**Why this matters**:
- **Norm-averaged operations** (iterative solvers, energy calculations): Frobenius error is appropriate
- **Element-sensitive operations** (pivoting, eigenvalue extraction, matrix inversions): Max element error dominates

**Applications requiring element-wise accuracy**:
- Gaussian elimination with pivoting
- Eigenvalue/eigenvector computations (QR, Jacobi)
- Matrix condition number estimation
- Any algorithm that makes decisions based on individual matrix entries

**Recommendation**: If your application depends on individual element accuracy better than `10⁻⁶`, verify with a representative test case or use CPU FP64.

**Max element error scaling** (empirical):
```
log(max_err) = 2.494 · log(N) - 21.9
R² = 0.810
```

This `~N²·⁵` scaling means max element error grows **faster than the Frobenius bound** and will eventually dominate.

### 2.4 Production Validation (Quantum ESPRESSO)

**Test Case**: Si64 SCF convergence (64-atom silicon crystal)

| Metric | OpenBLAS (FP64 Reference) | apple-bottom (DD) | Agreement |
|--------|---------------------------|-------------------|-----------|
| Total Energy | -2990.44276157 Ry | -2990.44276157 Ry | **11 decimal places** |
| Wall Time | 2m28s (6 threads) | 2m01s (1 thread + GPU) | 1.22× speedup |
| Final Convergence | `|ΔE| < 10⁻¹⁰ Ry` | `|ΔE| < 10⁻¹⁰ Ry` | Identical |

**Implication**: DD precision is sufficient for production DFT calculations requiring `~10⁻¹⁰` energy convergence. DFT energy is a norm-averaged quantity (sum over occupied states), so Frobenius-class error is appropriate.

**Data source**: `docs/BENCHMARK_SUMMARY.md`, QE v7.2 validation

---

## 3. Validated Operating Envelope

### 3.1 Matrix Dimensions

| Type | Validated Range | Notes |
|------|----------------|-------|
| Square matrices | `64 ≤ N ≤ 4096` | Optimal for N ≥ 2048 |
| Rectangular matrices | Any aspect ratio, `M,N ≤ 46340` | Error scales with K dimension |
| Small matrices | `N ≥ 64` | N < 64: CPU overhead dominates |
| Large matrices | `N ≤ 4096` (validated), `≤ 46340` (max) | Theoretical limit 46340 |

**K-dimension error scaling**: For DGEMM with dimensions M×K × K×N, max element error scales as `O(K)×2⁻⁴⁸`. With K=10000, expect ~1.5×10⁻¹³ max error (still within DD precision class). Rectangular matrices work correctly at all aspect ratios.

### 3.2 Numerical Range

| Condition | Validated Range | Behavior Outside Range |
|-----------|----------------|------------------------|
| Exponent range | `[10⁻³⁰, 10³⁰]` | FP32 limits; denormals flushed to zero |
| Magnitude ratios | `max/min < 10³⁰` | Mixed-magnitude OK within FP32 range |
| Condition number | `κ < 10¹²` | Higher κ: error grows as `κ · N · ε` |
| NaN/Inf | Documented | Propagates through computation |

**Denormal handling**: Values `< 10⁻³⁸` (FP32 denormal threshold) are flushed to zero in DD representation.

### 3.3 Precision Scaling by Operation

| Operation | Frobenius Precision | Max Element Precision | Validated | Notes |
|-----------|---------------------|----------------------|-----------|-------|
| `ab_dgemm(A, B, C)` | `~N·10⁻¹⁵` | `~N²·⁵·10⁻²¹` | ✓ V-2 | C = A × B |
| `ab_dgemm_scaled(α, A, B, β, C)` | `~N·10⁻¹⁵` | `~N²·⁵·10⁻²¹` | ✓ BUG-3 fix | C = αAB + βC, α,β in DD |
| `ab_zgemm(Ar, Ai, Br, Bi, Cr, Ci)` | `~3N·10⁻¹⁵` | `~3N²·⁵·10⁻²¹` | ✓ QE validation | Complex: 3 real DGEMMs |
| `ab_matrix_scale(α, A)` | `~10⁻¹⁵` | `~10⁻¹⁵` | ✓ BUG-4 fix | Element-wise A ← αA |
| `ab_dsyrk(A, C)` | `~N·10⁻¹⁵` | `~N²·⁵·10⁻²¹` | ✓ Tested | C = A × Aᵀ (symmetric) |

**DEPRECATED**: `ab_zherk` is 20× slower than CPU and not recommended for production.

---

## 4. Out-of-Scope Conditions

The following conditions are **outside the validated envelope**. Results are undefined or may fail:

### 4.1 Matrix Shapes

- **No longer an issue** — Rectangular matrices of any aspect ratio now work correctly (v1.2.0)
  - Previous reports of failures were from different codebase state
  - Error scales with K dimension but remains within DD precision bounds

### 4.2 Numerical Extremes

- **Exponents outside FP32 range**: `|x| > 10³⁸` or `|x| < 10⁻⁴⁵`
  - DD uses FP32 components; extreme exponents overflow/underflow
- **Condition numbers `κ > 10¹²`**: Error bound becomes `κ · N · 2⁻⁴⁸ > 10⁻³`
  - **Workaround**: Use IEEE FP64 (`cblas_dgemm`) for ill-conditioned problems

### 4.3 Special Values

| Input | Behavior | Tested |
|-------|----------|--------|
| Denormals (`< 10⁻³⁸`) | Flushed to zero | ⚠ V-4 (planned) |
| `±Infinity` | Propagates (NaN if `∞ - ∞`) | ⚠ V-4 (planned) |
| `NaN` | Propagates | ⚠ V-4 (planned) |

### 4.4 Concurrency

- **Thread safety**: apple-bottom is **NOT thread-safe**
  - Metal command queue serializes operations
  - **Workaround**: Use one `ab_init()` per thread, or external locking

---

## 5. Performance vs. Precision Trade-offs

### 5.1 GPU Crossover Point

apple-bottom is **faster than CPU** when:

```
FLOP_count ≥ 100M  (approximately N³ ≥ 100M for DGEMM)
```

**Practical thresholds**:
- Square DGEMM: `N ≥ 2048` (1.10-1.29× speedup vs 6-thread OpenBLAS)
- Complex ZGEMM: `N ≥ 1024` (1.18-1.29× speedup)
- Iterative algorithms: **Always faster** (memory pool amortizes overhead)

**For `N < 2048`**: GPU overhead (upload/download/dispatch) exceeds compute savings. Use CPU BLAS.

### 5.2 Precision vs. Performance

| Approach | Frobenius Precision | Max Element (N=4096) | Performance (N=2048) | Use Case |
|----------|---------------------|----------------------|----------------------|----------|
| CPU FP64 (`cblas_dgemm`) | `~10⁻¹⁶` | `~10⁻¹⁶` | 1.0× (baseline) | Reference, κ > 10¹², pivoting |
| apple-bottom DD | `~10⁻¹⁵` | `~10⁻⁶` | 1.10× | Iterative solvers, DFT, MD |
| GPU FP32 (cuBLAS) | `~10⁻⁸` | `~10⁻⁸` | ~5× | Deep learning only |

**Recommendation**: Use apple-bottom when:
1. Problem requires Frobenius error `< 10⁻¹⁴` (rules out FP32)
2. Matrices are square or moderately rectangular
3. Iterative algorithm (QE, CG, eigensolver) can keep data GPU-resident
4. **Application is norm-averaged** (energies, residuals), not element-sensitive

---

## 6. Validation Traceability

### 6.1 Test Coverage

| Test ID | Description | Status | Source |
|---------|-------------|--------|--------|
| **V-2** | DGEMM convergence study (N ≤ 4096) | ✓ PASS | `tests/verification/test_convergence.c` |
| **VAL-1** | QE Si64 production validation | ✓ PASS | `docs/BENCHMARK_SUMMARY.md` |
| **BUG-3** | Scaled DGEMM precision (α,β in DD) | ✓ PASS | `tests/test_correctness.c:786` |
| **BUG-4** | Matrix scaling precision (α in DD) | ✓ PASS | `tests/test_correctness.c:819` |

### 6.2 Regression Testing

**Automated suite**: 42 tests, all passing (v1.0.2-bugfix)
- **Correctness**: 37 tests (API, edge cases, vs-Accelerate)
- **Precision**: 1 test (Frobenius error < 10⁻¹⁴ for N ∈ {64..2048})
- **Regression**: 5 tests (BUG-1 through BUG-6 fixes)

**CI**: Run `make test && make test-verification` before each release

### 6.3 Configuration Management

| Item | Value | Notes |
|------|-------|-------|
| **Validated version** | v1.0.2-bugfix | Git tag `v1.0.2-bugfix` (SHA: 700934f) |
| **Compiler** | clang 15.0.0 (Apple) | `-O3 -std=c11` |
| **Metal SDK** | macOS 15.0+ SDK (Xcode 16+) | **`MTLMathModeSafe` is CRITICAL** - without it precision degrades from ~10⁻¹⁵ to ~10⁻⁸ |
| **Test system** | M2 Max, 64GB RAM | 12 CPU cores, 38 GPU cores |

**CRITICAL**: `MTLMathModeSafe` is a hard requirement, not optional. Without it (SDK < 15.0), DD precision degrades by 8 orders of magnitude (~10⁻⁸ instead of ~10⁻¹⁵). Do not modify validated code without re-running V&V tests. Tag validated states in Git.

---

## 7. Usage Guidance for Production

### 7.1 Production Deployment Recommendations

For scientific computing on Apple Silicon:

**✓ USE apple-bottom for**:
- DFT, molecular dynamics, FEM iterative solvers (norm-averaged quantities)
- Matrices `N ≥ 1024`, condition number `κ < 10⁶`
- Applications requiring `10⁻¹⁴` to `10⁻¹⁰` Frobenius convergence

**✗ DO NOT USE for**:
- Algorithms requiring element-wise accuracy `< 10⁻⁶` (e.g., pivoting, eigensolvers)
- Rectangular matrices with aspect ratio > 10:1
- Ill-conditioned systems (`κ > 10¹²`)
- Matrices `N < 512` (CPU faster due to overhead)
- Thread-unsafe contexts (add external locking)

**Validation requirement**: For mission-critical calculations, perform cross-verification:
1. Run apple-bottom simulation
2. Run CPU FP64 reference (`cblas_dgemm`) on 10% of cases
3. Verify agreement to `< 10⁻¹³` Frobenius error
4. Document discrepancies > 10⁻¹² in anomaly log

### 7.2 Error Budgeting

For multi-step simulations (e.g., 100-iteration SCF):

```
Total Frobenius error ≈ √(n_iterations) · single_step_error
                       ≈ √100 · (N · 10⁻¹⁵)
                       ≈ 10 · N · 10⁻¹⁵
                       ≈ N · 10⁻¹⁴
```

**Example** (N=2048, 100 SCF iterations):
```
Expected accumulated error ≈ 2048 · 10⁻¹⁴ ≈ 2×10⁻¹¹
```

This is well within typical DFT convergence criteria (`10⁻¹⁰ Ry`).

### 7.3 Fallback Strategy

If precision is insufficient:
1. **First**: Verify problem is well-conditioned (`κ < 10⁶`)
2. **Second**: Check if application is element-sensitive (max error) vs. norm-averaged
3. **Third**: Switch to CPU FP64 (`cblas_dgemm`) for that operation only
4. **Last resort**: Implement extended precision (triple-double) — contact maintainers

---

## 8. References

### 8.1 Numerical Analysis

- Dekker, T. J. (1971). "A floating-point technique for extending the available precision." *Numerische Mathematik*, 18(3), 224-242.
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM. Chapter 3: "Rounding Error Analysis of Inner and Outer Products."

### 8.2 Validation Standards

- ASME V&V 10-2006: *Guide for Verification and Validation in Computational Solid Mechanics*
- NASA-STD-7009A: *Standard for Models and Simulations*

### 8.3 apple-bottom Documentation

- `README.md`: Quick start, performance benchmarks
- `docs/INTEGRATION.md`: C/Fortran/Python/QE integration
- `docs/vv/VV_REPORT.md`: Full V&V documentation

---

## 9. Document Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-03-31 | Initial validated precision envelope | Grant Heileman |

---

## 10. Contact

**Maintainer**: Grant Heileman
**Repository**: https://github.com/grantdh/apple-bottom
**Issues**: File at GitHub Issues for production support

**For production validation support**: Provide (1) test case description, (2) measured error, (3) expected error from this document, (4) git SHA of your apple-bottom build.
