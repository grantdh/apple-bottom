# Lessons Learned: FP64 Emulation on Apple Silicon

**Project:** apple-bottom — Metal-native BLAS for Quantum ESPRESSO  
**Author:** Grant Heileman, UNM ECE  
**Date:** March 2026  
**Hardware:** Apple M2 Max (30-core GPU, 96GB unified memory)

---

## Executive Summary

This document captures critical technical discoveries from building FP64-emulated BLAS kernels on Apple Silicon. The key findings:

1. **Double-float (FP32×2) DGEMM beats AMX by 12-23% at sizes ≥1024**
2. **Gauss 3-multiply ZGEMM beats AMX by 23-43% at sizes ≥1024**
3. **DSYRK/ZHERK kernel wins but conversion overhead makes AMX faster overall**
4. **Triple-float (FP32×3) DGEMM achieves faithfully-rounded FP64: 99.5% correctly rounded, max 1 ULP, verified across 279K elements**
5. **Ziv rounding certification fails for GEMM — information-theoretic barrier, not bound-tightness**
6. **DD precision floor is accumulation-dominated, not input-dominated**
7. **Swift has numerous gotchas that will crash or silently corrupt**

---

## 1. Double-Float Arithmetic (FP32×2)

### 1.1 Precision Achieved

| Method | Mantissa Bits | Relative Error | Use Case |
|--------|---------------|----------------|----------|
| FP32 | 24 | ~10⁻⁷ | Graphics, ML inference |
| **FP32×2 (DD)** | **~48** | **~10⁻¹⁵** | **Scientific computing** ✓ |
| FP64 | 53 | ~10⁻¹⁶ | Full double precision |

**Key insight:** 48 bits is sufficient for DFT calculations where the SCF loop self-corrects inner solver errors.

### 1.2 CRITICAL: mathMode = .safe is MANDATORY

```swift
// CORRECT: mathMode = .safe is MANDATORY
let opts = MTLCompileOptions()
opts.mathMode = .safe  // ← CRITICAL: .fast breaks error-free transforms

// WRONG: This destroys DD precision
opts.mathMode = .fast  // ← NEVER use for DD arithmetic
```

**Why:** Fast math enables FMA fusion and reassociation that violates the exact algebraic identities required by error-free transforms (twoSum, twoProduct).

### 1.3 Core Primitives

```metal
// Error-free addition: s + e = a + b exactly
inline void twoSum(float a, float b, thread float &s, thread float &e) {
    s = a + b;
    float v = s - a;
    e = (a - (s - v)) + (b - v);  // 6 FLOPs
}

// Error-free multiplication via FMA
inline void twoProduct(float a, float b, thread float &p, thread float &e) {
    p = a * b;
    e = fma(a, b, -p);  // 2 FLOPs — REQUIRES mathMode = .safe
}
```

### 1.4 DD Storage Layout

**Always use interleaved struct layout:**

```metal
struct dd {
    float hi;  // High-order bits (main value)
    float lo;  // Low-order bits (error correction)
};

// ✓ CORRECT: Interleaved
device const dd *A [[buffer(0)]];

// ✗ WRONG: Separate buffers (3-5× slower)
device const float *A_hi [[buffer(0)]];
device const float *A_lo [[buffer(1)]];
```

**Why:** Interleaved layout doubles memory bandwidth utilization and enables coalesced access patterns.

---

## 2. Register Blocking: 4×4 is Optimal

### 2.1 Block Size Comparison (M2 Max)

| Block | Accumulators | Registers | Performance | Notes |
|-------|-------------|-----------|-------------|-------|
| 2×2 | 4 DD (8 floats) | ~36 | 560 GFLOP/s | Baseline |
| **4×4** | **16 DD (32 floats)** | **~68** | **640 GFLOP/s** | **Optimal** ✓ |
| 6×6 | 36 DD (72 floats) | ~116 | 215 GFLOP/s | Poor threadgroup fit |
| 8×4 | 32 DD (64 floats) | ~108 | 195 GFLOP/s | Poor occupancy |
| 8×8 | 64 DD (128 floats) | ~180 | Spills | Exceeds register limit |

### 2.2 Why 4×4 Wins

1. **256 threads per threadgroup** — Matches Apple's SIMD group architecture
2. **64×64 block dimensions** — Power-of-2 alignment for memory coalescing
3. **~68 registers** — Well under the ~192 limit, no spilling
4. **TK=16** — Larger K-tile means more compute per barrier sync

### 2.3 Production Configuration

```metal
#define BM 64       // Block rows
#define BN 64       // Block cols
#define TM 4        // Thread tile rows
#define TN 4        // Thread tile cols
#define TILE_K 16   // K-dimension tile
#define NUM_THREADS ((BM/TM) * (BN/TN))  // = 256
```

---

## 3. simdgroup_matrix is NOT Compatible with DD

### 3.1 The Problem

`simdgroup_multiply_accumulate` sums over the K dimension **internally in FP32**. By the time your code sees the result, precision is already lost.

```metal
// This gives FP32 precision (~10⁻⁷), NOT DD precision
simdgroup_matrix<float, 8, 8> acc;
simdgroup_multiply_accumulate(acc, tileA, tileB, acc);
// DD arithmetic applied HERE cannot recover precision lost INSIDE the hardware
```

### 3.2 Experimental Results

| Approach | Speed | Precision | Verdict |
|----------|-------|-----------|---------|
| simdgroup_matrix + DD post-process | 2.7× faster | ~10⁻⁷ (FP32) | ✗ Not viable |
| Ozaki scheme + simdgroup_matrix | 2× faster | ~10⁻⁷ (FP32) | ✗ Same problem |
| **Register-blocked DD** | **1× baseline** | **~10⁻¹⁵** | **✓ Required** |

**Conclusion:** For true ~48-bit precision, register-blocked scalar arithmetic is the only viable GPU path.

---

## 4. Performance Results

### 4.1 DGEMM (ex10)

| Size | DD GFLOP/s | AMX GFLOP/s | Winner | Margin |
|------|------------|-------------|--------|--------|
| 512 | 183 | 444 | AMX | -59% |
| **1024** | **551** | **493** | **DD ✓** | **+12%** |
| **2048** | **637** | **517** | **DD ✓** | **+23%** |
| 3072 | 648 | 607 | DD ✓ | +7% |

### 4.2 ZGEMM (ex11c Gauss 3-multiply)

| Size | GPU ms | AMX ms | Winner | Margin |
|------|--------|--------|--------|--------|
| 512 | 2.3 | 2.2 | AMX | -5% |
| **1024** | **12.3** | **15.1** | **GPU ✓** | **+23%** |
| **2048** | **84.9** | **114.6** | **GPU ✓** | **+35%** |
| **3072** | **284** | **407** | **GPU ✓** | **+43%** |

### 4.3 DSYRK (ex12f analysis)

| Size | GPU Kernel | AMX | Kernel Winner | Conversion Overhead |
|------|------------|-----|---------------|---------------------|
| 2048 | 16.3ms | 20.2ms | GPU +24% | 57.6ms (kills total) |
| 3072 | 49.8ms | 58.7ms | GPU +18% | 106.7ms |
| 4096 | 112.9ms | 157.5ms | GPU +40% | 219.2ms |

**Key insight:** GPU kernel wins by 18-40%, but FP64↔DD conversion takes 3-4× longer than the compute. Route DSYRK/ZHERK to AMX.

---

## 5. ZGEMM: Gauss 3-Multiply Algorithm

### 5.1 Why Native Complex DD Fails

Native complex DD requires ~100 scalar FP32 ops per complex multiply-add vs AMX's ~4. The GPU cannot compensate for this arithmetic intensity disadvantage.

### 5.2 Gauss Algorithm (25% FLOPs reduction)

```
K1 = Ar × Br      (real × real)
K2 = Ai × Bi      (imag × imag)
K3 = (Ar+Ai) × (Br+Bi)

Cr = K1 - K2
Ci = K3 - K1 - K2
```

Only 3 DGEMMs instead of 4 for the equivalent complex operation.

### 5.3 Implementation Requirements

- Split complex matrices into separate real/imaginary arrays
- Run 3× DD-DGEMM kernel invocations
- Combine results with element-wise DD arithmetic
- Each DGEMM reuses the optimized 4×4 kernel

---

## 6. DSYRK/ZHERK: Conversion Overhead Problem

### 6.1 The Issue

| Component | Time at 2048×2048 |
|-----------|-------------------|
| GPU Kernel | 16.3ms |
| FP64→DD Conversion | 57.6ms |
| **Total** | **73.9ms** |
| AMX Reference | 20.2ms |

The kernel beats AMX by 24%, but total time is 3.7× worse due to conversion.

### 6.2 Why Conversion is Expensive

1. **Data format mismatch:** DSYRK input is FP64 column-major; DD kernel needs interleaved row-major
2. **Double touch:** Must transpose AND convert FP64→Float×2
3. **No vectorization:** Per-element `Float(d)` + `Float(d - Double(hi))` cannot be SIMD'd
4. **Cache thrashing:** Transpose access pattern is cache-unfriendly

### 6.3 Solutions Attempted

| Approach | Result |
|----------|--------|
| vDSP vectorized conversion | No improvement (transpose still slow) |
| GPU conversion kernel | Metal lacks FP64; bit manipulation loses precision |
| Buffer pooling | Reduces alloc overhead but conversion still dominates |
| Unsafe pointers | Eliminates bounds checking but FP conversion still slow |

### 6.4 Conclusion

Route DSYRK/ZHERK to AMX. They're only ~10% of QE runtime.

---

## 7. Swift Gotchas (CRITICAL)

### 7.1 String Formatting Crash

```swift
// ✗ CRASHES: %s with Swift String
let name = "test"
String(format: "%s", name)  // EXC_BAD_ACCESS

// ✓ CORRECT: Use %@ with NSString cast
String(format: "%@", name as NSString)

// ✓ BETTER: Use string interpolation
"Result: \(name)"
```

**Why:** `%s` expects a C string pointer, not a Swift String object.

### 7.2 DD Struct Initialization

```swift
// ✗ ERROR: If only init(_ d: Double) exists
let zero = DD(hi: 0, lo: 0)  // "extra argument 'lo' in call"

// ✓ CORRECT: Provide explicit initializer
struct DD {
    var hi: Float
    var lo: Float
    
    init(_ d: Double) {
        hi = Float(d)
        lo = Float(d - Double(hi))
    }
    
    init(hi: Float, lo: Float) {
        self.hi = hi
        self.lo = lo
    }
}
```

### 7.3 Accelerate Deprecation Warnings

```bash
# ✗ WARNING: Old API deprecated
swiftc -O -o ex12 ex12.swift -framework Accelerate

# ✓ CORRECT: Use new LAPACK interface
swiftc -O -Xcc -DACCELERATE_NEW_LAPACK -o ex12 ex12.swift -framework Accelerate
```

The `-DACCELERATE_NEW_LAPACK` flag opts into the new API, not just suppresses warnings.

### 7.4 Unsafe Pointer Performance

```swift
// ✗ SLOW: Array subscript with bounds checking
for i in 0..<N {
    C[i] = A[i] + B[i]  // Each access has bounds check
}

// ✓ FAST: UnsafePointer eliminates bounds checking
A.withUnsafeBufferPointer { aPtr in
    B.withUnsafeBufferPointer { bPtr in
        C.withUnsafeMutableBufferPointer { cPtr in
            for i in 0..<N {
                cPtr[i] = aPtr[i] + bPtr[i]
            }
        }
    }
}
```

### 7.5 Metal Buffer Binding

```swift
// ✗ SLOWER: bindMemory allocates
let ptr = buffer.contents().bindMemory(to: DD.self, capacity: N)

// ✓ FASTER: assumingMemoryBound skips allocation
let ptr = buffer.contents().assumingMemoryBound(to: DD.self)
```

Use `assumingMemoryBound` when you're certain the buffer layout matches.

---

## 8. Metal-Specific Gotchas

### 8.1 Reserved Keywords

```metal
// ✗ WRONG: 'half' is a Metal type
float half = 0.5f;

// ✓ CORRECT: Use a different name
float oneHalf = 0.5f;
```

### 8.2 Integer Type Mixing

```metal
// ✗ WRONG: Ambiguous min() with int/uint
uint x = min(N, maxTG);

// ✓ CORRECT: Explicit cast
uint x = min(N, uint(maxTG));
```

### 8.3 Metal Lacks FP64 Operations

Metal supports FP64 **storage** but not **arithmetic**. Any attempt to do FP64 math via bit manipulation results in ~10⁻⁸ precision loss due to subnormal handling and rounding differences.

```metal
// ✗ FAILS: Bit manipulation loses precision
ulong bits = as_type<ulong>(doubleVal);
// ... bit manipulation ...
float hi = as_type<float>(uint(bits >> 32));  // ~10⁻⁸ error

// ✓ CORRECT: Convert on CPU in Swift
let hi = Float(doubleVal)
let lo = Float(doubleVal - Double(hi))  // Full precision
```

---

## 9. Routing Strategy for DYLD Interposition

### 9.1 Final Decision Tree

```c
void metal_blas_router(const char* routine, int M, int N, int K, ...) {
    int64_t size = max(M, max(N, K));
    
    if (strcmp(routine, "dgemm_") == 0 && size >= 1024) {
        metal_dd_dgemm_4x4(...);  // GPU: +12-23%
    } else if (strcmp(routine, "zgemm_") == 0 && size >= 1024) {
        metal_gauss_zgemm(...);   // GPU: +23-43%
    } else {
        // CPU fallback for:
        // - Small matrices (<1024)
        // - DSYRK/ZHERK (conversion overhead)
        // - All other BLAS routines
        accelerate_blas_call(...);
    }
}
```

### 9.2 QE/Yambo Coverage Analysis

| Operation | % Runtime | Route | Expected Speedup |
|-----------|-----------|-------|------------------|
| DGEMM | ~40% | GPU ≥1024 | +12-23% |
| ZGEMM | ~35% | GPU ≥1024 | +23-43% |
| DSYRK | ~5-10% | AMX | 0% |
| ZHERK | ~5-10% | AMX | 0% |
| Other | ~10% | AMX | 0% |

**Net expected speedup:** ~15-25% overall for large DFT calculations.

---

## 10. Error Measurement

### 10.1 Use Frobenius Norm, Not Element-wise

```swift
// ✗ WRONG: Element-wise relative error blows up for near-zero values
let relErr = abs(computed - reference) / abs(reference)
// If reference ≈ 0, relErr → ∞ even for tiny absolute error

// ✓ CORRECT: Frobenius norm-based relative error
var maxErr = 0.0, normSq = 0.0
for i in 0..<N*N {
    maxErr = max(maxErr, abs(computed[i] - reference[i]))
    normSq += reference[i] * reference[i]
}
let relErr = maxErr / sqrt(normSq)  // Stable even with near-zero elements
```

### 10.2 Acceptable Thresholds

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Relative error (DD) | < 10⁻¹⁴ | ~48-bit precision |
| Relative error (FP64) | < 10⁻¹⁵ | Full double precision |
| Max absolute error | Context-dependent | Check matrix scale |

---

## Appendix: Exercise Index

| Exercise | Topic | Status | Key Learning |
|----------|-------|--------|--------------|
| ex01-02 | Complex arithmetic | ✓ Done | SIMD2<Float> layout |
| ex03-04 | FP32 SGEMM/CGEMM | ✓ Done | Register blocking basics |
| ex05-06 | Stockham FFT | ✓ Done | Batched 3D transforms |
| ex07 | DD primitives | ✓ Done | twoSum, twoProduct, mathMode |
| ex08 | DD-DGEMM (2×2) | ✓ Done | Baseline DD kernel |
| ex08b | simdgroup DD | ✗ Failed | FP32 precision only |
| ex08c | Ozaki scheme | ✗ Failed | Same problem |
| ex09a-e | Crossover analysis | ✓ Done | 4×4 optimal, 1024+ crossover |
| ex10 | Production DD-DGEMM | ✓ Done | Final 4×4 kernel |
| ex11 | Native DD-ZGEMM | ✗ Failed | ~100 ops/element too slow |
| ex11b | Split ZGEMM (4×DGEMM) | ✓ Done | Works but wasteful |
| ex11c | Gauss ZGEMM (3×DGEMM) | ✓ Done | +23-43% vs AMX |
| ex12 | DD-DSYRK | Analysis | Kernel wins but conversion kills |
| ex12f | DSYRK benchmarking | ✓ Done | Definitive overhead analysis |
| ex13 | DD-ZHERK | Analysis | Same as DSYRK |
| ex14 | DYLD Interposition | Planned | Drop-in library |
| ex15d | TD proof + CPU validation | ✓ Done | Lemma 1 + 2 + Theorem validated |
| ex16 | TD-DGEMM GPU kernel | ✓ Done | 99.5% correct, 148 GFLOP/s |
| ex17e | Ziv certification (final) | ✗ Failed | Information-theoretic barrier |
| ex18 | DD + correction SGEMM | ✗ Failed | Accumulation error dominates |
| ex19 | Comprehensive validation | ✓ Done | Faithful rounding, C₀, seq. FP64 |
| ex19b | κ-binned ULP analysis | ✓ Done | Max ULP = 1 across all κ bins |

---

## 11. Triple-Float Arithmetic (FP32×3) — TD-DGEMM

### 11.1 The Key Insight: Lossless FP64 → TD Conversion

FP64 has 53 mantissa bits. Three FP32 values provide 3 × 24 = 72 bits. The conversion `h = float(d), m = float(d - h), l = float(d - h - m)` captures all 53 bits exactly because the residuals have 29, then 5 significant bits — both fit in FP32's 24-bit mantissa without rounding. This is **Lemma 1**, validated on 100,000 random values.

**This eliminates the 5-bit truncation that limits DD to ~48-bit precision.** DD loses 5 bits at input conversion and can never recover them. TD loses zero bits.

### 11.2 The 19-Bit Margin

TD provides ~72 bits of mantissa for a 53-bit target: 19 bits of headroom. This margin means per-step errors of O(u³) = O(2⁻⁷²) are negligible relative to the FP64 rounding unit u₆₄ = 2⁻⁵³. The ratio 2⁻⁵³/2⁻⁷² = 2¹⁹ ≈ 524,288 — over half a million times more precision than needed for a single rounding decision.

### 11.3 TD: 2×2 Blocking Beats 4×4

Unlike DD (where 4×4 is optimal at ~68 registers), TD's 4×4 requires ~97 registers, pushing occupancy down. The 2×2 variant at ~49 registers achieves 148 GFLOP/s vs 108 for 4×4. **More registers per thread = fewer concurrent threads = less latency hiding.**

```
TD register budget:
  2×2: 4 acc × 3 + 2 Av × 3 + 2 Bv × 3 + ~25 temps ≈ 49 registers → 148 GFLOP/s ✓
  4×4: 16 acc × 3 + 4 Av × 3 + 4 Bv × 3 + ~25 temps ≈ 97 registers → 108 GFLOP/s ✗
```

### 11.4 TD FMA Cost: ~65 FLOPs

The td_fma_full operation (acc += a × b) requires:
- 3 TwoProduct calls (6 FLOPs) for hi×hi, hi×md, md×hi
- 3 approximate multiplies (3 FLOPs) for lower cross-terms
- 6 TwoSum calls (36 FLOPs) for the exact accumulation chain
- 7 additions (7 FLOPs) for lo collection
- 1 td_renorm (15 FLOPs) for output normalization

Total: ~65 FLOPs vs DD's ~25 FLOPs = 2.6× more work per element. Measured throughput ratio is 640/148 = 4.3× slower (additional overhead from 3-wide memory access and TILE_K reduction from 16 to 8).

### 11.5 CRITICAL: The Accumulation Analysis Gap

**Proposition 1 (per-step bound) is correct:** C₀ = 19 per td_fma_full step.

**Proposition 2 (accumulation) has a documented gap.** The linear model assumes max(|aₖbₖ|, |accₖ|) ≤ S/K, but the running accumulator grows linearly, making the true dependence quadratic in K. Measured effective C₀:

| K | Analytical C₀ | Measured C₀ | Pattern |
|---|---|---|---|
| 128 | 19 | 8,181 | ≈ 2¹⁹/128 = 2¹² |
| 256 | 19 | 4,095 | ≈ 2¹⁹/256 = 2¹¹ |
| 512 | 19 | 2,041 | ≈ 2¹⁹/512 = 2¹⁰ |

**Lesson:** Present the per-step bound as a Proposition with proof sketch, then honestly report the accumulation gap with empirical constants. The faithful-rounding claim stands on exhaustive verification (279K elements), independent of the analytical bound.

---

## 12. Negative Result: Ziv Certification Fails for GEMM

### 12.1 What Ziv's Strategy Is

Ziv (1991) is the foundation of correctly-rounded math libraries. Compute at precision p, check if the error interval fits within one rounding basin. If ambiguous, recompute at 2p. Works brilliantly for elementary functions (sin, exp, log) where error-to-ULP ratio is ~10⁻⁴.

### 12.2 Why It Fails for GEMM

Five experiments (ex17–17e) all produced exactly ~346 false certifications per 65,536 elements (0.53%), invariant across all bound formulations and safety factors. The root cause is information-theoretic:

When TD error pushes a value past a FP64 rounding boundary, the rounded result lands in the neighboring basin — potentially far from the boundary. No local test on the computed value alone can distinguish:
- A correctly-rounded element at position x
- A wrongly-rounded element that landed at position x after crossing the boundary

For elementary functions, this regime is vanishingly rare (error/ULP ~ 10⁻⁴). For TD-DGEMM, error/ULP ~ 10⁻¹, leaving 0.5% of elements in the undetectable wrong-side regime.

### 12.3 The Structural Connection

The false certification rate (0.53%) equals the complement of the correct-rounding rate (1 − 0.995 = 0.5%). Same population, identified from two independent directions. This closes the loop: Ziv cannot improve TD-DGEMM because the elements it fails to certify are exactly the elements that are incorrectly rounded.

---

## 13. Negative Result: DD + Correction SGEMM Is Ineffective

**Hypothesis:** DD's ~170 ULP floor comes from 5-bit input truncation (48 vs 53 bits). Adding the missing residuals via two cheap FP32 SGEMMs should close the gap.

**Test:** Compute ΔC = A_lo5 × B_hi + A_hi × B_lo5 and add to DD result.

**Result:** Zero improvement. Identical ULP distributions.

**Root cause:** DD's dominant error is FP32 rounding in the accumulation chain (five rounding ops per dd_fma), not input truncation. Over K steps, accumulated arithmetic error O(K × u² × S) swamps the correction of magnitude O(K × u² × S × 2⁻⁵).

**Lesson:** The DD–TD gap is structural. You cannot patch DD to approach TD precision without replacing the inner loop arithmetic. This is why TD-DGEMM exists as a separate kernel rather than an enhancement of DD-DGEMM.

---

## 14. Accelerate and IEEE 754: Setting the Record Straight

### 14.1 Accelerate Is IEEE 754 Compliant

Every individual FP64 operation inside Accelerate's DGEMM is correctly rounded per IEEE 754. The standard requires this for add, multiply, FMA, sqrt — not for compound operations like dot products.

### 14.2 Accelerate's ~20% Correct-Rounding Rate Is Normal

Sequential FP64 dot products (naive `for k { acc += a*b }`) achieve the same ~20% rate as Accelerate's blocked algorithm (ex19). The blocked accumulation doesn't meaningfully degrade accuracy — the ~20% rate is inherent to FP64 dot products at K=128–512.

### 14.3 Accelerate Is Reproducible on M2 Max

100 runs of the same 256×256 DGEMM produced bit-identical output (ex19). However, this is empirical — reproducibility may vary across hardware configurations or OS versions. TD-DGEMM's reproducibility is guaranteed by construction.

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DD-GEMM QUICK REFERENCE                              │
├─────────────────────────────────────────────────────────────────────────┤
│ Metal compile:     opts.mathMode = .safe  // MANDATORY                  │
│ Block size:        BM=64, BN=64, TM=4, TN=4, TK=16, 256 threads         │
│ DD struct:         struct dd { float hi; float lo; };  // interleaved   │
│ Crossover:         Use GPU when max(M,N,K) ≥ 1024                       │
│ ZGEMM:             Gauss 3-multiply (3× DGEMM), not native complex      │
│ DSYRK/ZHERK:       Route to AMX (conversion overhead)                   │
│ Precision:         ~10⁻¹⁵ relative error (48-bit effective)            │
│ Correct rounding:  ~0.5% of elements                                    │
│ Swift strings:     Use %@ not %s, or use string interpolation           │
│ Accelerate:        -Xcc -DACCELERATE_NEW_LAPACK for production          │
│ Error metric:      Frobenius norm, not element-wise                     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    TD-GEMM QUICK REFERENCE                              │
├─────────────────────────────────────────────────────────────────────────┤
│ Metal compile:     opts.mathMode = .safe  // MANDATORY                  │
│ Block size:        BM=32, BN=32, TM=2, TN=2, TK=8, 256 threads          │
│ TD struct:         struct td { float hi; float md; float lo; };         │
│ Conversion:        Lossless FP64→TD via Lemma 1 (24+24+5 = 53 bits)    │
│ Precision:         ~10⁻²² relative error (72-bit effective)            │
│ Correct rounding:  99.5% of elements, max 1 ULP (faithfully rounded)   │
│ Faithful:          Verified: r̂ ∈ {⌊c⌋₆₄, ⌈c⌉₆₄} for 1457/1457       │
│ Reproducible:      Yes, by construction (sequential accumulation)       │
│ FMA cost:          ~65 FLOPs per td_fma_full vs ~25 for dd_fma         │
│ Throughput:        148 GFLOP/s (2×2 blocking, M2 Max)                   │
│ Exponent range:    |inputs| < 10¹⁸ (FP32 exponent limit)              │
│ Ziv certification: DOES NOT WORK for GEMM (ex17e)                       │
└─────────────────────────────────────────────────────────────────────────┘
```
