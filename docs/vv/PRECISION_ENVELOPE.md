# Precision Envelope — apple-bottom v1.0.2

**Document Status**: Validated for v1.0.2 (2026-03-31); revised since (§9)
**Validation Baseline**: Git tag `v1.0.2-bugfix` (SHA: f3e2fc1)
**Test System**: Apple M2 Max, macOS 14.7, Metal 3

> **Status (Revision 1.4, 2026-10-30).** Written for v1.0.2 and kept as its
> record, with corrections: the per-operation bounds of §1.2 (Revision 1.4:
> `dd_mul` ≤ 6.000001u², the fused `dd_fma` ≤ 13u²|ab| + 3u²|c|, both proven;
> GEMM ≤ (3K+13)u²|A||B| componentwise, §1.3), the
> summary's guarantee restated as the target the measurements meet, notes
> on the max-element measure (§2.3) and on threads (§4.4), and §5.1 pointed
> at 1.5.0. The exponent-range text (§3.2, §4.2, §4.3) describes 1.5.0.

> **Correction (2026-10-14).** The Quantum ESPRESSO result of §2.4 came from
> runs that did not use the GPU (versions before 1.4.0 needed an `ab_init()`
> that QE never called), and its timing compared a serial with an MPI QE
> build; it is withdrawn and replaced by VAL-1 revision 2. The performance
> thresholds of §5.1 are replaced by the 1.4.0 measurements. The precision
> analysis and the V-2 convergence data are unaffected. Test counts are
> those of v1.0.2; see `tests/README.md` for the current suite.

---

## Executive Summary

**apple-bottom** provides FP64-class precision through double-float (DD) arithmetic on Apple Silicon GPUs. This document defines the validated precision envelope—the operating conditions under which accuracy guarantees hold.

### Stated Guarantee

For matrix multiplication `C = A × B` where:
- Matrix dimensions: `64 ≤ N ≤ 4096` (square matrices, validated range)
- Matrix entries: finite FP64 values. The BLAS ABI scales rows and columns
  far from 1 by powers of two, and runs on Accelerate what that cannot keep
  in DD's range. A native `ABMatrix` keeps its nonzero entries within a span
  of 2⁷⁸ under one power-of-two scale; `LIMITATIONS.md#exponent-range`.
- Condition number: `κ(A), κ(B) < 10¹²`

~~apple-bottom guarantees:~~ The V-2 study checks each measured size against:

```
‖C_gpu - C_ref‖_F / ‖C_ref‖_F  <  N · 2⁻⁴⁸
```

where `C_ref` is the IEEE FP64 reference result and `2⁻⁴⁸` is the DD mantissa precision.

*Corrected in Revision 1.3: this is a target the measurements meet, not a guarantee for every input. A worst-case bound that holds for every input takes the componentwise form |C_gpu − C|_ij ≤ c·K·u²·(|A||B|)_ij (Higham 2002, Ch. 3, with c from the per-operation envelope of §1.2); a bound relative to ‖C‖ can fail where cancellation makes C small.*

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

**Error bound**: ~~Single DD operation introduces `≤ 2⁻⁴⁸` relative error~~ a DD product (`dd_mul`) has relative error at most 6.000001u² (u² = 2⁻⁴⁸), and the fused multiply-add (`dd_fma`) has absolute error at most (13+43u)u²|a·b| + (3+5u)u²|c| (§1.2; corrected in Revision 1.3, and again in Revision 1.4: the 5u² and 8u² figures were not proven for this code)

### 1.2 Error-Free Transformations

DD operations use Dekker/Knuth algorithms:

| Operation | Algorithm | FLOPs | Error Bound |
|-----------|-----------|-------|-------------|
| FP32 addition (error-free transform) | `twoSum` | 6 | `0` (exact: `s + e = a + b`) |
| FP32 multiplication (error-free transform) | `twoProduct` (FMA) | 2 | `0` (exact: `p + e = a × b`) |
| DD multiplication | `dd_mul` | — | `≤ 6.000001u²` relative (proven below; ≤ 4.33u² observed). Under `DWTIMESDW3=1`: `< 4u²` (Muller–Rideau 2022, Theorem 2.8) |
| DD fused multiply-add | `dd_fma` | 18 | `≤ (13+43u)u²·|a·b| + (3+5u)u²·|c|` absolute (proven below; Revision 1.4 replaced the unproven `≲ 8u²` envelope) |

where `u = 2⁻²⁴` is the FP32 unit roundoff (so `ε_mach = 2⁻²³` under the ulp convention used by IEEE 754; `u²  = 2⁻⁴⁸` is the effective DD unit roundoff). The `twoSum` and `twoProduct` algorithms are *error-free transforms*: their outputs represent the exact FP32 operation result as an unevaluated sum with no rounding error, under the standard no-overflow / no-subnormal-input assumptions. ~~The `dd_fma` bound is the per-operation DD-level bound from §1.2 and applies to both `dd_mul` and the compound multiply-add.~~ `dd_mul` and `dd_fma` have the separate bounds proven below (Revision 1.4).

~~Cross-term products in `dd_mul` and `dd_fma` use nested FMA chains (JMP 2017 Algorithm 11, "DWTimesDW2") to reduce rounding from 4 truncations to 2 per operation. The relative error of `dd_mul` is bounded by `5u² / (1 + u)² < 5u²` where `u = 2⁻²⁴`, per Muller-Rideau 2022 Theorem 2.7 (tightened from JMP 2017's original `< 6u²` bound via formal Coq proof).~~

**The default product is not Algorithm 11 (Revision 1.4).** Revisions 1.2 and 1.3 credited `dd_mul` with the `5u²/(1+u)²` bound of Muller–Rideau 2022, Theorem 2.7. That theorem is for DWTimesDW2 (Joldes–Muller–Popescu 2017, Algorithm 11), which rounds the low part three times:

```
(ch, cl1) = 2Prod(xh, yh);  tl = RN(xh·yl);  cl2 = RN(tl + xl·yh);  cl3 = RN(cl1 + cl2)
```

apple-bottom's default rounds it twice, feeding `cl1` into the first FMA:

```
(p, e) = 2Prod(xh, yh);  t = RN(xl·yh + e);  f = RN(xh·yl + t)
```

This is a different rounding sequence, so the theorem does not cover it. The opt-in `DWTIMESDW3=1` product is Algorithm 12 line for line, so Theorem 2.8 (`(4u² + u³/2)/(1+u)² < 4u²`) does apply to it. The library makes no claim for `dd_fma` under that option.

**Proven bounds for the default build.** Assume no underflow or overflow (§3.2, §4.2) and round-to-nearest as written. Write the fused step `dd_fma(a, b, c)` as

```
(p, e) = 2Prod(ah, bh);  t = RN(al·bh + e);  f = RN(ah·bl + t)      product P = p + f
(s, g) = 2Sum(p, ch);    v = RN(f + cl);      w = RN(g + v)
(rh, rl) = Fast2Sum(s, w)
```

`dd_mul` is the first line followed by `Fast2Sum(p, f)`. Let `A = |ah·bh|` and `C = |ch|`. Use `|RN(z) − z| ≤ u|z|` and `|xl| ≤ u|xh|`.

1. **Product.** `|e| ≤ uA`. The rounding of `t` costs at most `2u²A`, and `|t| ≤ 2u(1+u)A`. The rounding of `f` costs at most `(3u² + 2u³)A`, and `|f| ≤ (1+u)(3u+2u²)A`. The dropped `al·bl` is at most `u²A`. So `|P − ab| ≤ (6u² + 2u³)A ≤ 6.000001u²|a||b|`. `|f|` is far below `|p|`, so the `Fast2Sum` that ends `dd_mul` is exact, and `dd_mul`'s relative error is at most 6.000001u².
2. **Sum.** `|g| ≤ u((1+u)A + C)`. With `L = (1+u)(3u+2u²)A + uC ≥ |f + cl|`, rounding `v` costs at most `uL` and rounding `w` at most `u(|g| + (1+u)L)`. If the final `Fast2Sum` is exact, `rh + rl − (ab + c) = (P − ab) + δv + δw`. Then `A ≤ |a||b|/(1−u)²` and `C ≤ |c|/(1−u)` give

   ```
   |rh + rl − (ab + c)| ≤ (13 + 43u)·u²·|a||b| + (3 + 5u)·u²·|c|
   ```

   The constants were evaluated in exact rational arithmetic and rounded up. To first order they come from: the product (6), rounding `v` (3 + 1), and rounding `w` (4 + 2).
3. **The final `Fast2Sum` is always exact.** Dekker's condition is that `s = Ms·2^es` and `w = Mw·2^ew` with `|Ms|, |Mw| ≤ 2²⁴ − 1` and `es ≥ ew`. It holds whenever `|s| ≥ |w|`. Suppose instead `|s| < |w|`, and let `M = max(|p|, C)`.
   - The bounds above give `|w| ≤ 6.000002uM < 2⁻²¹M`, so `|p + ch| < 2⁻²¹M/(1−u)`.
   - Then `p` and `ch` have opposite signs and `min(|p|, C) > M/2`, so their exponents differ by at most one.
   - Let `μ = min(ulp(p), ulp(ch))`. Then `μ > 2⁻²⁵M`, and `p`, `ch`, `p + ch` and `s` are all multiples of `μ`.
   - But `ulp(w) ≤ 2u|w| < 2⁻⁴⁴M < μ`. So `s` is a multiple of `ulp(w)` with `|s|/ulp(w) < |w|/ulp(w) < 2²⁴`, and the condition holds with `es = ew`.

**GEMM.** Every kernel accumulates each output as `ŝ₀ = 0`, `ŝₖ = dd_fma(aₖ, bₖ, ŝₖ₋₁)`. Its interleaved renormalizations are `Fast2Sum`s of normalized pairs, so they are exact. By induction, `|ŝ_K − s_K| ≤ (α + β(K−1))(1 + βu²)^(K−1)·u²·Σ|aₖ||bₖ|`, which is at most `(3K + 11)u²(|A||B|)ᵢⱼ` for `K ≤ 2²⁰`. Through the FP64 interface the input conversion `hi = fl₃₂(x)`, `lo = fl₃₂(x − hi)` perturbs each element by at most `u²|x|`, and the result is rounded to FP64 once. The bound then becomes `|Ĉ − C| ≤ (3K + 13)u²|A||B| + 2⁻⁵³|Ĉ|`.

**What the tests show.** `make test-dd-bound` cuts the shader primitives out of `src/apple_bottom.m`, compiles them on the CPU without contraction, and compares 5×10⁶ triples with MPFR at 400 bits. Half are adversarial and half are cancelling. Over 5×10⁷ triples:
- `Fast2Sum` was exact in every case.
- The product error reached 4.33u² (Algorithm 11 reached 4.20u² on the same inputs).
- The step error reached 0.93 of the bound.
- With `|c| ≈ 94|ab|`, the step error reached 2.92u²|c|, so `β = 3` is nearly sharp.
- The worst step error relative to `max(|ab|, |c|)` was 6.80u², so the old 8u² envelope held in these tests even though it had no proof.

`β` multiplies the partial sum, so it sets the slope in K. Saturating sums (VAL-1, QE-like operands: 10.1u² at K = 18,277) are the regime where it matters. The proof is on paper, not checked in a proof assistant like the Muller–Rideau bounds.

References:
- Joldes, Muller, Popescu (2017). "Tight and Rigorous Error Bounds for Basic Building Blocks of Double-Word Arithmetic." *ACM TOMS* 44(2), Article 15. DOI: 10.1145/3121432
- Muller, Rideau (2022). "Formalization of Double-Word Arithmetic, and Comments on 'Tight and Rigorous Error Bounds for Basic Building Blocks of Double-Word Arithmetic.'" *ACM TOMS* 48(1), Article 9. DOI: 10.1145/3484514

### 1.3 DGEMM Error Analysis

For `C = A × B` with dimensions `M × N` and `K`:

**Theoretical bound** (Higham, 2002):

```
‖ΔC‖_F ≤ γ_K · ‖A‖_F · ‖B‖_F
```

where `γ_K = K · ε / (1 - K · ε) ≈ K · ε` for `K · ε ≪ 1`, and `ε = 2⁻⁴⁸` for DD.

> *Revision 1.4: double-float does not satisfy the standard model with ε = u². The proven componentwise bound is `(3K + 13)u²|A||B|` (§1.2), three times this estimate's slope. The figures below are kept as written.*

**For normalized matrices** (`‖A‖_F = ‖B‖_F = √N`):

```
relative_error ≈ K · 2⁻⁴⁸ = N · 2⁻⁴⁸
```

### 1.4 Probabilistic Error Bounds

The effective DD unit roundoff is `u² = 2⁻⁴⁸ ≈ 3.55 × 10⁻¹⁵`.

For DGEMM with inner dimension K, two regimes apply:

**Worst-case (adversarial inputs)**: The Higham bound gives `O(K · u²)`. ~~For K=4096, this yields `~1.5 × 10⁻¹¹` — a hard ceiling that holds for any input, including all-positive or maximally-correlated matrices.~~ The proven componentwise bound (§1.2) is `(3K+13) · u²` of `(|A||B|)_ij`, about `4.4 × 10⁻¹¹` at K=4096.

**Probabilistic (random inputs)**: When rounding errors are uncorrelated (the typical case for random or well-conditioned scientific matrices), the Wilkinson random walk model applies. Errors accumulate as a random walk rather than coherently, giving `O(√K · u²)`. For K=4096: `√4096 · 3.55 × 10⁻¹⁵ ≈ 2.3 × 10⁻¹³`.

This explains the empirically observed `√N` scaling in Section 2.1 (slope = 0.493 ≈ 0.5). The measured Frobenius errors of `~10⁻¹⁵` to `~10⁻¹⁴` across N=64 to N=4096 fall between the probabilistic `√K · u²` and the worst-case ~~`K · u²`~~ `(3K+13) · u²` bounds, exactly where theory predicts for Uniform(-1,1) random matrices with stochastic cancellation.

**The stated guarantee uses the conservative `O(N · 2⁻⁴⁸)` bound.** The `√N` scaling is a property of the input distribution, not of the algorithm, and should not be assumed for production error budgets.

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

*Note (Revision 1.3).* The max-element column is max |ΔC_ij| / |C_ij| over the entries with |C_ij| > 10⁻¹⁵ (`tests/verification/test_convergence.c`). With random data some entries of C come close to zero by cancellation, and dividing by them makes this measure grow with the number of entries, for any finite-precision GEMM. Errors relative to (|A||B|)_ij, the componentwise measure of `LIMITATIONS.md` and of the tests, do not grow this way. The advice above against pivoting and eigensolvers rests on this column; whether they need more than double-float gives has not been tested either way.

### 2.4 Production Validation (Quantum ESPRESSO)

*Revised 2026-10-14.* VAL-1 revision 2 (`tests/validation/VAL001_QE_Si64.md`)
ran Quantum ESPRESSO (develop, 7.5) on 64-atom Si with PAW, with the GEMMs
on Accelerate (`AB_MODE=cpu`) and on the GPU in DD (`AB_MODE=gpu`, the calls
that make up 89–99% of the GEMM work). The total energy at every SCF
iteration and every printed band energy agree in every printed digit
(10⁻⁸ Ry, 10⁻⁴ eV), on the k-point (ZGEMM) and Γ-point (DGEMM) paths, with
the same number of iterations.

**Implication**: DD precision is sufficient for these SCF calculations. DFT
energy is a norm-averaged quantity (sum over occupied states), so
Frobenius-class error is appropriate.

---

## 3. Validated Operating Envelope

### 3.1 Matrix Dimensions

| Type | Validated Range | Notes |
|------|----------------|-------|
| Square matrices | `64 ≤ N ≤ 4096` | Optimal for N ≥ 2048 |
| Rectangular matrices | Any aspect ratio; tested to 143666 × 171 and K = 143666 | Error scales with K dimension |
| Small matrices | `N ≥ 64` | N < 64: CPU overhead dominates |
| Large matrices | `N ≤ 4096` (validated); `rows·cols ≤ 2³¹ − 1` per matrix (max) | Square limit N = 46340 |

**K-dimension error scaling**: For DGEMM with dimensions M×K × K×N, max element error scales as `O(K)×2⁻⁴⁸`. ~~With K=10000, expect ~1.5×10⁻¹³ max error (still within DD precision class).~~ (Revision 1.3: that figure followed from neither the O(K) nor the √K form.) Rectangular matrices work correctly at all aspect ratios.

### 3.2 Numerical Range

| Condition | Validated Range | Behavior Outside Range |
|-----------|----------------|------------------------|
| Exponent range, BLAS ABI | Any finite FP64 | Rows of op(A) and columns of op(B) outside `[2⁻²⁰, 2²⁰)` are scaled by powers of two; calls a guard cannot certify run on Accelerate |
| Exponent range, native API | Nonzero entries of a matrix spanning at most `2⁷⁸` | One power-of-two scale per matrix; a wider span keeps FP32's range relative to it |
| Condition number | `κ < 10¹²` | Higher κ: error grows as `κ · N · ε` |
| NaN/Inf | Documented | BLAS ABI: the call runs on Accelerate; native API: propagates (DD's `∞ − ∞` gives NaN) |

**Subnormal handling**: the GPU flushes FP32 subnormals (below 2⁻¹²⁶) to zero, as inputs and as results, and a DD product keeps its full precision only above about 2⁻⁸⁰. The power-of-two scaling keeps data out of that region; see `LIMITATIONS.md#exponent-range` for the measurements and the guard's thresholds.

### 3.3 Precision Scaling by Operation

| Operation | Frobenius Precision | Max Element Precision | Validated | Notes |
|-----------|---------------------|----------------------|-----------|-------|
| `ab_dgemm(A, B, C)` | `~N·10⁻¹⁵` | `~N²·⁵·10⁻²¹` | ✓ V-2 | C = A × B |
| `ab_dgemm_scaled(α, A, B, β, C)` | `~N·10⁻¹⁵` | `~N²·⁵·10⁻²¹` | ✓ BUG-3 fix | C = αAB + βC, α,β in DD |
| `ab_zgemm(Ar, Ai, Br, Bi, Cr, Ci)` | `~3N·10⁻¹⁵` | `~3N²·⁵·10⁻²¹` | ✓ QE validation | Complex: 3 real DGEMMs, 4 for imbalanced parts (`LIMITATIONS.md#complex-products`) |
| `ab_matrix_scale(α, A)` | `~10⁻¹⁵` | `~10⁻¹⁵` | ✓ BUG-4 fix | Element-wise A ← αA |
| `ab_dsyrk(A, C)` | `~N·10⁻¹⁵` | `~N²·⁵·10⁻²¹` | ✓ Tested | C = A × Aᵀ (symmetric) |

**DEPRECATED**: `ab_zherk` runs at 0.06–0.67× Accelerate's speed (N ≤ 2,048, 2026-10-14 measurement); use `cblas_zherk`.

---

## 4. Out-of-Scope Conditions

The following conditions are **outside the validated envelope**. Results are undefined or may fail:

### 4.1 Matrix Shapes

- **No longer an issue** — Rectangular matrices of any aspect ratio now work correctly (v1.2.0)
  - Previous reports of failures were from different codebase state
  - Error scales with K dimension but remains within DD precision bounds

### 4.2 Numerical Extremes

- **Exponent spans a power-of-two scale cannot bridge**: a native `ABMatrix`
  whose nonzero entries span more than 2⁷⁸, or BLAS-ABI data whose products
  the guard cannot certify (those calls run on Accelerate)
  - DD uses FP32 components; products far below the result lose their low
    parts, and values beyond FP32's range overflow or flush
- **Condition numbers `κ > 10¹²`**: Error bound becomes `κ · N · 2⁻⁴⁸ > 10⁻³`
  - **Workaround**: Use IEEE FP64 (`cblas_dgemm`) for ill-conditioned problems

### 4.3 Special Values

| Input | Behavior | Tested |
|-------|----------|--------|
| FP32 subnormals (`< 2⁻¹²⁶` after scaling) | Flushed to zero by the GPU | ✓ exponent-range suite (`test_blas_abi`) |
| `±Infinity` | BLAS ABI: Accelerate (IEEE result); native: propagates (NaN if `∞ - ∞`) | ✓ BLAS ABI (`range-inf` cases); native ⚠ V-4 (planned) |
| `NaN` | BLAS ABI: Accelerate; native: propagates | ⚠ V-4 (planned) |

### 4.4 Concurrency

- ~~**Thread safety**: apple-bottom is **NOT thread-safe**~~
  - ~~Metal command queue serializes operations~~
  - ~~**Workaround**: Use one `ab_init()` per thread, or external locking~~
- *Corrected in Revision 1.3* (`include/apple_bottom.h`, Thread Safety): `ab_init` and `ab_shutdown` are safe to call from any thread. Calls on different matrices may come from different threads; one matrix, pool or session must not be used by two threads at once. All work shares one Metal command queue.

---

## 5. Performance vs. Precision Trade-offs

### 5.1 GPU Crossover Point

*Revised 2026-10-14 and 2026-10-30* (M2 Max, against Accelerate; data in
`benchmarks/results/2026-10-14-3253a22-bench-paper/` and
`benchmarks/results/2026-10-14-52f1880-calibrate/`):

- **Matrices resident on the GPU** (native API): DGEMM is faster from about
  N = 1,500 (1.10–1.16× from N = 2,048); ZGEMM from about N = 1,000
  (1.30–1.39× from N = 1,536).
- **One call at a time from host memory** (BLAS ABI, 1.4.0's host path,
  which packed and converted on the CPU): slower for every measured shape
  except ZGEMM with both operands transposed at N ≥ 4096 (1.11–1.20×).
  `AB_MODE=auto` routes by this measurement. 1.5.0's staged path converts
  on the GPU and has not been calibrated yet; Quantum ESPRESSO forced onto
  it takes 1.01–1.06× its time on Accelerate
  (`benchmarks/results/2026-10-29-351d56c-staged-refusals/`).

### 5.2 Precision vs. Performance

| Approach | Frobenius Precision | Max Element (N=4096) | Performance (N=2048) | Use Case |
|----------|---------------------|----------------------|----------------------|----------|
| CPU FP64 (`cblas_dgemm`) | `~10⁻¹⁶` | `~10⁻¹⁶` | 1.0× (baseline) | Reference, κ > 10¹², pivoting |
| apple-bottom DD | `~10⁻¹⁵` | `~10⁻⁶` | 1.15× (resident, 2026-10-14) | Iterative solvers, DFT, MD |
| GPU FP32 (for comparison) | `~10⁻⁸` | `~10⁻⁸` | not measured here | Deep learning only |

**Recommendation**: Use apple-bottom when:
1. Problem requires Frobenius error `< 10⁻¹⁴` (rules out FP32)
2. Matrices are square or moderately rectangular
3. The algorithm can keep its large matrices on the GPU between products (codes that call BLAS one call at a time, such as Quantum ESPRESSO, cannot)
4. **Application is norm-averaged** (energies, residuals), not element-sensitive

---

## 6. Validation Traceability

### 6.1 Test Coverage

| Test ID | Description | Status | Source |
|---------|-------------|--------|--------|
| **V-2** | DGEMM convergence study (N ≤ 4096) | ✓ PASS | `tests/verification/test_convergence.c` |
| **VAL-1** | QE Si64 production validation | revision 1 withdrawn; revision 2 ✓ PASS (accuracy) | `tests/validation/VAL001_QE_Si64.md` |
| **BUG-3** | Scaled DGEMM precision (α,β in DD) | ✓ PASS | `test_bug3_dgemm_scaled_precision` in `tests/test_correctness.c` |
| **BUG-4** | Matrix scaling precision (α in DD) | ✓ PASS | `test_bug4_matrix_scale_precision` in `tests/test_correctness.c` |

### 6.2 Regression Testing

**Automated suite**: 42 tests, all passing (v1.0.2-bugfix)
- **Correctness**: 37 tests (API, edge cases, vs-Accelerate)
- **Precision**: 1 test (Frobenius error < 10⁻¹⁴ for N ∈ {64..2048})
- **Regression**: 5 tests (BUG-1 through BUG-6 fixes)

**CI**: Run `make ci-local` before each release (`docs/RELEASING.md`)

### 6.3 Configuration Management

| Item | Value | Notes |
|------|-------|-------|
| **Validated version** | v1.0.2-bugfix | Git tag `v1.0.2-bugfix` (SHA: f3e2fc1) |
| **Compiler** | clang 15.0.0 (Apple) | `-O3 -std=c11` |
| **Metal SDK** | macOS 15.0+ SDK (Xcode 16+) | **Fast math must be off** (`MTLMathModeSafe`, or `fastMathEnabled = NO` before macOS 15 / SDK 15) - with Metal's default fast math, precision degrades from ~10⁻¹⁵ to ~10⁻⁸ |
| **Test system** | M2 Max, 64GB RAM | 12 CPU cores, 38 GPU cores |

**CRITICAL**: the shaders must be compiled without fast math. Metal's default fast math lets the compiler reorder and fuse the error-free transformations, and DD precision degrades by about 7 orders of magnitude (~10⁻⁸ instead of ~10⁻¹⁵). The validated v1.0.2 baseline required `MTLMathModeSafe` (SDK 15+). Since 1.4.0 the library uses `MTLMathModeSafe` where the SDK and OS provide it and `fastMathEnabled = NO` otherwise, which keeps full DD precision; `test_precision` and the V-2 study fail if precision is lost. Do not modify validated code without re-running V&V tests. Tag validated states in Git.

---

## 7. Usage Guidance for Production

### 7.1 Production Deployment Recommendations

For scientific computing on Apple Silicon:

**✓ USE apple-bottom for**:
- DFT, molecular dynamics, FEM iterative solvers (norm-averaged quantities; candidates: Quantum ESPRESSO is the only application run with apple-bottom)
- Large products on GPU-resident matrices (§5.1: DGEMM from about N = 1,500, ZGEMM from about N = 1,000 on an M2 Max), condition number `κ < 10⁶`
- Applications requiring `10⁻¹⁴` to `10⁻¹⁰` Frobenius convergence

**✗ DO NOT USE for**:
- Algorithms requiring element-wise accuracy `< 10⁻⁶` (e.g., pivoting, eigensolvers)
- Ill-conditioned systems (`κ > 10¹²`)
- Smaller products, and GEMMs issued one call at a time from host memory (§5.1)
- ~~Thread-unsafe contexts (add external locking)~~ One matrix, pool or session used by two threads at once (§4.4)

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
- Joldes, M., Muller, J.-M., Popescu, V. (2017). "Tight and rigorous error bounds for basic building blocks of double-word arithmetic." *ACM Trans. Math. Softw.*, 44(2), Article 15. DOI: 10.1145/3121432
- Muller, J.-M., Rideau, L. (2022). "Formalization of double-word arithmetic, and comments on 'Tight and rigorous error bounds for basic building blocks of double-word arithmetic.'" *ACM Trans. Math. Softw.*, 48(1), Article 9. DOI: 10.1145/3484514
- Higham, N. J., Mary, T. (2022). "Mixed precision algorithms in numerical linear algebra." *Acta Numerica*, 31, 347-414.

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
| 1.1 | 2026-04-04 | Add theoretical precision bounds, FMA optimization reference | Grant Heileman |
| 1.2 | 2026-04-16 | Correct DWTimesDW2 error bound to Muller-Rideau 2022 Thm 2.7 (`< 5u²/(1+u)² < 5u²`); fix EFT table (twoSum/twoProduct are error-free, `dd_fma` at DD level) | Grant Heileman |
| 1.3 | 2026-10-14 to 2026-10-30 | Correction notice (§2.4 replaced by VAL-1 revision 2, §5.1 by the 1.4.0 measurements); exponent range and complex products for 1.5.0 (§3.2, §3.3, §4.2, §4.3); per-operation bounds corrected (§1.2: `dd_mul` < 5u², `dd_fma` ≲ 8u²); guarantee restated; notes on the max-element measure (§2.3) and threads (§4.4); §5.1 points at 1.5.0 | Grant Heileman |
| 1.4 | 2026-10-30 | §1.2: the default product is not JMP Algorithm 11, so Theorem 2.7 does not apply. Proven bounds for `dd_mul` (≤ 6.000001u²) and `dd_fma` ((13+43u)u²|ab| + (3+5u)u²|c|), the final `Fast2Sum` shown to be exact, and the GEMM bound (3K+13)u²|A||B|. They replace the unproven 5u² and 8u² figures. New `make test-dd-bound`. §1.3: slope note | Grant Heileman |

---

## 10. Contact

**Maintainer**: Grant Heileman
**Repository**: https://github.com/grantdh/apple-bottom
**Issues**: File at GitHub Issues for production support

**For production validation support**: Provide (1) test case description, (2) measured error, (3) expected error from this document, (4) git SHA of your apple-bottom build.
