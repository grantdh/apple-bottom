# Limitations and Epistemic Posture

This document captures the epistemic stance shared across the apple-bottom V&V package. It is referenced from `PRECISION_ENVELOPE.md` and `VV_REPORT.md` and is intended to be read alongside them.

## Framing

Throughout the V&V package we scope claims to what we measure rather than to what prior theorems might be extended to cover; we state conservative envelopes where tight bounds are open, and preempt cross-hardware generalization until revalidation is done.

Specific instances of this stance, linked to the sections where each is developed:

- **Fused double-word FMA bound.** `dd_fma(a, b, c) = a·b + c` and the default `dd_mul` use a two-FMA variant of Joldes–Muller–Popescu 2017 Algorithm 11 (DWTimesDW2), not the algorithm itself. The formally verified Muller–Rideau 2022 bounds therefore do not apply. `PRECISION_ENVELOPE.md §1.2` proves bounds for the code as written:
  - `dd_mul` ≤ 6.000001u²;
  - `dd_fma` ≤ (13+43u)u²|ab| + (3+5u)u²|c|, with its final `fastTwoSum` always exact;
  - GEMM ≤ (3K+13)u²|A||B| componentwise.

  These replace the earlier `≲ 8u²` envelope, which had no proof. The proofs are on paper, not machine-checked. They assume no underflow. Only the BLAS ABI's and BLAS-2's range guards check for it, and they accept underflow in a product that a larger product of the same element (for complex data, of the same part) carries ([Exponent range](#exponent-range)). 1.5.x let some data through the BLAS ABI's guard unchecked, and the 1.6.0 candidate at 8b48826 also through BLAS-2's ([What the guard misses](#what-the-guard-misses)). The native API and the device GEMMs have no guard ([Exponent range](#exponent-range)). `make test-dd-bound` checks the `dd_fma` bound and its product step against MPFR on the extracted shader code.

- **Hardware scoping to Apple M2 Max.** All measurements in the V&V package are taken on a single Apple M2 Max (Apple8 GPU family) under `mathMode = .safe`. The Metal Shading Language specification commits to per-operation ULP tolerances but does not document cross-GPU-family bit-identity across the Apple7/Apple8/Apple9 families. The reported numerical envelope is an M2 Max measurement, not a platform guarantee; cross-family revalidation on M1, M3, and M4 is future work. See `PRECISION_ENVELOPE.md §2`.

- **DWTimesDW3 opt-in variant retained as non-default.** The DWTimesDW3 primitive (JMP 2017 Algorithm 12, MR 2022 Theorem 2.8) is available as a compile-time opt-in via `DWTIMESDW3=1` but is not the default. The default retains DWTimesDW2 based on measured cost/benefit: the tighter theoretical bound does not produce a measurable accuracy improvement on the DFT workloads targeted by this library, while the extra operations add FP32 cost on the hot path. The opt-in path builds with `DWTIMESDW3=1`, but no automated test or CI job builds or runs it (`tests/test_precision.c` notes that its thresholds were measured on both builds); the decision to retain DWTimesDW2 as default is empirical, not theoretical. See `DD_PRIMITIVES_CURRENT.md §1.2`.

- **Exponent range: scaled where a power of two can keep it, refused where it cannot.** The thresholds below come from M2 Max measurements; a call the guard refuses runs on Accelerate rather than returning a degraded GPU result. 1.5.0 and 1.5.1 let some data through with degraded results; 1.6.0 runs the measured cases on Accelerate or keeps them at DD accuracy. See [Exponent range](#exponent-range) and [What the guard misses](#what-the-guard-misses).

- **Complex products: three by default, four where the largest parts are imbalanced.** Gauss's three-product ZGEMM loses accuracy in the imaginary part when real and imaginary parts are imbalanced. A call runs as four real products when the largest parts of its lines (BLAS ABI, device API) or of its matrices (native API) are imbalanced past a measured threshold. Data whose imbalance those largest parts do not show stays on three products, and its imaginary part can exceed the 8 u² the tests hold it to. See [Complex products](#complex-products).

## Exponent range

A DD value is a pair of FP32 numbers, so it has FP32's exponent range. On the M2 Max GPU (measured, `MTLMathModeSafe`):
- FP32 subnormals are flushed to zero, as inputs and as results.
- A DD product keeps its full precision while it stays above about 2⁻⁸⁰. In a K = 512 DGEMM of uniform data scaled so that its largest products are 2⁻⁸⁴, the componentwise error against a compensated reference is 1.8 u², against 1.6 u² unscaled. With the largest products at 2⁻⁸⁸ it is 49 u², at 2⁻¹⁰⁰ 2×10⁶ u², and below 2⁻¹²⁶ everything is lost.
- Sums overflow at 2¹²⁸.

The library keeps data out of those limits with exact powers of two. A power of two scales every product and partial sum of an entry of C alike, so wherever no intermediate leaves FP32's normal range either way the bits are unchanged.

- **BLAS ABI** (`dgemm_`, `zgemm_`, `ab_*gemm_blas`).
  - Each row of op(A) and column of op(B) whose largest magnitude lies outside [2⁻²⁰, 2²⁰) is scaled on its way to DD so that it lies in [2³⁰, 2³¹), and each C_ij is scaled back by the product of its row's and column's factors. That leaves room both below, for the products of a line's smaller entries to stay above 2⁻⁷⁸, and above, for sums to stay below 2¹²².
  - A DGEMM alpha or beta outside [2⁻²⁰, 2²⁰] gives its exponent to C's scale the same way.
  - A call runs on the GPU only when a guard certifies, from the scaled operands, that:
    - no sum can reach 2¹²²;
    - every nonzero product is at least 2⁻⁷⁸, and every nonzero entry is too (below about 2⁻¹⁰² an entry's FP32 low part is subnormal, and the GPU reads it as zero); or, failing that, for every C_ij one of two products through the largest entries of its row and column reaches K·2⁻⁷⁸ and, where its row (column) has an entry below 2⁻⁷⁸, K·2⁻⁷⁸ times its column's (row's) largest entry, twice that for ZGEMM (added after 1.5.1, which had no entry floor and could return FP32-level errors, 4.5×10⁻⁸ relative, for such data). For ZGEMM, "entry" means each nonzero real and imaginary part, and the products through the largest entries must reach those amounts in each part of C_ij that has products, in its own scale (|Ar||Br| + |Ai||Bi| for the real part, |Ar||Bi| + |Ai||Br| for the imaginary part);
    - C as scaled for the upload, C′, and beta times it, stay below 2¹²², and beta·C keeps full precision wherever it is the whole of C_ij: every nonzero C′ is at least 2⁻⁷⁸ and so is β′C′; or, failing that, each C′_ij below those floors has a product in its own entry, through the largest entry of its row or of its column, that α′ brings to 2⁻⁷⁸·max(1, |β′|) (a zero row or column, or a row and column whose nonzero entries never meet, leave β′C′_ij all of C_ij).
    - Until these floors on parts and on C′ were added after 1.5.1, a small imaginary part, a C′ between 2⁻⁹⁸ and 2⁻⁷⁸ with β′ up to 2²⁰, or a small C′ where the supports of its row and column miss each other, could lose its FP32 low word (2⁻²⁹ relative in the probes) or the whole value below 2⁻¹²⁶, with the call reporting success.
    - The guard measures magnitudes by their FP64 high words, and a nonzero magnitude below 2⁻¹⁰⁴² has high word 0, as zero has. It counts such a value, in op(A), op(B) or C, or a C′ that the scaling takes below 2⁻¹⁰⁴² or to 0, as the smallest nonzero high word: a line made only of such values is scaled like any other (by 2¹⁰⁷²), and as a smallest magnitude it reads as 2⁻¹⁰⁷⁴, below every floor above. Until this was added after 1.5.1 such values were invisible to the guard, and a line of them read as zero: the GPU flushed them, so a C entry made of them came back as 0 (a C′ from C = 2⁻¹⁰⁰ included), with the call reporting success.
  - Any other call runs on Accelerate, counted as a `range` fallback in `AB_REPORT`, as do calls with NaN or Inf in op(A) or op(B), or in C with beta ≠ 0. Those therefore get IEEE results instead of DD's NaN from Inf − Inf, and for ZGEMM instead of the (NaN, NaN) the epilogue's complex arithmetic would make of (Inf, y).
    - Accelerate and Netlib differ on such an element in C. With beta = 1 and TRANSA 'T' or 'C', (NaN, y) stays (NaN, finite) on Accelerate where Netlib returns (NaN, NaN), and (Inf, y) stays (Inf, finite) where Netlib returns (Inf, NaN). With TRANSA = 'N', or with beta = 0.25, the two agree.
  - A ZGEMM applies alpha and beta in FP64 after scaling each part of its product P back by its row's and column's factors, so a P beyond FP64's range there, or alpha·P, or beta·C, would become Inf, and the epilogue's complex arithmetic NaN, even where C itself is in range. With a and b the largest real or imaginary parts of op(A) and op(B), each part of P is a sum of 2K terms below 2^(⌊log₂ a⌋ + 1)·2^(⌊log₂ b⌋ + 1), so below 2^E, E = ⌊log₂ a⌋ + ⌊log₂ b⌋ + ⌊log₂ K⌋ + 5 (a factor 2 for DD's rounding). A ZGEMM also runs on Accelerate, as a `range` fallback, unless E ≤ 1023, (|Re α| + |Im α|)·2^E ≤ 2¹⁰²² and, where C is read, (|Re β| + |Im β|)·2^(⌊log₂ c⌋ + 1) ≤ 2¹⁰²² for C's largest part c; every value the epilogue forms then stays below 2¹⁰²³(1 + 2⁻⁵⁰). NaN and Inf in alpha or beta also go to Accelerate, and so does a finite alpha or beta whose |Re| + |Im| overflows (DBL_MAX(1 + i), say), whatever the data. The bound refuses only calls whose data, with those largest parts, could make a part of P reach 2¹⁰¹⁹, of alpha·P 2¹⁰¹⁷ (2¹⁰¹⁸ for a real or purely imaginary alpha: a part of alpha·P peaks at 2Kab·max(|Re α|, |Im α|), while the bound reads |Re α| + |Im α|), or of beta·C 2¹⁰²¹; it also refuses such calls whose results are finite because their largest parts never meet in a product or their terms cancel. Lines in the window [2⁻²⁰, 2²⁰) give E ≤ 73. Until this was added after 1.5.1, a 32 × 32 × 32 call with A at 2⁶⁰⁰, B at 2⁵⁰⁰, alpha 2⁻²⁰⁰ and beta 0 (C = 2⁹⁰⁵) returned NaN + NaN i on the GPU, where Accelerate returns Inf + NaN i (its own P overflows). A DGEMM applies alpha and beta in DD, scaled, before its one scale-back, which overflows only where the result itself does.
  - Rows and columns inside the window are not scaled, so ordinary data computes exactly as in 1.4. `tests/test_qe_golden.c` holds such calls to 1.4.0's bits. The exponent-range suite of `tests/test_blas_abi.c` checks scaled calls against cblas and, bit for bit, against the unscaled call times its power of two.
- **Native `ABMatrix` API.** A matrix holds one power-of-two scale, chosen by `ab_matrix_upload`.
  - It is 0 when every nonzero magnitude lies in [2⁻³⁹, 2³⁹]; the products of two such matrices then stay in [2⁻⁷⁸, 2⁷⁸].
  - A matrix outside that window whose nonzero entries span at most 2⁷⁸ is centred on 1.
  - The GPU operations carry the scales to their results; real and imaginary parts of different scales are aligned before Gauss's 3M sums (4M takes each product on its own scale), and operands of `ab_matrix_add` and `ab_matrix_sub` before the sum. `ab_dsyrk` writes C's upper triangle only, and brings the rest to the new scale so that it keeps its values. `tests/test_correctness.c` runs each of these operations on data near 2^±73 (`test_xr_*`).
  - Values below 2⁻¹⁰⁴², whose FP64 high word is 0, count only as a largest magnitude: a matrix made only of them is scaled as if they spanned [2⁻¹⁰⁷⁴, 2⁻¹⁰⁴²] (until this was added after 1.5.1 such a matrix measured as zero and was stored as zeros), and a matrix that also holds larger values keeps the scale those give it, so its smallest values can fall below FP32's range.
  - A single matrix whose entries span more than 2⁷⁸, or results of the GPU whose span grows past it, keep FP32's range relative to their scale. There, products of entries far apart can still lose their low part without a warning. The native API has no Accelerate fallback; use the BLAS ABI, or split such data.
- **Device-buffer API** (`ab_dev_dgemm`, `ab_dev_zgemm`). It converts through `ABMatrix` handles, so it behaves as the native API: one scale per matrix, no fallback. `ab_dev_zgemm` also applies alpha and beta in FP64 after scaling its product back, with no bound on that epilogue: on the BLAS ABI's example above (A at 2⁶⁰⁰, B at 2⁵⁰⁰, alpha 2⁻²⁰⁰) it returns NaN + NaN i with `AB_OK`.
- **BLAS-2** (`dgemv_`, `zgemv_`, `zhemv_`, `ab_*gemv_blas`, `ab_dev_dgemv`, `ab_dev_zgemv`, `ab_dev_zhemv`). The kernels compute this guard from the FP64 exponents, at no extra memory traffic. Only x is scaled, by a power of two so that max |x| is in [1/2, 1). The guard holds each output, and each part of a complex output, to its own products: the real part to ar·xr and ai·xi, the imaginary part to ar·xi and ai·xr. A call runs on Accelerate, before y is written, if in one of them the largest product after that scaling has a binary exponent outside L−75..124−L, with L = ceil(log2) of the real products added into it; if what FP32's range can cost it exceeds 2⁻⁴⁹ of that largest product; or if its sum of products is not finite, which NaN and Inf in A or x make it. A part far below the other is therefore held to its own scale. These count as `range` fallbacks. NaN or Inf in alpha or beta also sends the call to Accelerate, not counted as a fallback. NaN or Inf in y with beta ≠ 0 stays on the GPU and reaches only that element of y.
- **Not covered**: NaN and Inf through the native API propagate as before, and DD's Inf − Inf gives NaN where IEEE arithmetic gives Inf.

### What the guard misses

1.5.0 and 1.5.1 let the GEMM data below pass the guard with a degraded result and no `range` fallback, and the 1.6.0 candidate at 8b48826 also the BLAS-2 calls. In 1.6.0 each of these calls runs on Accelerate, or on the GPU within 2.2 u² (uniform data: at most 2.3 u² at 4096³). "Default" is the routing with `AB_MODE` unset, which in 1.6.0 sends these shapes to the GPU only on the M2 Max (38-core GPU) its table was measured on (1.5.x's table had no chip check). "Forced" is `AB_MODE=gpu AB_MIN_GPU_DIM=0`.

Each error is the largest at sampled outputs for one part of the result, against a Dot2 reference, in u² = 2⁻⁴⁸ of that part's own scale: |Ar||Br| + |Ai||Bi| for the real part, |Ar||Bi| + |Ai||Br| for the imaginary part, |A||B| + |beta C| for a real result. Uniform data gives at most 2.3 u² on the 4096³ ZGEMM and under 0.2 u² on these BLAS-2 calls. 2.8 × 10¹⁴ u² means the part came back zero. Measured on M2 Max with `benchmarks/probe_guard_gaps.c` at 8b48826 ([results](../../benchmarks/results/2026-11-06-8b48826-pretag/guard-gaps/)) and at fb4dfe0 ([results](../../benchmarks/results/2026-11-07-fb4dfe0-guard-gaps/)).

| Data | Why the guard passed it | Call | Route | 8b48826 | fb4dfe0 |
|---|---|---|---|---|---|
| Imaginary parts 2⁻¹⁰⁰ of the real parts | The guard read a complex element by its larger part | `zgemm_` 'T','T' 4096³ | default | 1.4 × 10⁵ u² | Accelerate |
| | | `zgemv_` 'N' 3072 | default | 2.9 × 10⁵ u² | Accelerate |
| | | `zhemv_` 'U' 2048 | default | 3.7 × 10⁵ u² | Accelerate |
| | | `zgemm_` 'N','N' 8 × 8 × 2 | forced | 2.7 × 10⁶ u² | Accelerate |
| | | `zgemm_` 'N','N' 544 × 481 × 600 | forced | 2.9 × 10⁵ u² | Accelerate |
| Imaginary parts 2⁻¹³⁰ of the real parts | The same | the first four calls | default, forced | 2.8 × 10¹⁴ u²: the imaginary part is zero | Accelerate |
| Entries of A near 2⁻¹⁰⁶⁰ with B near 2¹⁰⁰⁰, products near 2⁻⁶⁰ | GEMM's line statistics read an FP64 high word, which is 0 below 2⁻¹⁰⁴² | `zgemm_` 'T','T' 4096³ | default | 1.4–1.9 × 10¹³ u²: zero (the terms cancel, so the true parts are at most 5–7% of their scale at the sampled outputs) | GPU, 1.9 u² |
| | | `zgemm_`, `dgemm_` 'N','N' 8 × 8 × 2 | forced | 2.8 × 10¹⁴ u²: zero | GPU, 2.2 u² and 1.0 u² |
| C near 2⁻⁹⁸ with beta ≠ 0, in elements whose row of op(A) and column of op(B) are nonzero but share no k | The guard checked C′ only through \|β′C′\| ≥ 2⁻⁷⁸, and below that only where a whole row or column is zero | `dgemm_` 'N','N' 8 × 8 × 2 | forced | 5.2 × 10⁵ u²; zero for C near 2⁻¹³⁰ | Accelerate |

On Accelerate these calls return at most 1.75 u², as with `AB_MODE=cpu`. Imaginary parts 2⁻⁸⁰ of the real parts now run on Accelerate as well, where 8b48826 ran them on the GPU within 36.5 u². Two limits remain by design. The guard accepts a product that loses bits where a larger product carries its element (its part, for complex data), because the loss stays at that element's rounding level. And the native API and the device GEMMs (`ab_dev_dgemm`, `ab_dev_zgemm`) have no guard ([Exponent range](#exponent-range)). The probe covers the cases above only; it does not show that no other data passes.

## Complex products

ZGEMM uses Gauss's three-multiplication form (3M): Re(C) = Ar·Br − Ai·Bi and Im(C) = (Ar + Ai)(Br + Bi) − Ar·Br − Ai·Bi. Its imaginary part therefore carries the rounding errors of products that can be much larger than Im(C). Relative to Im(C)'s own componentwise scale |Ar||Bi| + |Ai||Br|, the error exceeds that of the four-product form (4M, Im(C) = Ar·Bi + Ai·Br) by about ρ(p, q) = (1 + p)(1 + q)/(p + q), with p = |Ai|/|Ar| and q = |Bi|/|Br|. ρ is 2 for balanced parts and grows like 1/(2p) when both operands are nearly real (p = q ≪ 1) or nearly imaginary. Both forms compute Re(C) the same way.

Measured on M2 Max (M = N = 96, p = q, componentwise against a compensated Dot2 reference, in u² = 2⁻⁴⁸):

| ρ | 3M Im, K = 72 | 3M Im, K = 2048 | 4M Im |
|---|---:|---:|---:|
| 2 (balanced) | 1.8 | 1.6 | 0.9–1.1 |
| 4 | 7.9 | 6.9 | 1.0–1.1 |
| 5.1 | 9.8 | 11.8 | 0.9 |
| 8 | 14.6 | 19.3 | 1.0–1.1 |
| 129 | 321 | 304 | 1.0 |
| 5 × 10⁵ (parts 10⁻⁶ apart) | 1.2 × 10⁶ | 1.1 × 10⁶ | 1.2–1.4 |

A ZGEMM runs as 4M when ρ exceeds 4 (`AB_GAUSS3M_MAX_RHO`). That holds 3M's imaginary part to about 8 u², the bound the tests hold both parts to, on data whose elements are balanced like their lines' largest parts. [§1.2](PRECISION_ENVELOPE.md#12-error-free-transformations) proves (3K + 13) u²|A||B| for one real product, but gives no such limit for 3M's imaginary part on that part's own scale: that part subtracts two products from a third, and those products can be far larger than it (the ρ = 129 row above reaches 321 u² at K = 72, where 3K + 13 is 229). Two kinds of data go beyond 8 u²; these are measurements, not bounds:
- ρ reads the largest parts of a line, not of each element. On data whose magnitudes spread over 2⁰ to 2⁻⁵⁹ within each line (`spread` in `benchmarks/probe_guard_gaps.c`), 3M's imaginary part reaches 26 u² for `zgemm_` 'T','T' 4096³ on the default routing, 58–82 u² for 'N','N' and 118–166 u² for 'C','N' across draws at 544 × 481 × 600 through `ab_dev_zgemm`, and 887 u² for `zgemm_` 'N','C' 257 × 129 × 300 under `AB_MODE=gpu AB_MIN_GPU_DIM=0`, against about 2 u² on uniform data at those calls ([data](../../benchmarks/results/2026-11-06-8b48826-pretag/comparison/ref-data-classes.txt)). The figures depend on the data drawn.
- Long sums of terms of one sign accumulate: 36.5 u² in the real part at K = 4096, on real parts of 1 + u/4 (measured at 8b48826 on the probe's `imag-80` data, which 1.6.0 runs on Accelerate).

- **BLAS ABI** (`zgemm_`, `ab_zgemm_blas`) and **device-buffer API** (`ab_dev_zgemm`, `ab_dev_zgemm_offset`): p and q come from the largest real and imaginary parts of each row of op(A) and each column of op(B), and the call takes the largest ρ of any row and column (at a corner of the ranges of p and q). Rows or columns imbalanced in opposite directions are caught even where A and B as a whole look balanced. The device API gathers those parts while it converts op(A) and op(B) in their buffers.
- **Native API** (`ab_zgemm`, `ab_zgemm_ex`, `ab_zgemm_async`, `ab_zgemm_batched`, `ab_batch_zgemm`): p and q come from each part's largest magnitude over the whole matrix, which `ab_matrix_upload` measures. Two cases therefore stay 3M: rows or columns imbalanced in opposite directions in a matrix whose parts balance as a whole, and a matrix computed on the GPU, which has no such measurement. Use the BLAS ABI or the device API where either is possible.
- **Cost of 4M** on M2 Max: per native call, 1.07× on 160×160×9140 and 150×150×18277, 1.24× on 9140×160×160 and 1.27–1.34× from 512³ to 2048³; per `ab_dev_zgemm` call, 1.05–1.17× on 160×160×9140 and 1.12–1.19× on 9140×160×160; end to end through `zgemm_`, 1.00–1.06× on those QE shapes and 1.10× at 1024³.
- Balanced data keeps 3M and its bits (`tests/test_qe_golden.c`). `test_zgemm_imbalance` in `tests/test_blas_abi.c` and in `tests/test_device_api.c`, and `test_imbalance_zgemm_paths` in `tests/test_correctness.c`, hold Re(C) and Im(C) of nearly real, nearly imaginary and line-wise mixed data (parts 10⁻⁶ apart) each to its own reference; under 3M they missed Im(C) by about 10⁶ u².

## Command buffers stopped by macOS

macOS can stop a long GPU command buffer while the display needs the GPU (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`). What a call then does:

- Finish on the host path or on Accelerate, with a correct result: a BLAS-ABI call, `ab_dev_dgemv`, `ab_dev_zgemv`, `ab_dev_zhemv`, `ab_dtrsm` and `ab_dgesv_mpir` (which redo a failed GPU GEMM step with `cblas_dgemm` in FP64).
- Return `AB_ERROR_KERNEL_FAILED` with no fallback, the output's contents then unspecified: `ab_fft_execute` (which may have written part of `out`), `ab_dev_dgemm`, `ab_dev_zgemm`, the native `ab_dgemm`, `ab_dgemm_scaled`, `ab_zgemm`, `ab_zgemm_ex`, `ab_dsyrk`, `ab_zherk`, `ab_dgemm_batched`, `ab_zgemm_batched`, `ab_matrix_add`, `ab_matrix_sub`, `ab_matrix_scale`, `ab_matrix_copy`, `ab_matrix_zero`, `ab_batch_wait`, `ab_future_wait`. A batch stops at the failed command buffer, so later operations of it have not run.

## What this posture rules in and out

This posture is deliberately narrower than "the library is correct under all conditions." It is also deliberately stronger than "the library works in practice." Specifically:

- Claims carry a named scope (workload, matrix size, hardware, compile-time flag set) and an envelope defensible within that scope.
- Claims outside that scope are either flagged as open (fused FMA tight bound), flagged as future revalidation work (cross-family hardware generality), or flagged as empirical choices that may not generalize (DWTimesDW3 opt-in default).
- Claims that would require extending published theorems beyond their stated preconditions are not made.

Production evaluators should read each numerical claim in the V&V package as scoped by this framing.
