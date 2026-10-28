# Limitations and Epistemic Posture

This document captures the epistemic stance shared across the apple-bottom V&V package. It is referenced from `PRECISION_ENVELOPE.md` and `VV_REPORT.md` and is intended to be read alongside them.

## Framing

Throughout the V&V package we scope claims to what we measure rather than to what prior theorems might be extended to cover; we state conservative envelopes where tight bounds are open, and preempt cross-hardware generalization until revalidation is done.

Specific instances of this stance, linked to the sections where each is developed:

- **Fused double-word FMA envelope.** The `dd_fma(a, b, c) = a·b + c` primitive elides the mul-side `fastTwoSum` renormalization of Joldes–Muller–Popescu 2017 Algorithm 11 (DWTimesDW2). The Muller–Rideau 2022 Theorem 2.7 bound on DWTimesDW2 therefore does not apply by inheritance to the fused form. We state a conservative `≲ 8u² · max(|a·b|, |c|)` envelope from triangle inequality over the non-fused composition and validate it empirically at `N ∈ {1024, 2048, 3072}` in the `N·u²` accumulation regime. The tight bound for the fused form is an open theoretical question. See `PRECISION_ENVELOPE.md §1.2`.

- **Hardware scoping to Apple M2 Max.** All measurements in the V&V package are taken on a single Apple M2 Max (Apple8 GPU family) under `mathMode = .safe`. The Metal Shading Language specification commits to per-operation ULP tolerances but does not document cross-GPU-family bit-identity across the Apple7/Apple8/Apple9 families. The reported numerical envelope is an M2 Max measurement, not a platform guarantee; cross-family revalidation on M1, M3, and M4 is future work. See `PRECISION_ENVELOPE.md §2` and the hardware generality discussion in the companion paper manuscript.

- **DWTimesDW3 opt-in variant retained as non-default.** The DWTimesDW3 primitive (JMP 2017 Algorithm 12, MR 2022 Theorem 2.8) is available as a compile-time opt-in via `DWTIMESDW3=1` but is not the default. The default retains DWTimesDW2 based on measured cost/benefit: the tighter theoretical bound does not produce a measurable accuracy improvement on the DFT workloads targeted by this library, while the extra operations add FP32 cost on the hot path. The opt-in path is validated and supported; the decision to retain DWTimesDW2 as default is empirical, not theoretical. See `DD_PRIMITIVES_CURRENT.md §1.2`.

- **Exponent range: scaled where a power of two can keep it, refused where it cannot.** The thresholds below come from M2 Max measurements and are conservative; a call the guard cannot certify runs on Accelerate rather than returning a degraded GPU result. See [Exponent range](#exponent-range).

- **Complex products: three where they hold their accuracy, four where they do not.** Gauss's three-product ZGEMM loses accuracy in the imaginary part when real and imaginary parts are imbalanced; such calls run as four real products, at a measured threshold. See [Complex products](#complex-products).

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
    - every nonzero product is at least 2⁻⁷⁸; or, failing that, for every C_ij one of two products through the largest entries of its row and column reaches K·2⁻⁷⁸;
    - C as scaled for the upload, and beta times it, stay below 2¹²², and beta·C keeps full precision wherever it is the whole of C_ij.
  - Any other call runs on Accelerate, counted as a `range` fallback in `AB_REPORT`, as do calls with NaN or Inf in op(A) or op(B) (or in C with beta ≠ 0, for DGEMM). Those therefore get IEEE results instead of DD's NaN from Inf − Inf.
  - Rows and columns inside the window are not scaled, so ordinary data computes exactly as in 1.4. `tests/test_qe_golden.c` holds such calls to 1.4.0's bits. The exponent-range suite of `tests/test_blas_abi.c` checks scaled calls against cblas and, bit for bit, against the unscaled call times its power of two.
- **Native `ABMatrix` API.** A matrix holds one power-of-two scale, chosen by `ab_matrix_upload`.
  - It is 0 when every nonzero magnitude lies in [2⁻³⁹, 2³⁹]; the products of two such matrices then stay in [2⁻⁷⁸, 2⁷⁸].
  - A matrix outside that window whose nonzero entries span at most 2⁷⁸ is centred on 1.
  - The GPU operations carry the scales to their results; real and imaginary parts of different scales are aligned before Gauss's 3M sums (4M takes each product on its own scale), and operands of `ab_matrix_add` and `ab_matrix_sub` before the sum. `ab_dsyrk` writes C's upper triangle only, and brings the rest to the new scale so that it keeps its values. `tests/test_correctness.c` runs each of these operations on data near 2^±73 (`test_xr_*`).
  - A single matrix whose entries span more than 2⁷⁸, or results of the GPU whose span grows past it, keep FP32's range relative to their scale. There, products of entries far apart can still lose their low part without a warning. The native API has no Accelerate fallback; use the BLAS ABI, or split such data.
- **Device-buffer API** (`ab_dev_dgemm`, `ab_dev_zgemm`). It converts through `ABMatrix` handles, so it behaves as the native API: one scale per matrix, no fallback.
- **Not covered**: NaN and Inf through the native API propagate as before, and DD's Inf − Inf gives NaN where IEEE arithmetic gives Inf.

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

A ZGEMM runs as 4M when ρ exceeds 4 (`AB_GAUSS3M_MAX_RHO`), so 3M's imaginary part stays within 8 u², the bound the tests hold both parts to.

- **BLAS ABI** (`zgemm_`, `ab_zgemm_blas`) and **device-buffer API** (`ab_dev_zgemm`, `ab_dev_zgemm_offset`): p and q come from the largest real and imaginary parts of each row of op(A) and each column of op(B), and the call takes the largest ρ of any row and column (at a corner of the ranges of p and q). Rows or columns imbalanced in opposite directions are caught even where A and B as a whole look balanced. The device API gathers those parts while it copies op(A) and op(B) out of their buffers.
- **Native API** (`ab_zgemm`, `ab_zgemm_ex`, `ab_zgemm_async`, `ab_zgemm_batched`, `ab_batch_zgemm`): p and q come from each part's largest magnitude over the whole matrix, which `ab_matrix_upload` measures. Two cases therefore stay 3M: rows or columns imbalanced in opposite directions in a matrix whose parts balance as a whole, and a matrix computed on the GPU, which has no such measurement. Use the BLAS ABI or the device API where either is possible.
- **Cost of 4M** on M2 Max: per native call, 1.07× on 160×160×9140 and 150×150×18277, 1.24× on 9140×160×160 and 1.27–1.34× from 512³ to 2048³; per `ab_dev_zgemm` call, 1.05–1.17× on 160×160×9140 and 1.12–1.19× on 9140×160×160; end to end through `zgemm_`, 1.00–1.06× on those QE shapes and 1.10× at 1024³.
- Balanced data keeps 3M and its bits (`tests/test_qe_golden.c`). `test_zgemm_imbalance` in `tests/test_blas_abi.c` and in `tests/test_device_api.c`, and `test_imbalance_zgemm_paths` in `tests/test_correctness.c`, hold Re(C) and Im(C) of nearly real, nearly imaginary and line-wise mixed data (parts 10⁻⁶ apart) each to its own reference; under 3M they missed Im(C) by about 10⁶ u².

## What this posture rules in and out

This posture is deliberately narrower than "the library is correct under all conditions." It is also deliberately stronger than "the library works in practice." Specifically:

- Claims carry a named scope (workload, matrix size, hardware, compile-time flag set) and an envelope defensible within that scope.
- Claims outside that scope are either flagged as open (fused FMA tight bound), flagged as future revalidation work (cross-family hardware generality), or flagged as empirical choices that may not generalize (DWTimesDW3 opt-in default).
- Claims that would require extending published theorems beyond their stated preconditions are not made.

Production evaluators should read each numerical claim in the V&V package as scoped by this framing.
