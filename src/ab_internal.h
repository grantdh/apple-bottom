// =============================================================================
// ab_internal.h — internal entry points shared by the library's sources
// =============================================================================
//
// apple_bottom.m owns the ABMatrix layout and the FP64 <-> double-float (DD)
// conversion; blas_wrapper.c's BLAS-ABI path uses these to scale operands by
// powers of two on the way in (the exponent-range guard, blas_wrapper.c), and
// device_api.m's ZGEMM to choose between Gauss-3M and 4M as that path does.
// =============================================================================
#ifndef AB_INTERNAL_H
#define AB_INTERNAL_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "apple_bottom.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Double-float's exponent range
// =============================================================================
// DD values are FP32 pairs. On the GPU, FP32 values and results below 2^-126
// flush to zero, a product keeps its full DD precision only from about
// 2^AB_PRODUCT_MIN_EXP up (its low part's own error terms must stay normal),
// and sums overflow at 2^128 (measured on an M2 Max; docs/vv/LIMITATIONS.md,
// "Exponent range"). The power-of-two scaling of both APIs keeps data inside.
#define AB_PRODUCT_MIN_EXP (-78)    // products from 2^-78 up keep full DD precision
#define AB_SUM_MAX_EXP 122          // sums up to 2^122 stay clear of FP32's 2^128
// A native matrix whose nonzero magnitudes all lie in 2^±AB_NATIVE_WINDOW
// gives products of at least 2^AB_PRODUCT_MIN_EXP with any other such matrix.
#define AB_NATIVE_WINDOW (-AB_PRODUCT_MIN_EXP / 2)

// 2^e for -1022 <= e <= 1023, built from its bits.
static inline double ab_pow2(int e) {
    uint64_t bits = (uint64_t)(e + 1023) << 52;
    double d;
    memcpy(&d, &bits, sizeof d);
    return d;
}

// x * 2^e, rounded once (only a subnormal or overflowing result rounds), as
// ldexp rounds; a multiply by an exact power of two where 2^e is a normal double.
static inline double ab_scale2(double x, int e) {
    return e == 0 ? x : e >= -1022 && e <= 1023 ? x * ab_pow2(e) : ldexp(x, e);
}

// The exponent a scalar gives to a power-of-two scale: 0 for 0, NaN or Inf
// and inside 2^±window, else ilogb(x).
static inline int ab_scalar_exp(double x, int window) {
    if (x == 0.0 || !isfinite(x)) return 0;
    int e = ilogb(x);
    return e >= -window && e <= window ? 0 : e;
}

// =============================================================================
// Host loops across cores
// =============================================================================
// apple_bottom.m's FP64 <-> DD conversions and device_api.m's staging copies
// split large arrays into chunks across cores.
#define AB_CONVERT_PARALLEL_MIN ((size_t)1 << 20)  // elements; serial below this
#define AB_CONVERT_CHUNKS 8

#ifdef __BLOCKS__
// Runs body(chunk, offset, n) on contiguous ranges [offset, offset + n) of at
// least one unit that cover [0, count) units of `work` elements each: all of
// it as chunk 0 below AB_CONVERT_PARALLEL_MIN elements (or for one unit),
// else up to min(count, AB_CONVERT_CHUNKS) chunks in parallel, numbered
// from 0.
void ab_internal_for_each_chunk(size_t count, size_t work,
                                void (^body)(size_t chunk, size_t offset, size_t n));
#endif

// =============================================================================
// Magnitudes as high words
// =============================================================================
// Magnitudes are compared by the high 32 bits of |x| (sign cleared): its
// exponent and top 20 mantissa bits, which order as |x| does, with Inf and
// NaN above every finite value. Unsigned max and min of them vectorize where
// floating-point ones do not, and they give each value's exponent exactly.
static inline uint32_t ab_abs_hiword(double x) {
    uint64_t b;
    memcpy(&b, &x, sizeof b);
    return (uint32_t)((b & 0x7FFFFFFFFFFFFFFFULL) >> 32);
}

#define AB_HIWORD_NONFINITE 0x7FF00000u     // high words from here up: Inf or NaN

// ilogb(|x|) for a finite |x| with high word h != 0; -1075 for h == 0 (|x| < 2^-1042).
static inline int ab_hiword_ilogb(uint32_t h) {
    if (h >= 0x00100000u) return (int)(h >> 20) - 1023;
    return h ? (31 - __builtin_clz(h)) - 1042 : -1075;
}

// |x| truncated to its high word, as a double (within 2^-20 of |x|).
static inline double ab_hiword_value(uint32_t h) {
    uint64_t b = (uint64_t)h << 32;
    double d;
    memcpy(&d, &b, sizeof d);
    return d;
}

// The magnitudes of some FP64 values, as high words: of the largest |x| (0 if
// all are zero; AB_HIWORD_NONFINITE or more if any is NaN or Inf) and of the
// smallest |x| with a nonzero high word (UINT32_MAX if none).
typedef struct {
    uint32_t max, min;
} ab_range_t;

#define AB_RANGE_EMPTY ((ab_range_t){0u, UINT32_MAX})

static inline bool ab_range_nonfinite(ab_range_t r) { return r.max >= AB_HIWORD_NONFINITE; }

// Adds high word h to a maximum mx, or to a minimum mn as in ab_range_t
// (ignoring 0), or to both, in selects that vectorize.
#define AB_HIWORD_MAX(h, mx) do {                          \
        uint32_t h_ = (h);                                 \
        (mx) = h_ > (mx) ? h_ : (mx);                      \
    } while (0)
#define AB_HIWORD_MIN(h, mn) do {                          \
        uint32_t z_ = (h);                                 \
        z_ = z_ ? z_ : UINT32_MAX;                         \
        (mn) = z_ < (mn) ? z_ : (mn);                      \
    } while (0)
#define AB_HIWORD_ADD(h, mx, mn) do {                      \
        uint32_t hw_ = (h);                                \
        AB_HIWORD_MAX(hw_, mx);                            \
        AB_HIWORD_MIN(hw_, mn);                            \
    } while (0)
#define AB_RANGE_ADD(x, mx, mn) AB_HIWORD_ADD(ab_abs_hiword(x), mx, mn)

// =============================================================================
// The BLAS ABI's exponent-range guard (blas_wrapper.c)
// =============================================================================
// For C = alpha op(A) op(B) + beta C, op(A) M x K and op(B) K x N, from the
// caller's column-major operands:
//   1. Statistics of each row of op(A) and column of op(B) (ab_line_stats_scan,
//      or the pack that gathers the same).
//   2. Scales (ab_line_scales): a line whose largest |x| lies outside
//      [2^-AB_SCALE_WINDOW, 2^AB_SCALE_WINDOW) is multiplied by 2^-e on its way
//      to DD, e = ilogb(largest |x|) - AB_SCALE_TARGET; any other line by 1. A
//      DGEMM alpha (beta) outside [2^-AB_SCALAR_WINDOW, 2^AB_SCALAR_WINDOW] is
//      multiplied by 2^-ea (2^-eb), ea = ab_scalar_exp(alpha, AB_SCALAR_WINDOW).
//   3. Certification (ab_range_guard, after ab_line_scales): the call goes to
//      Accelerate (the "range" fallback) if a line has NaN or Inf, or if the
//      scaled sums could reach 2^AB_SUM_MAX_EXP. It is certified if every
//      nonzero scaled product, times alpha's scale when that is below 1, is at
//      least 2^AB_PRODUCT_MIN_EXP (stage 1). Otherwise stage 2 checks, for each
//      C_ij of a row and column that stage 1 leaves open, the products through
//      the largest elements of that row and that column, on the operands.
//   4. A DGEMM with beta != 0 checks C as it uploads, C' = C_ij 2^-(ra[i] +
//      cb[j] + ea - eb), in the same way (ab_range_guard_c): NaN or Inf, or a
//      C' or beta' C' that could reach 2^AB_SUM_MAX_EXP, go to Accelerate;
//      where a row of op(A) or a column of op(B) is zero, beta' C' must be at
//      least 2^AB_PRODUCT_MIN_EXP.
// The unscale, bit for bit (dgemm_blas, zgemm_blas): each GPU result C'_ij
// becomes FP64 as (double)hi + (double)lo, then is multiplied by
// 2^(ra[i] + cb[j] + ea) (ea = 0 for ZGEMM) with ab_scale2, one rounding. A
// DGEMM's alpha 2^-ea and beta 2^-eb were applied on the GPU, in DD; a ZGEMM
// applies alpha and beta afterwards, in FP64 (zgemm_blas's epilogue).
#define AB_SCALE_WINDOW 20      // lines with their largest |x| in [2^-20, 2^20) are not scaled
#define AB_SCALE_TARGET 30      // a scaled line's largest |x| lands in [2^30, 2^31)
#define AB_SCALAR_WINDOW 20     // a DGEMM alpha or beta within 2^±20 keeps its scale

// A column-major operand as the BLAS ABI passes it: op(X) is rows x cols,
// op(X)[i][j] = x[j*ld + i], or x[i*ld + j] if trans (a conjugate transpose
// has the same magnitudes); complex_pairs: x holds (re, im) pairs, and ld
// counts pairs.
typedef struct {
    const double* x;
    int ld, rows, cols;
    bool trans, complex_pairs;
} ab_op_view_t;

// Statistics of the lines of op(X), its rows (op(A)) or columns (op(B)): per
// line, the high words of its largest |x| (0 for a zero line;
// AB_HIWORD_NONFINITE or more if it has NaN or Inf) and of its smallest
// nonzero |x| (UINT32_MAX for a zero line), with |x| = max(|Re x|, |Im x|)
// for complex; and the exponent of its scale (ab_line_scales). Complex
// statistics also hold the high words of each line's largest |Re x| and
// largest |Im x| (rmax, imax), from which ab_lines_rho chooses between
// Gauss-3M and 4M. The arrays are plain pointers a caller may set (a GPU
// pass, say, gathering the same per line), and the functions below read
// nothing else.
typedef struct {
    int n;
    uint32_t *max, *min;
    int* e;
    uint32_t *rmax, *imax;  // complex only, else NULL
} ab_line_stats_t;

// n lines (of a complex operand, if complex_parts), all zero so far; false
// if out of memory (free it anyway).
bool ab_line_stats_init(ab_line_stats_t* s, int n, bool complex_parts);
// n lines' rmax and imax only, zeroed (max, min and e stay NULL): what
// ab_lines_rho reads; false if out of memory (free it anyway).
bool ab_line_parts_init(ab_line_stats_t* s, int n);
void ab_line_stats_free(ab_line_stats_t* s);

// Adds the statistics of op(X)'s rows (by_row) or columns to s, as the BLAS
// ABI's pack gathers them, without packing.
void ab_line_stats_scan(const ab_op_view_t* v, bool by_row, ab_line_stats_t* s);

// Sets the lines' scale exponents s->e (step 2; 0 for a line with NaN or
// Inf). Returns s->e, or NULL if no line is scaled.
const int* ab_line_scales(ab_line_stats_t* s);

typedef enum {
    AB_GUARD_CERTIFIED,     // the GPU computes the call within DD precision
    AB_GUARD_STAGE2,        // stage 1 cannot tell: ab_range_guard checks the operands
    AB_GUARD_REJECT,        // Accelerate: NaN, Inf or overflow
} ab_guard_t;

// Step 3 without stage 2, from the statistics alone (after ab_line_scales).
// alpha_exp: ilogb of the scalar the GPU multiplies by (DGEMM: alpha 2^-ea;
// 0 for ZGEMM, whose epilogue applies alpha in FP64).
ab_guard_t ab_range_guard_stage1(int K, const ab_line_stats_t* sa, const ab_line_stats_t* sb,
                                 int alpha_exp, bool complex_products);

// Step 3: true if the GPU may compute the call. a and b are the operands sa
// and sb describe.
bool ab_range_guard(int K, const ab_line_stats_t* sa, const ab_line_stats_t* sb, int alpha_exp,
                    const ab_op_view_t* a, const ab_op_view_t* b);

// Step 4 without its entrywise part: r is the range of C as uploaded
// (C itself when nothing is scaled and ea == eb), beta_exp ilogb of the
// scalar the GPU multiplies it by (beta 2^-eb).
ab_guard_t ab_range_guard_c_stage1(const ab_line_stats_t* sa, const ab_line_stats_t* sb,
                                   ab_range_t r, int beta_exp);

// Step 4: true if the GPU may take C. C is the caller's M x N C (leading
// dimension ldC), uploaded as C_ij 2^-(ra[i] + cb[j] + shift), shift = ea - eb.
bool ab_range_guard_c(const ab_line_stats_t* sa, const ab_line_stats_t* sb, ab_range_t r,
                      int beta_exp, const double* C, int ldC, int shift);

// =============================================================================
// Complex products: Gauss-3M or 4M
// =============================================================================
// Gauss-3M forms Im(C) = (Ar + Ai)(Br + Bi) - Ar Br - Ai Bi, three real
// products instead of 4M's Ar Bi + Ai Br. Its error follows the largest of
// those products, so relative to Im(C)'s own scale |Ar||Bi| + |Ai||Br| it is
// larger than 4M's by about
//     rho(p, q) = (1 + p)(1 + q) / (p + q),  p = |Ai|/|Ar|, q = |Bi|/|Br|:
// 2 for balanced parts, about 1/(2p) when p = q << 1 (or p = q >> 1). A ZGEMM
// whose parts give rho above AB_GAUSS3M_MAX_RHO runs as 4M (docs/vv/
// LIMITATIONS.md, "Complex products"): measured on M2 Max, 3M's imaginary
// part stays within 8 u^2 of its componentwise scale up to rho = 4 (6.9 u^2
// at K = 2048) and grows as about 2.3 rho u^2 above; 4M's stays near 1 u^2.
#define AB_GAUSS3M_MAX_RHO 4.0

static inline double ab_gauss3m_rho(double p, double q) {
    if (isinf(p)) return isinf(q) ? INFINITY : 1.0 + q;
    if (isinf(q)) return 1.0 + p;
    return (1.0 + p) * (1.0 + q) / (p + q);     // Inf for p = q = 0
}

// The largest rho of a row of op(A) and a column of op(B), from their
// complex statistics (rmax, imax); 0 if op(A) or op(B) is zero.
double ab_lines_rho(const ab_line_stats_t* sa, const ab_line_stats_t* sb);

typedef enum {
    AB_ZGEMM_AUTO,      // as ab_zgemm chooses, from the parts' largest magnitudes
    AB_ZGEMM_3M,
    AB_ZGEMM_4M,
} ab_zgemm_method_t;

// The method of a BLAS-ABI or device-API ZGEMM: 4M where ab_lines_rho
// exceeds AB_GAUSS3M_MAX_RHO, else Gauss-3M. The BLAS ABI's staged path
// keeps a result only where it is 3M.
static inline ab_zgemm_method_t ab_lines_method(const ab_line_stats_t* sa,
                                                const ab_line_stats_t* sb) {
    return ab_lines_rho(sa, sb) > AB_GAUSS3M_MAX_RHO ? AB_ZGEMM_4M : AB_ZGEMM_3M;
}

// =============================================================================
// Entry points of apple_bottom.m
// =============================================================================

// C = op(A) op(B) for split-complex A (Ar, Ai) and B (Br, Bi), as ab_zgemm_ex
// computes it, by the given method.
ABStatus ab_internal_zgemm(ABTranspose transA, ABTranspose transB, ABMatrix Ar, ABMatrix Ai,
                           ABMatrix Br, ABMatrix Bi, ABMatrix Cr, ABMatrix Ci,
                           ab_zgemm_method_t method);

// Uploads row-major rows x cols data (the matrix's own shape) as
// x * 2^-(row_exp[i] + col_exp[j] + shift) into m, whose power-of-two scale
// becomes 0; NULL row_exp or col_exp counts as all zeros. Each scaled value is
// rounded once, as ldexp rounds. If range is not NULL it receives the range of
// the scaled values.
ABStatus ab_internal_upload_scaled(ABMatrix m, const double* data, const int* row_exp,
                                   const int* col_exp, int shift, ab_range_t* range);

#ifdef __cplusplus
}
#endif

#endif // AB_INTERNAL_H
