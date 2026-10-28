// =============================================================================
// blas_wrapper.c
// BLAS-compatible interface for Fortran interoperability
// =============================================================================

#include "apple_bottom.h"
#include <complex.h>
#include <limits.h>
#include <math.h>
#include <dispatch/dispatch.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <sys/param.h>
#include <Accelerate/Accelerate.h>
#include <stdio.h>

#include "ab_internal.h"
#include "blas_trans.h"
#include "dispatch.h"
#include "gpu_stage.h"
#include "profiling/blas_profiler.h"

// Thread-local record of the path the last BLAS-ABI call on this thread
// actually took, for bench harnesses and the profiler (prof_end). Only the
// two places that produce C write it: "cpu" in ab_internal_*gemm_cpu, "gpu"
// once the GPU result is in C (H-P1-2). Routing (dispatch.c) never touches
// it. Zero runtime cost.
static __thread const char *_ab_last_path = "none";
const char* ab_get_last_dispatch_path(void) { return _ab_last_path; }

// =============================================================================
// Host-side helpers shared by the BLAS-ABI entry points
// =============================================================================
// Inputs are column-major with leading dimension ld; ABMatrix is row-major.
// All index arithmetic is size_t: ld*cols can exceed 2^31 even when every
// operand is within the ABMatrix size limits.

// =============================================================================
// Exponent range: power-of-two scaling of the operands, and its guard
// =============================================================================
// DD has FP32's exponent range. Rows of op(A) and columns of op(B) whose
// largest |x| lies outside 2^±AB_SCALE_WINDOW are scaled by exact powers of
// two on their way to DD and C is scaled back; calls that no such scaling
// keeps in range, or with NaN or Inf, run on Accelerate. The rule, and the
// exact order of the unscale, are in ab_internal.h ("The BLAS ABI's
// exponent-range guard"). A power of two scales every product and partial
// sum of C_ij alike, so wherever no intermediate leaves FP32's normal range
// either way, the bits are those of the unscaled computation; data whose
// lines all lie inside the window runs exactly as before.
#define AB_RANGE_WHY "values beyond double-float's exponent range, or NaN/Inf"

bool ab_line_stats_init(ab_line_stats_t* s, int n, bool complex_parts) {
    s->n = n;
    s->max = malloc((size_t)n * sizeof *s->max);
    s->min = malloc((size_t)n * sizeof *s->min);
    s->e = malloc((size_t)n * sizeof *s->e);
    s->rmax = complex_parts ? calloc((size_t)n, sizeof *s->rmax) : NULL;
    s->imax = complex_parts ? calloc((size_t)n, sizeof *s->imax) : NULL;
    if (!s->max || !s->min || !s->e || (complex_parts && (!s->rmax || !s->imax))) return false;
    for (int i = 0; i < n; i++) { s->max[i] = 0; s->min[i] = UINT32_MAX; }
    return true;
}

bool ab_line_parts_init(ab_line_stats_t* s, int n) {
    *s = (ab_line_stats_t){.n = n};
    s->rmax = calloc((size_t)n, sizeof *s->rmax);
    s->imax = calloc((size_t)n, sizeof *s->imax);
    return s->rmax && s->imax;
}

void ab_line_stats_free(ab_line_stats_t* s) {
    free(s->max);
    free(s->min);
    free(s->e);
    free(s->rmax);
    free(s->imax);
    s->max = s->min = s->rmax = s->imax = NULL;
    s->e = NULL;
}

const int* ab_line_scales(ab_line_stats_t* s) {
    bool any = false;
    for (int i = 0; i < s->n; i++) {
        uint32_t h = s->max[i];
        int e = h == 0 || h >= AB_HIWORD_NONFINITE ? 0 : ab_hiword_ilogb(h);
        s->e[i] = e >= -AB_SCALE_WINDOW && e < AB_SCALE_WINDOW ? 0 : e - AB_SCALE_TARGET;
        any = any || s->e[i] != 0;
    }
    return any ? s->e : NULL;
}

// The exponent of nonzero line i's smallest scaled |x|.
static inline int line_min_exp(const ab_line_stats_t* s, int i) {
    return ab_hiword_ilogb(s->min[i]) - s->e[i];
}

// Adds a complex element to a line's statistics: its parts' high words to
// their maxima (rmx, imx), and the larger of them to the smallest (mn), as
// AB_RANGE_ADD adds a real element's. The line's max follows from rmx and imx
// (pack_op_complex_stats).
#define LINE_ADD_COMPLEX(re, im, rmx, imx, mn) do {                \
        uint32_t hr_ = ab_abs_hiword(re), hi_ = ab_abs_hiword(im); \
        AB_HIWORD_MAX(hr_, rmx);                                   \
        AB_HIWORD_MAX(hi_, imx);                                   \
        AB_HIWORD_MIN(hr_ > hi_ ? hr_ : hi_, mn);                  \
    } while (0)

// The range [*lo, *hi] of |Im x|/|Re x| over s's nonzero lines, each line's
// from its parts' largest magnitudes (Inf for a line whose real parts are all
// zero); false if every line is zero.
static bool line_ratios(const ab_line_stats_t* s, double* lo, double* hi) {
    *lo = INFINITY;
    *hi = 0.0;
    for (int i = 0; i < s->n; i++) {
        uint32_t r = s->rmax[i], m = s->imax[i];
        if (r == 0 && m == 0) continue;
        double p = ab_hiword_value(m) / ab_hiword_value(r);
        *lo = fmin(*lo, p);
        *hi = fmax(*hi, p);
    }
    return *lo <= *hi;
}

// In p, rho grows where q > 1 and falls where q < 1 (and in q likewise), so
// over [plo, phi] x [qlo, qhi] it is largest at a corner, and each corner is
// a row of op(A) and a column of op(B).
double ab_lines_rho(const ab_line_stats_t* sa, const ab_line_stats_t* sb) {
    double plo, phi, qlo, qhi;
    if (!line_ratios(sa, &plo, &phi) || !line_ratios(sb, &qlo, &qhi)) return 0.0;
    return fmax(fmax(ab_gauss3m_rho(plo, qlo), ab_gauss3m_rho(plo, qhi)),
                fmax(ab_gauss3m_rho(phi, qlo), ab_gauss3m_rho(phi, qhi)));
}

// Pack op(X), rows x cols, into row-major dst (statistics only if dst is
// NULL), gathering the magnitudes of op(X)'s rows (by_row) or columns into
// st. op(X)[i][j] is X[j*ld+i] for CblasNoTrans and X[i*ld+j] otherwise; each
// loop order walks X contiguously, keeping its line's statistics in registers
// or a contiguous array. For CblasNoTrans the rows go in blocks of
// AB_PACK_ROWS, so that the block's strided stores into dst stay in cache
// from one column to the next.
#define AB_PACK_ROWS 256
static void pack_op_real_stats(enum CBLAS_TRANSPOSE t, const double* X, int ld,
                               int rows, int cols, double* restrict dst, bool by_row,
                               ab_line_stats_t* st) {
    size_t r = (size_t)rows, c = (size_t)cols, l = (size_t)ld;
    uint32_t* restrict mx = st->max;
    uint32_t* restrict mn = st->min;
    if (t == CblasNoTrans) {
        for (size_t i0 = 0; i0 < r; i0 += AB_PACK_ROWS) {
            const size_t i1 = r - i0 > AB_PACK_ROWS ? i0 + AB_PACK_ROWS : r;
            for (size_t j = 0; j < c; j++) {
                const double* restrict x = X + j * l;
                if (by_row && dst) {
                    for (size_t i = i0; i < i1; i++) {
                        double v = x[i];
                        dst[i * c + j] = v;
                        AB_RANGE_ADD(v, mx[i], mn[i]);
                    }
                } else if (by_row) {
                    for (size_t i = i0; i < i1; i++) AB_RANGE_ADD(x[i], mx[i], mn[i]);
                } else {
                    // The strided stores keep the copy scalar, so the
                    // column's reduction is a pass of its own, which vectorizes.
                    if (dst) for (size_t i = i0; i < i1; i++) dst[i * c + j] = x[i];
                    uint32_t lmx = mx[j], lmn = mn[j];
                    for (size_t i = i0; i < i1; i++) AB_RANGE_ADD(x[i], lmx, lmn);
                    mx[j] = lmx;
                    mn[j] = lmn;
                }
            }
        }
    } else {
        for (size_t i = 0; i < r; i++) {
            const double* restrict x = X + i * l;
            double* restrict d = dst ? dst + i * c : NULL;
            if (by_row) {
                uint32_t lmx = mx[i], lmn = mn[i];
                if (d) {
                    for (size_t j = 0; j < c; j++) {
                        double v = x[j];
                        d[j] = v;
                        AB_RANGE_ADD(v, lmx, lmn);
                    }
                } else {
                    for (size_t j = 0; j < c; j++) AB_RANGE_ADD(x[j], lmx, lmn);
                }
                mx[i] = lmx;
                mn[i] = lmn;
            } else if (d) {
                for (size_t j = 0; j < c; j++) {
                    double v = x[j];
                    d[j] = v;
                    AB_RANGE_ADD(v, mx[j], mn[j]);
                }
            } else {
                for (size_t j = 0; j < c; j++) AB_RANGE_ADD(x[j], mx[j], mn[j]);
            }
        }
    }
}

// Pack C (rows x cols, CblasNoTrans) as pack_op_real_stats does, gathering
// the range of its values.
static void pack_c(const double* X, int ld, int rows, int cols, double* restrict dst,
                   ab_range_t* range) {
    size_t r = (size_t)rows, c = (size_t)cols, l = (size_t)ld;
    uint32_t mx = 0, mn = UINT32_MAX;
    for (size_t i0 = 0; i0 < r; i0 += AB_PACK_ROWS) {
        const size_t i1 = r - i0 > AB_PACK_ROWS ? i0 + AB_PACK_ROWS : r;
        for (size_t j = 0; j < c; j++) {
            const double* restrict x = X + j * l;
            for (size_t i = i0; i < i1; i++) dst[i * c + j] = x[i];
            for (size_t i = i0; i < i1; i++) AB_RANGE_ADD(x[i], mx, mn);
        }
    }
    *range = (ab_range_t){mx, mn};
}

// Complex variant: split op(X) into row-major real and imaginary parts
// (statistics only if re and im are NULL), conjugating for CblasConjTrans,
// and gather the lines' magnitudes into st (complex statistics).
static void pack_op_complex_stats(enum CBLAS_TRANSPOSE t, const double complex* X, int ld,
                                  int rows, int cols, double* restrict re, double* restrict im,
                                  bool by_row, ab_line_stats_t* st) {
    size_t r = (size_t)rows, c = (size_t)cols, l = (size_t)ld;
    uint32_t* restrict rx = st->rmax;
    uint32_t* restrict ix = st->imax;
    uint32_t* restrict mn = st->min;
    if (t == CblasNoTrans) {
        for (size_t i0 = 0; i0 < r; i0 += AB_PACK_ROWS) {
            const size_t i1 = r - i0 > AB_PACK_ROWS ? i0 + AB_PACK_ROWS : r;
            for (size_t j = 0; j < c; j++) {
                const double* restrict x = (const double*)(X + j * l);
                // The strided copy, then the statistics over the same
                // segment (still in L1), which vectorize on their own.
                if (re) {
                    for (size_t i = i0; i < i1; i++) {
                        re[i * c + j] = x[2 * i];
                        im[i * c + j] = x[2 * i + 1];
                    }
                }
                if (by_row) {
                    for (size_t i = i0; i < i1; i++)
                        LINE_ADD_COMPLEX(x[2 * i], x[2 * i + 1], rx[i], ix[i], mn[i]);
                } else {
                    uint32_t lrx = rx[j], lix = ix[j], lmn = mn[j];
                    for (size_t i = i0; i < i1; i++)
                        LINE_ADD_COMPLEX(x[2 * i], x[2 * i + 1], lrx, lix, lmn);
                    rx[j] = lrx;
                    ix[j] = lix;
                    mn[j] = lmn;
                }
            }
        }
    } else {
        double sign = t == CblasConjTrans ? -1.0 : 1.0;
        for (size_t i = 0; i < r; i++) {
            const double* restrict x = (const double*)(X + i * l);
            if (!re) {                  // statistics only: the sign does not matter
                if (by_row) {
                    uint32_t lrx = rx[i], lix = ix[i], lmn = mn[i];
                    for (size_t j = 0; j < c; j++)
                        LINE_ADD_COMPLEX(x[2 * j], x[2 * j + 1], lrx, lix, lmn);
                    rx[i] = lrx;
                    ix[i] = lix;
                    mn[i] = lmn;
                } else {
                    for (size_t j = 0; j < c; j++)
                        LINE_ADD_COMPLEX(x[2 * j], x[2 * j + 1], rx[j], ix[j], mn[j]);
                }
                continue;
            }
            double* restrict dr = re + i * c;
            double* restrict di = im + i * c;
            if (by_row) {
                uint32_t lrx = rx[i], lix = ix[i], lmn = mn[i];
                for (size_t j = 0; j < c; j++) {
                    double vr = x[2 * j], vi = sign * x[2 * j + 1];
                    dr[j] = vr;
                    di[j] = vi;
                    LINE_ADD_COMPLEX(vr, vi, lrx, lix, lmn);
                }
                rx[i] = lrx;
                ix[i] = lix;
                mn[i] = lmn;
            } else {
                for (size_t j = 0; j < c; j++) {
                    double vr = x[2 * j], vi = sign * x[2 * j + 1];
                    dr[j] = vr;
                    di[j] = vi;
                    LINE_ADD_COMPLEX(vr, vi, rx[j], ix[j], mn[j]);
                }
            }
        }
    }
    for (int i = 0; i < st->n; i++) AB_HIWORD_MAX(rx[i] > ix[i] ? rx[i] : ix[i], st->max[i]);
}

void ab_line_stats_scan(const ab_op_view_t* v, bool by_row, ab_line_stats_t* s) {
    const enum CBLAS_TRANSPOSE t = v->trans ? CblasTrans : CblasNoTrans;
    if (v->complex_pairs)
        pack_op_complex_stats(t, (const double complex*)v->x, v->ld, v->rows, v->cols, NULL, NULL,
                              by_row, s);
    else
        pack_op_real_stats(t, v->x, v->ld, v->rows, v->cols, NULL, by_row, s);
}

// |op(X)_ik|, unscaled; max(|Re|, |Im|) for complex.
static inline double op_mag(const ab_op_view_t* v, int i, int k) {
    const size_t x = v->trans ? (size_t)i * v->ld + k : (size_t)k * v->ld + i;
    if (!v->complex_pairs) return fabs(v->x[x]);
    return fmax(fabs(v->x[2 * x]), fabs(v->x[2 * x + 1]));
}

static inline int imax2(int a, int b) { return a > b ? a : b; }
static inline int imin2(int a, int b) { return a < b ? a : b; }

// The exponents of the largest scaled |x| over s's lines (*hi; INT_MIN if all
// are zero) and of the smallest nonzero one (*lo); false if a line has NaN or
// Inf.
static bool lines_extremes(const ab_line_stats_t* s, int* hi, int* lo) {
    *hi = INT_MIN;
    *lo = INT_MAX;
    for (int i = 0; i < s->n; i++) {
        if (s->max[i] >= AB_HIWORD_NONFINITE) return false;
        if (s->max[i] == 0) continue;
        *hi = imax2(*hi, ab_hiword_ilogb(s->max[i]) - s->e[i]);
        *lo = imin2(*lo, line_min_exp(s, i));
    }
    return true;
}

// The exponent stage 1 asks of a nonzero scaled product: AB_PRODUCT_MIN_EXP,
// and that again times alpha's scalar when it is below 1.
static inline int stage1_floor(int alpha_exp) { return AB_PRODUCT_MIN_EXP - imin2(alpha_exp, 0); }

// The scaled product op(A)' op(B)' must keep full DD precision and stay in
// FP32's range in every C_ij, times a scalar below 2^(alpha_exp + 1):
//   - Overflow: the largest scaled magnitudes bound every product and sum.
//   - Stage 1: every nonzero scaled product is at least 2^AB_PRODUCT_MIN_EXP,
//     by the smallest nonzero magnitudes of the lines.
//   - Stage 2 (ab_range_guard), for the pairs of lines stage 1 cannot tell
//     about (lines whose magnitudes span many binades): C_ij's scaled
//     |op(A)||op(B)| sum is at least |A'_ik| |B'_kj| for k = the column of
//     row i's largest |A'| and for k = the row of column j's largest |B'|;
//     one of the two must reach K * 2^AB_PRODUCT_MIN_EXP, so that the at most
//     K flushed parts below 2^-126 (about 2^-48 of 2^AB_PRODUCT_MIN_EXP) stay
//     at DD's rounding level.
// Rows of op(A) and columns of op(B) that are zero, and C_ij with nonzero
// lines whose terms all vanish, are exact. Pairs with disjoint supports that
// stage 1 passes are not checked further (their C_ij is 0 either way).
ab_guard_t ab_range_guard_stage1(int K, const ab_line_stats_t* sa, const ab_line_stats_t* sb,
                                 int alpha_exp, bool complex_products) {
    int amax, amin, bmax, bmin;
    if (!lines_extremes(sa, &amax, &amin) || !lines_extremes(sb, &bmax, &bmin))
        return AB_GUARD_REJECT;
    if (amax == INT_MIN || bmax == INT_MIN) return AB_GUARD_CERTIFIED;   // op(A) or op(B) is zero
    // A complex product's parts (and 3M's sums of them) are at most 4x the
    // magnitudes' product; alpha's scalar is below 2^(alpha_exp + 1).
    int headroom = (complex_products ? 2 : 0) + 2 + ilogb((double)K) + 1 + alpha_exp + 1;
    if (amax + bmax + headroom > AB_SUM_MAX_EXP) return AB_GUARD_REJECT;
    return amin + bmin >= stage1_floor(alpha_exp) ? AB_GUARD_CERTIFIED : AB_GUARD_STAGE2;
}

bool ab_range_guard(int K, const ab_line_stats_t* sa, const ab_line_stats_t* sb, int alpha_exp,
                    const ab_op_view_t* a, const ab_op_view_t* b) {
    ab_guard_t g = ab_range_guard_stage1(K, sa, sb, alpha_exp, a->complex_pairs);
    if (g != AB_GUARD_STAGE2) return g == AB_GUARD_CERTIFIED;

    // Stage 2, on the operands as scaled, for the rows and columns with a
    // partner that stage 1 does not pass: the column of each such row's
    // largest element, the row of each such column's, then the pairs.
    const int M = sa->n, N = sb->n, lo = stage1_floor(alpha_exp);
    int amax, amin, bmax, bmin;
    lines_extremes(sa, &amax, &amin);
    lines_extremes(sb, &bmax, &bmin);
    const double floor_ij = fmax(ldexp((double)K, AB_PRODUCT_MIN_EXP),
                                 ldexp(1.0, AB_PRODUCT_MIN_EXP - alpha_exp));
    int *rows = malloc((size_t)M * sizeof *rows), *ka = malloc((size_t)M * sizeof *ka);
    int *cols = malloc((size_t)N * sizeof *cols), *kb = malloc((size_t)N * sizeof *kb);
    double *av = malloc((size_t)M * sizeof *av), *bv = malloc((size_t)N * sizeof *bv);
    bool ok = rows && ka && cols && kb && av && bv;
    int nr = 0, nc = 0;
    for (int i = 0; ok && i < M; i++)
        if (sa->max[i] != 0 && line_min_exp(sa, i) + bmin < lo) rows[nr++] = i;
    for (int j = 0; ok && j < N; j++)
        if (sb->max[j] != 0 && line_min_exp(sb, j) + amin < lo) cols[nc++] = j;
    for (int r = 0; r < nr; r++) {
        double best = -1.0;
        for (int k = 0; k < K; k++) {
            double m = op_mag(a, rows[r], k);
            if (m > best) { best = m; ka[r] = k; }
        }
        av[r] = ldexp(best, -sa->e[rows[r]]);
    }
    for (int c = 0; c < nc; c++) {
        double best = -1.0;
        for (int k = 0; k < K; k++) {
            double m = op_mag(b, k, cols[c]);
            if (m > best) { best = m; kb[c] = k; }
        }
        bv[c] = ldexp(best, -sb->e[cols[c]]);
    }
    for (int r = 0; ok && r < nr; r++) {
        const int i = rows[r];
        for (int c = 0; ok && c < nc; c++) {
            const int j = cols[c];
            if (line_min_exp(sa, i) + line_min_exp(sb, j) >= lo) continue;
            ok = av[r] * ldexp(op_mag(b, ka[r], j), -sb->e[j]) >= floor_ij ||
                 ldexp(op_mag(a, i, kb[c]), -sa->e[i]) * bv[c] >= floor_ij;
        }
    }
    free(rows); free(ka); free(cols); free(kb); free(av); free(bv);
    return ok;
}

// The DGEMM's beta' C', with C' the C scaled as uploaded (r: the range of its
// values) and |beta'| below 2^(beta_exp + 1), must stay in FP32's range and,
// where the product part of C_ij is 0 for want of products (a zero row of
// op(A) or column of op(B)), keep its full DD precision. Elsewhere a C' far
// below the product part only rounds away, as it would in FP64.
ab_guard_t ab_range_guard_c_stage1(const ab_line_stats_t* sa, const ab_line_stats_t* sb,
                                   ab_range_t r, int beta_exp) {
    if (ab_range_nonfinite(r)) return AB_GUARD_REJECT;
    if (r.max == 0) return AB_GUARD_CERTIFIED;
    // C' is stored in DD before beta' multiplies it, and beta' C' joins the sums.
    if (ab_hiword_ilogb(r.max) + imax2(beta_exp, 0) + 1 > AB_SUM_MAX_EXP) return AB_GUARD_REJECT;
    if (ab_hiword_ilogb(r.min) + beta_exp >= AB_PRODUCT_MIN_EXP) return AB_GUARD_CERTIFIED;
    for (int i = 0; i < sa->n; i++) if (sa->max[i] == 0) return AB_GUARD_STAGE2;
    for (int j = 0; j < sb->n; j++) if (sb->max[j] == 0) return AB_GUARD_STAGE2;
    return AB_GUARD_CERTIFIED;
}

// Whether beta' times C'_ij = C_ij 2^-e keeps full DD precision.
static inline bool c_entry_ok(double c, int e, int beta_exp) {
    double v = ldexp(fabs(c), -e);
    return v == 0.0 || ilogb(v) + beta_exp >= AB_PRODUCT_MIN_EXP;
}

bool ab_range_guard_c(const ab_line_stats_t* sa, const ab_line_stats_t* sb, ab_range_t r,
                      int beta_exp, const double* C, int ldC, int shift) {
    ab_guard_t g = ab_range_guard_c_stage1(sa, sb, r, beta_exp);
    if (g != AB_GUARD_STAGE2) return g == AB_GUARD_CERTIFIED;
    const int M = sa->n, N = sb->n;
    for (int i = 0; i < M; i++)
        if (sa->max[i] == 0)
            for (int j = 0; j < N; j++)
                if (!c_entry_ok(C[(size_t)j * ldC + i], sa->e[i] + sb->e[j] + shift, beta_exp))
                    return false;
    for (int j = 0; j < N; j++)
        if (sb->max[j] == 0)
            for (int i = 0; i < M; i++)
                if (!c_entry_ok(C[(size_t)j * ldC + i], sa->e[i] + sb->e[j] + shift, beta_exp))
                    return false;
    return true;
}

// The gate of dgemm_blas and zgemm_blas (dispatch.h): Accelerate where the
// routing says so, and for the BLAS quick return for alpha == 0 (A and B must
// not be read) and invalid arguments.
bool ab_internal_blas_routes_gpu(char routine, char transA, char transB, int M, int N, int K,
                                 int ldA, int ldB, int ldC, bool alpha_nonzero) {
    bool z = routine == 'z';
    return ab_blas_args_valid(ab_blas_trans(transA, z), ab_blas_trans(transB, z),
                              M, N, K, ldA, ldB, ldC) &&
           alpha_nonzero && ab_dispatch_route_gpu(routine, transA, transB, M, N, K);
}

// Whole-call CPU paths (Accelerate cblas, column-major). Used for calls the
// routing keeps on Accelerate, for BLAS quick returns and invalid arguments,
// and as the fallback when any GPU step fails before C is written, so C
// always gets a correct result.
static void ab_internal_dgemm_cpu(enum CBLAS_TRANSPOSE ta, enum CBLAS_TRANSPOSE tb,
                                  int M, int N, int K, double alpha,
                                  const double* A, int ldA, const double* B, int ldB,
                                  double beta, double* C, int ldC) {
    _ab_last_path = "cpu";
    cblas_dgemm(CblasColMajor, ta, tb, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}

static void ab_internal_zgemm_cpu(enum CBLAS_TRANSPOSE ta, enum CBLAS_TRANSPOSE tb,
                                  int M, int N, int K, double complex alpha,
                                  const double complex* A, int ldA,
                                  const double complex* B, int ldB,
                                  double complex beta, double complex* C, int ldC) {
    _ab_last_path = "cpu";
    cblas_zgemm(CblasColMajor, ta, tb, M, N, K, &alpha, A, ldA, B, ldB, &beta, C, ldC);
}

// =============================================================================
// Call profile (AB_PROFILE, AB_REPORT, AB_TRACE)
// =============================================================================
// Every BLAS-ABI call is timed once, around routing and compute, and recorded
// by where C was computed (the path record). The GPU path charges each step
// to a stage as it finishes. p is NULL when profiling and tracing are off,
// so such a call pays one branch per stage.

typedef struct {
    uint64_t t0, mark;                         // ticks: call start, end of the last stage
    uint64_t stage_ns[AB_PROF_STAGE_COUNT];
    ab_prof_fallback_t fallback;               // why a GPU-routed call ran on cblas
    unsigned staged;                           // whether it kept the staged result, or why not
                                               // (ab_prof_staged_t bits; 0: not tried)
    bool trace;                                // AB_TRACE: write the call's trace line
    ab_trace_call_t call;                      // its arguments, hashes and kernel split
} CallProf;

static CallProf *prof_begin(CallProf *p) {
    if (!ab_prof_active()) return NULL;
    memset(p, 0, sizeof *p);
    p->fallback = AB_PROF_FB_NONE;
    p->trace = ab_trace_enabled();
    p->t0 = p->mark = ab_prof_now_ticks();
    (void)ab_internal_take_gpu_seconds(NULL); // drop GPU time of earlier native-API calls
    return p;
}

static void prof_stage(CallProf *p, ab_prof_stage_t s) {
    if (!p) return;
    uint64_t now = ab_prof_now_ticks();
    p->stage_ns[s] += ab_prof_ticks_to_ns(now - p->mark);
    p->mark = now;
}

// The kernel stage (ab_prof_take_kernel_ns); the rest of the kernel call is
// left to "other", which AB_TRACE splits. A call whose staged attempt gave
// way to the host path has two kernel steps, which add up.
static void prof_kernel(CallProf *p) {
    if (!p) return;
    uint64_t now = ab_prof_now_ticks();
    p->stage_ns[AB_PROF_STAGE_KERNEL] +=
        ab_prof_take_kernel_ns(ab_prof_ticks_to_ns(now - p->mark), &p->call.split);
    p->mark = now;
}

// Records the finished call; under AB_PROFILE also prints the stages of a GPU
// call, under AB_TRACE writes its trace line.
static void prof_end(CallProf *p, const char *routine, int M, int N, int K) {
    if (!p) return;
    uint64_t ns = ab_prof_ticks_to_ns(ab_prof_now_ticks() - p->t0);
    bool gpu = !strcmp(_ab_last_path, "gpu");
    ab_prof_record(routine, ns, M > 0 ? (size_t)M : 0, N > 0 ? (size_t)N : 0,
                   K > 0 ? (size_t)K : 0, gpu, gpu ? p->stage_ns : NULL, p->fallback, p->staged);
    if (p->trace)
        ab_trace_record(&p->call, ab_prof_ticks_to_ns(p->t0), ns, gpu ? p->stage_ns : NULL,
                        p->fallback, p->staged);
    if (!gpu || !ab_prof_verbose()) return;
    // One write per line, so lines from concurrent callers do not interleave.
    char line[512];
    size_t n = (size_t)snprintf(line, sizeof line, "AB_PROFILE %s M=%d N=%d K=%d ", routine, M, N, K);
    uint64_t named = 0;
    for (int s = 0; s < AB_PROF_STAGE_COUNT && n < sizeof line; s++) {
        named += p->stage_ns[s];
        n += (size_t)snprintf(line + n, sizeof line - n, " %s=%.3f", ab_prof_stage_names[s],
                              p->stage_ns[s] / 1e6);
    }
    if (n < sizeof line)
        snprintf(line + n, sizeof line - n, " other=%.3f  total=%.3f ms  staged=%s\n",
                 (ns > named ? ns - named : 0) / 1e6, ns / 1e6, ab_prof_staged_name(p->staged));
    fputs(line, stderr);
}

// Adds reason r to the call's staged-path record (CallProf.staged).
static void note_staged(CallProf *p, ab_prof_staged_t r) {
    if (p) p->staged |= 1u << r;
}

// A GPU-routed call that has to run on cblas instead. The path record says
// "cpu" either way; the profile counts the reason and AB_PROFILE prints it,
// so a fallback is not mistaken for a routing decision (H-P2-8).
static void note_gpu_fallback(CallProf *p, const char *routine, int M, int N, int K,
                              ab_prof_fallback_t reason, const char *why) {
    if (!p) return;
    p->fallback = reason;
    if (ab_prof_verbose())
        fprintf(stderr, "AB_PROFILE %s M=%d N=%d K=%d  GPU fallback to cblas: %s\n",
                routine, M, N, K, why);
}

// FNV-1a over the 64-bit words of the referenced elements of a stored
// rows x cols operand (column-major, leading dimension ld, elem = 8 or 16
// bytes), column by column: from the 64-bit offset basis, h = (h ^ word) *
// FNV prime per word. The padding below row `rows` is not read.
static uint64_t hash_stored(const void *X, size_t elem, int ld, int rows, int cols) {
    const unsigned char *x = X;
    size_t col_words = (size_t)rows * (elem / sizeof(uint64_t));
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t j = 0; j < (size_t)cols; j++) {
        const unsigned char *col = x + j * (size_t)ld * elem;
        for (size_t w = 0; w < col_words; w++) {
            uint64_t word;
            memcpy(&word, col + w * sizeof word, sizeof word);
            h = (h ^ word) * 0x100000001b3ULL;
        }
    }
    return h;
}

// AB_TRACE: the call's arguments as passed and, under AB_TRACE_HASH, the
// hashes of what it reads: op(A) and op(B) as stored unless the BLAS quick
// return skips them (M, N or K == 0, alpha == 0), and C on entry when
// beta != 0. Invalid arguments are not hashed (cblas rejects the call). The
// clock restarts after hashing, so the hashes cost the call nothing.
static void trace_begin(CallProf *p, const char *routine, char transA, char transB,
                        int M, int N, int K, const double alpha[2],
                        const void *A, int ldA, const void *B, int ldB,
                        const double beta[2], const void *C, int ldC) {
    bool z = routine[0] == 'z';
    ab_trace_call_t *c = &p->call;
    c->routine = routine;
    c->ta = transA;
    c->tb = transB;
    c->m = M; c->n = N; c->k = K;
    c->lda = ldA; c->ldb = ldB; c->ldc = ldC;
    memcpy(c->alpha, alpha, sizeof c->alpha);
    memcpy(c->beta, beta, sizeof c->beta);
    if (!ab_trace_hash_enabled()) return;

    enum CBLAS_TRANSPOSE ta = ab_blas_trans(transA, z), tb = ab_blas_trans(transB, z);
    size_t elem = z ? sizeof(double complex) : sizeof(double);
    if (ab_blas_args_valid(ta, tb, M, N, K, ldA, ldB, ldC) && M > 0 && N > 0) {
        if (K > 0 && (alpha[0] != 0.0 || alpha[1] != 0.0)) {
            bool na = ta == CblasNoTrans, nb = tb == CblasNoTrans;
            c->hash[0] = hash_stored(A, elem, ldA, na ? M : K, na ? K : M);
            c->hash[1] = hash_stored(B, elem, ldB, nb ? K : N, nb ? N : K);
            c->hashed[0] = c->hashed[1] = true;
        }
        if (beta[0] != 0.0 || beta[1] != 0.0) {
            c->hash[2] = hash_stored(C, elem, ldC, M, N);
            c->hashed[2] = true;
        }
    }
    p->t0 = p->mark = ab_prof_now_ticks();
}

// Hosts that reach us through dgemm_/zgemm_ (QE, Yambo) never call ab_init(),
// so create the Metal context on first GPU use (H-P1-2); ab_init() is
// idempotent and thread-safe. It fails for no device and in a fork() child.
// Then op(A) (M x K), op(B) (K x N) and C (M x N) must each fit the ABMatrix
// size limits.
static ABStatus gpu_ready(int M, int N, int K) {
    ABStatus st = ab_init();
    if (st == AB_OK) st = ab_matrix_check_size(M, K);
    if (st == AB_OK) st = ab_matrix_check_size(K, N);
    if (st == AB_OK) st = ab_matrix_check_size(M, N);
    return st;
}

static ab_prof_fallback_t fallback_reason(ABStatus st) {
    return st == AB_ERROR_NO_DEVICE ? AB_PROF_FB_NO_DEVICE
         : st == AB_ERROR_TOO_LARGE ? AB_PROF_FB_TOO_LARGE : AB_PROF_FB_OTHER;
}

// The ZGEMM epilogue of the host path, C = alpha * P + beta * C, from the
// product P = op(A) * op(B) as row-major real and imaginary parts (M x N).
// Bit for bit (ab_internal.h): with the line scales re (rows of op(A)) and
// ce (columns of op(B)), both NULL if none, each part of the product, as the
// download made it FP64, times 2^(re[row] + ce[col]) in one rounding
// (ab_scale2), then alpha and beta in FP64, in the order below.
static void zgemm_epilogue(int M, int N, double complex alpha, double complex beta,
                           const double *Cr_data, const double *Ci_data,
                           double complex *C, int ldC, const int *re, const int *ce) {
    double ar = creal(alpha), ai = cimag(alpha);
    double br = creal(beta), bi = cimag(beta);
    bool needs_beta = (br != 0.0 || bi != 0.0);
    bool non_unit_alpha = (ar != 1.0 || ai != 0.0);

    for (size_t col = 0; col < (size_t)N; col++) {
        const int ec = ce ? ce[col] : 0;
        for (size_t row = 0; row < (size_t)M; row++) {
            size_t src = row * N + col;
            size_t dst = col * ldC + row;
            double pr = Cr_data[src], pi = Ci_data[src];
            if (re) {
                pr = ab_scale2(pr, re[row] + ec);
                pi = ab_scale2(pi, re[row] + ec);
            }

            // Apply alpha: alpha * (A*B)
            double result_r, result_i;
            if (non_unit_alpha) {
                result_r = ar * pr - ai * pi;
                result_i = ar * pi + ai * pr;
            } else {
                result_r = pr;
                result_i = pi;
            }

            // Apply beta: + beta * C_old
            if (needs_beta) {
                double complex c_old = C[dst];
                result_r += br * creal(c_old) - bi * cimag(c_old);
                result_i += br * cimag(c_old) + bi * creal(c_old);
            }

            C[dst] = result_r + I * result_i;
        }
    }
}

// =============================================================================
// Staged path (gpu_stage.h)
// =============================================================================
// A GPU-routed call first copies its operands into reused GPU buffers and
// leaves packing, FP64 <-> DD conversion and the epilogue to the GPU (the
// host epilogue for a ZGEMM with other alpha or beta); gpu_stage.h has the
// whole of it, and ab_prof_stage_t what its profile stages hold. A call the
// exponent-range guard does not let stand, and one that fails or meets Inf
// or NaN, takes the host path below instead.

// This thread's last call: whether the staged path gave its result, and
// with how many operands in place (ab_internal_last_staged).
static __thread struct { bool staged; int in_place; } t_last_stage;

bool ab_internal_last_staged(int *in_place) {
    if (in_place) *in_place = t_last_stage.in_place;
    return t_last_stage.staged;
}

// The staged path converts on the GPU what the host path converts on the
// CPU, so the bits agree only under the CPU's default floating-point
// control: round to nearest, no flush to zero, no default NaN (FPCR RMode,
// FZ, DN, and FIZ/AH where implemented, all clear). A caller that changed it
// keeps the host path, as does AB_HOST_PACK=1.
static bool staged_path(CallProf *p) {
    ab_prof_staged_t why = AB_PROF_STAGED_KEPT;
    if (ab_host_pack()) why = AB_PROF_STAGED_HOST_PACK;
#if defined(__aarch64__)
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    if (why == AB_PROF_STAGED_KEPT && (fpcr & ((3ULL << 22) | (1ULL << 24) | (1ULL << 25) | 3ULL)))
        why = AB_PROF_STAGED_FPCR;
#endif
    if (why == AB_PROF_STAGED_KEPT) return true;
    note_staged(p, why);
    return false;
}

// Copies of at least STAGE_COPY_PARALLEL bytes run in STAGE_COPY_CHUNKS
// pieces on the global queue (~54 GB/s on one thread, ~100 on several; cf.
// AB_CONVERT_PARALLEL_MIN in ab_internal.h for the host path's conversion).
#define STAGE_COPY_PARALLEL ((size_t)4 << 20)
#define STAGE_COPY_CHUNKS 4

typedef struct {
    char *dst;
    const char *src;
    size_t dst_ld, src_ld, col_bytes, cols;   // cols columns of col_bytes, strides in bytes
    size_t per;                               // bytes (one column) or columns per piece
} CopyJob;

static void copy_piece(void *arg, size_t i) {
    const CopyJob *j = arg;
    if (j->cols == 1) {
        size_t off = i * j->per;
        memcpy(j->dst + off, j->src + off, MIN(j->per, j->col_bytes - off));
        return;
    }
    for (size_t c = i * j->per, end = MIN(c + j->per, j->cols); c < end; c++)
        memcpy(j->dst + c * j->dst_ld, j->src + c * j->src_ld, j->col_bytes);
}

// rows x cols elements of elem bytes from src (leading dimension src_ld) to
// dst (dst_ld): one block when both are dense, else column by column, so
// padding is neither read nor written.
static void copy_matrix(void *dst, int dst_ld, const void *src, int src_ld, size_t elem,
                        int rows, int cols) {
    size_t c = (size_t)cols, col_bytes = (size_t)rows * elem;
    CopyJob j = {dst, src, (size_t)dst_ld * elem, (size_t)src_ld * elem, col_bytes, c, 0};
    if ((dst_ld == rows && src_ld == rows) || cols == 1) {
        j.col_bytes = col_bytes * c;
        j.cols = 1;
    }
    size_t pieces = col_bytes * c >= STAGE_COPY_PARALLEL ? STAGE_COPY_CHUNKS : 1;
    if (j.cols == 1) {
        j.per = (j.col_bytes / pieces + 63) & ~(size_t)63;
        pieces = (j.col_bytes + j.per - 1) / j.per;
    } else {
        j.per = (c + pieces - 1) / pieces;
        pieces = (c + j.per - 1) / j.per;
    }
    if (pieces == 1)
        copy_piece(&j, 0);
    else
        dispatch_apply_f(pieces, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &j,
                         copy_piece);
}

// Whether the exponent-range guard (see "Exponent range") lets the staged
// result stand: exactly where the host path would compute the call on the
// GPU, as the staged path did. From the statistics the GPU gathered: the
// lines' scales the host path would give them, which the GPU applied, the
// product certified, and a DGEMM's C when it is read, stage 2 reading the
// caller's operands as the host path does (the staged run wrote none of them
// it reads). A call it does not certify runs on Accelerate, as the host
// path's guard, from the same statistics, would send it. (A DGEMM's alpha
// and beta were checked before the run: dgemm_staged.) Returns the
// ab_prof_staged_t bit of the reason it does not (0 if it does). Also returns
// whether a ZGEMM's parts call for 4M (ab_lines_method) and the lines'
// scales, *re for op(A)'s rows and *ce for op(B)'s columns (NULL if none).
static unsigned staged_refusals(ab_stage_t *s, const ab_stage_gemm_t *g, bool *four_m,
                                const int **re, const int **ce) {
    ab_line_stats_t sa, sb;
    ab_range_t c;
    const int *gpu;
    ab_internal_stage_stats(s, g, &sa, &sb, &c, &gpu);
    bool z = g->routine == 'z';
    const ab_stage_matrix_t *A = &g->x[0], *B = &g->x[1], *C = &g->x[2];
    const ab_op_view_t va = {A->ptr, A->ld, g->M, g->K, g->ta != 'N', z};
    const ab_op_view_t vb = {B->ptr, B->ld, g->K, g->N, g->tb != 'N', z};
    *four_m = z && ab_lines_method(&sa, &sb) == AB_ZGEMM_4M;
    bool scaled = (ab_line_scales(&sa) != NULL) | (ab_line_scales(&sb) != NULL);
    *re = scaled ? sa.e : NULL;
    *ce = scaled ? sb.e : NULL;
    if (memcmp(sa.e, gpu, (size_t)g->M * sizeof *gpu) ||
        memcmp(sb.e, gpu + g->M, (size_t)g->N * sizeof *gpu))
        return 1u << AB_PROF_STAGED_SCALED;
    if (!ab_range_guard(g->K, &sa, &sb, z ? 0 : ilogb(g->alpha[0]), &va, &vb) ||
        (!z && g->c_in && !ab_range_guard_c(&sa, &sb, c, ilogb(g->beta[0]), C->ptr, C->ld, 0)))
        return 1u << AB_PROF_STAGED_RANGE;
    return 0;
}

static unsigned stage_result_why(ab_stage_result_t r) {
    return r == AB_STAGE_FAILED ? 1u << AB_PROF_STAGED_GPU_FAILED
         : r == AB_STAGE_NONFINITE ? 1u << AB_PROF_STAGED_NONFINITE : 0;
}

static double ms_since(uint64_t ticks) {
    return ab_prof_ticks_to_ns(ab_prof_now_ticks() - ticks) / 1e6;
}

// One staged call, g describing it: 0 if C holds the result; else the
// ab_prof_staged_t bits of why not, C as it was.
static unsigned run_staged(CallProf *p, const ab_stage_gemm_t *g) {
    size_t elem = g->routine == 'z' ? sizeof(double complex) : sizeof(double);
    ab_stage_t *s = ab_internal_stage_acquire(g);
    prof_stage(p, AB_PROF_STAGE_ALLOC);
    if (!s) {
        note_staged(p, AB_PROF_STAGED_ALLOC);
        return 1u << AB_PROF_STAGED_ALLOC;
    }

    uint64_t t = ab_prof_now_ticks();
    for (int i = 0; i < 2 + g->c_in; i++) {
        const ab_stage_matrix_t *x = &g->x[i];
        int ld;
        void *dst = ab_internal_stage_host(s, (ab_stage_operand_t)i, &ld);
        if (dst) copy_matrix(dst, ld, x->ptr, x->ld, elem, x->rows, x->cols);
    }
    double in_ms = ms_since(t);
    prof_stage(p, AB_PROF_STAGE_UPLOAD);

    bool four_m = false;
    const int *re = NULL, *ce = NULL;
    unsigned why = stage_result_why(ab_internal_stage_run(s, g));
    prof_kernel(p);
    if (!why) why = staged_refusals(s, g, &four_m, &re, &ce);
    prof_stage(p, AB_PROF_STAGE_PACK);
    if (!why && four_m) {
        // 4M instead of the run's Gauss-3M, on the GPU (gpu_stage.h), C
        // copied in again where the epilogue reads it.
        if (g->c_in) {
            int ld;
            t = ab_prof_now_ticks();
            copy_matrix(ab_internal_stage_host(s, AB_STAGE_C, &ld), ld, g->x[2].ptr, g->x[2].ld,
                        elem, g->M, g->N);
            in_ms += ms_since(t);
            prof_stage(p, AB_PROF_STAGE_UPLOAD);
        }
        why = stage_result_why(ab_internal_stage_redo_4m(s, g));
        prof_kernel(p);
    }
    if (p) p->staged |= why ? why : 1u << (four_m ? AB_PROF_STAGED_KEPT_4M : AB_PROF_STAGED_KEPT);
    bool ok = !why;
    if (ok) {
        void *C = (void *)g->x[2].ptr;
        int ldC = g->x[2].ld, ld;
        t = ab_prof_now_ticks();
        if (g->split) {
            const double *Cr_data, *Ci_data;
            ab_internal_stage_split(s, g, &Cr_data, &Ci_data);
            double out_ms = ms_since(t);
            prof_stage(p, AB_PROF_STAGE_DOWNLOAD);
            zgemm_epilogue(g->M, g->N, CMPLX(g->alpha[0], g->alpha[1]),
                           CMPLX(g->beta[0], g->beta[1]), Cr_data, Ci_data, C, ldC, re, ce);
            prof_stage(p, AB_PROF_STAGE_EPILOGUE);
            ab_internal_stage_count(g, in_ms, out_ms);
        } else {
            const void *out = ab_internal_stage_host(s, AB_STAGE_C, &ld);
            if (out) copy_matrix(C, ldC, out, ld, elem, g->M, g->N);
            ab_internal_stage_count(g, in_ms, ms_since(t));
            prof_stage(p, AB_PROF_STAGE_DOWNLOAD);
        }
        t_last_stage.staged = true;
        t_last_stage.in_place = ab_internal_stage_in_place(s);
    }
    ab_internal_stage_release(s);
    prof_stage(p, AB_PROF_STAGE_ALLOC);
    return why;
}

// op(X) is op_rows x op_cols: X as stored.
static ab_stage_matrix_t stored_matrix(const void *X, int ld, enum CBLAS_TRANSPOSE t, int op_rows,
                                       int op_cols) {
    bool n = t == CblasNoTrans;
    return (ab_stage_matrix_t){X, ld, n ? op_rows : op_cols, n ? op_cols : op_rows};
}

static char op_char(enum CBLAS_TRANSPOSE t) {
    return t == CblasNoTrans ? 'N' : t == CblasConjTrans ? 'C' : 'T';
}

static unsigned zgemm_staged(CallProf *p, enum CBLAS_TRANSPOSE ta, enum CBLAS_TRANSPOSE tb,
                         int M, int N, int K, double complex alpha,
                         const double complex *A, int ldA, const double complex *B, int ldB,
                         double complex beta, double complex *C, int ldC) {
    bool beta0 = creal(beta) == 0.0 && cimag(beta) == 0.0;
    bool beta1 = creal(beta) == 1.0 && cimag(beta) == 0.0;
    bool gpu_epi = creal(alpha) == 1.0 && cimag(alpha) == 0.0 && (beta0 || beta1);
    ab_stage_gemm_t g = {
        .routine = 'z', .ta = op_char(ta), .tb = op_char(tb), .M = M, .N = N, .K = K,
        .x = {stored_matrix(A, ldA, ta, M, K), stored_matrix(B, ldB, tb, K, N), {C, ldC, M, N}},
        .c_in = gpu_epi && beta1, .split = !gpu_epi,
        .alpha = {creal(alpha), cimag(alpha)}, .beta = {creal(beta), cimag(beta)},
    };
    return run_staged(p, &g);
}

static unsigned dgemm_staged(CallProf *p, enum CBLAS_TRANSPOSE ta, enum CBLAS_TRANSPOSE tb,
                         int M, int N, int K, double alpha, const double *A, int ldA,
                         const double *B, int ldB, double beta, double *C, int ldC) {
    // An alpha or beta the host path would fold into C's scale, or one it
    // rejects, is its own to handle.
    if (!isfinite(alpha) || !isfinite(beta) || ab_scalar_exp(alpha, AB_SCALAR_WINDOW) ||
        ab_scalar_exp(beta, AB_SCALAR_WINDOW)) {
        note_staged(p, AB_PROF_STAGED_SCALAR);
        return 1u << AB_PROF_STAGED_SCALAR;
    }
    ab_stage_gemm_t g = {
        .routine = 'd', .ta = op_char(ta), .tb = op_char(tb), .M = M, .N = N, .K = K,
        .x = {stored_matrix(A, ldA, ta, M, K), stored_matrix(B, ldB, tb, K, N), {C, ldC, M, N}},
        .c_in = beta != 0.0, .alpha = {alpha, 0.0}, .beta = {beta, 0.0},
    };
    return run_staged(p, &g);
}

// =============================================================================
// BLAS-compatible ZGEMM - this is what Fortran calls
// Handles: transA, transB, alpha, beta, leading dimensions
// =============================================================================
static void zgemm_blas(
    CallProf *p,
    char transA, char transB,
    int M, int N, int K,
    double complex alpha,
    const double complex* A, int ldA,
    const double complex* B, int ldB,
    double complex beta,
    double complex* C, int ldC
) {
    enum CBLAS_TRANSPOSE ta = ab_blas_trans(transA, true);
    enum CBLAS_TRANSPOSE tb = ab_blas_trans(transB, true);
    t_last_stage.staged = false;

    if (!ab_internal_blas_routes_gpu('z', transA, transB, M, N, K, ldA, ldB, ldC, alpha != 0.0)) {
        ab_internal_zgemm_cpu(ta, tb, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
        return;
    }

    ABStatus ready = gpu_ready(M, N, K);
    if (ready != AB_OK) {
        note_gpu_fallback(p, "zgemm", M, N, K, fallback_reason(ready), ab_status_string(ready));
        ab_internal_zgemm_cpu(ta, tb, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
        return;
    }
    prof_stage(p, AB_PROF_STAGE_INIT);
    if (staged_path(p)) {
        unsigned why = zgemm_staged(p, ta, tb, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
        if (!why) {
            _ab_last_path = "gpu";
            return;
        }
        // The guard's verdict on the same statistics is the host path's too.
        if (why & 1u << AB_PROF_STAGED_RANGE) {
            note_gpu_fallback(p, "zgemm", M, N, K, AB_PROF_FB_RANGE, AB_RANGE_WHY);
            ab_internal_zgemm_cpu(ta, tb, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
            return;
        }
    }

    // The host path: compute op(A)*op(B) on GPU, apply alpha/beta in epilogue.
    // Rows of op(A) and columns of op(B) outside the window are scaled by
    // powers of two on upload, and C back here (see "Exponent range").
    bool ok = false, scaled = false;
    ab_prof_fallback_t fallback = AB_PROF_FB_GPU_FAILED;
    double *Ar_data = NULL, *Ai_data = NULL, *Br_data = NULL, *Bi_data = NULL;
    double *Cr_data = NULL, *Ci_data = NULL;
    ab_line_stats_t sa = {0}, sb = {0};

    // Allocate split-complex GPU matrices: op(A) is M×K, op(B) is K×N
    ABMatrix mAr = ab_matrix_create(M, K);
    ABMatrix mAi = ab_matrix_create(M, K);
    ABMatrix mBr = ab_matrix_create(K, N);
    ABMatrix mBi = ab_matrix_create(K, N);
    ABMatrix mCr = ab_matrix_create(M, N);
    ABMatrix mCi = ab_matrix_create(M, N);
    if (!mAr || !mAi || !mBr || !mBi || !mCr || !mCi) goto cleanup;
    if (!ab_line_stats_init(&sa, M, true) || !ab_line_stats_init(&sb, N, true)) goto cleanup;

    // Upload op(A) (column-major to row-major, transposed/conjugated as asked)
    Ar_data = malloc((size_t)M * K * sizeof(double));
    Ai_data = malloc((size_t)M * K * sizeof(double));
    if (!Ar_data || !Ai_data) goto cleanup;
    prof_stage(p, AB_PROF_STAGE_ALLOC);
    pack_op_complex_stats(ta, A, ldA, M, K, Ar_data, Ai_data, true, &sa);
    const int* ra = ab_line_scales(&sa);
    prof_stage(p, AB_PROF_STAGE_PACK);
    if (ab_internal_upload_scaled(mAr, Ar_data, ra, NULL, 0, NULL) != AB_OK ||
        ab_internal_upload_scaled(mAi, Ai_data, ra, NULL, 0, NULL) != AB_OK)
        goto cleanup;
    prof_stage(p, AB_PROF_STAGE_UPLOAD);
    free(Ar_data); free(Ai_data);
    Ar_data = Ai_data = NULL;

    // Upload op(B)
    Br_data = malloc((size_t)K * N * sizeof(double));
    Bi_data = malloc((size_t)K * N * sizeof(double));
    if (!Br_data || !Bi_data) goto cleanup;
    prof_stage(p, AB_PROF_STAGE_ALLOC);
    pack_op_complex_stats(tb, B, ldB, K, N, Br_data, Bi_data, false, &sb);
    const int* cb = ab_line_scales(&sb);
    prof_stage(p, AB_PROF_STAGE_PACK);
    if (ab_internal_upload_scaled(mBr, Br_data, NULL, cb, 0, NULL) != AB_OK ||
        ab_internal_upload_scaled(mBi, Bi_data, NULL, cb, 0, NULL) != AB_OK)
        goto cleanup;
    prof_stage(p, AB_PROF_STAGE_UPLOAD);
    free(Br_data); free(Bi_data);
    Br_data = Bi_data = NULL;
    prof_stage(p, AB_PROF_STAGE_ALLOC);
    scaled = ra || cb;

    const ab_op_view_t va = {(const double*)A, ldA, M, K, ta != CblasNoTrans, true};
    const ab_op_view_t vb = {(const double*)B, ldB, K, N, tb != CblasNoTrans, true};
    if (!ab_range_guard(K, &sa, &sb, 0, &va, &vb)) {
        fallback = AB_PROF_FB_RANGE;
        goto cleanup;
    }
    prof_stage(p, AB_PROF_STAGE_PACK);

    // Compute C = op(A) * op(B), by 4M where some row of op(A) and column of
    // op(B) are imbalanced enough to cost Gauss-3M's imaginary part its
    // accuracy (ab_internal.h).
    if (ab_internal_zgemm(AB_NO_TRANS, AB_NO_TRANS, mAr, mAi, mBr, mBi, mCr, mCi,
                          ab_lines_method(&sa, &sb)) != AB_OK)
        goto cleanup;
    prof_kernel(p);

    // Download result and apply alpha/beta epilogue
    Cr_data = malloc((size_t)M * N * sizeof(double));
    Ci_data = malloc((size_t)M * N * sizeof(double));
    if (!Cr_data || !Ci_data) goto cleanup;
    prof_stage(p, AB_PROF_STAGE_ALLOC);
    if (ab_matrix_download(mCr, Cr_data, true) != AB_OK ||
        ab_matrix_download(mCi, Ci_data, true) != AB_OK) goto cleanup;
    prof_stage(p, AB_PROF_STAGE_DOWNLOAD);

    // Convert row-major back to column-major with alpha/beta:
    // C = alpha * (A*B) + beta * C_old (zgemm_epilogue, scaled back first).
    zgemm_epilogue(M, N, alpha, beta, Cr_data, Ci_data, C, ldC, scaled ? sa.e : NULL,
                   scaled ? sb.e : NULL);
    ok = true;
    prof_stage(p, AB_PROF_STAGE_EPILOGUE);

cleanup:
    free(Ar_data); free(Ai_data); free(Br_data); free(Bi_data);
    free(Cr_data); free(Ci_data);
    ab_line_stats_free(&sa); ab_line_stats_free(&sb);
    ab_matrix_destroy(mAr); ab_matrix_destroy(mAi);
    ab_matrix_destroy(mBr); ab_matrix_destroy(mBi);
    ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);
    prof_stage(p, AB_PROF_STAGE_ALLOC);

    // Any failure happened before C was written: redo the whole call on CPU.
    if (!ok) {
        note_gpu_fallback(p, "zgemm", M, N, K, fallback,
                          fallback == AB_PROF_FB_RANGE ? AB_RANGE_WHY : "GPU step failed");
        ab_internal_zgemm_cpu(ta, tb, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
        return;
    }
    _ab_last_path = "gpu";
}

// =============================================================================
// BLAS-compatible DGEMM: C = alpha * op(A) * op(B) + beta * C
// Handles transA, transB, alpha, beta, leading dimensions.
// GPU path supports all transpose combinations via upload-time reordering.
// =============================================================================
static void dgemm_blas(
    CallProf *p,
    char transA, char transB,
    int M, int N, int K,
    double alpha,
    const double* A, int ldA,
    const double* B, int ldB,
    double beta,
    double* C, int ldC
) {
    enum CBLAS_TRANSPOSE ta = ab_blas_trans(transA, false);
    enum CBLAS_TRANSPOSE tb = ab_blas_trans(transB, false);
    t_last_stage.staged = false;

    if (!ab_internal_blas_routes_gpu('d', transA, transB, M, N, K, ldA, ldB, ldC, alpha != 0.0)) {
        ab_internal_dgemm_cpu(ta, tb, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
        return;
    }

    ABStatus ready = gpu_ready(M, N, K);
    if (ready != AB_OK) {
        note_gpu_fallback(p, "dgemm", M, N, K, fallback_reason(ready), ab_status_string(ready));
        ab_internal_dgemm_cpu(ta, tb, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
        return;
    }
    prof_stage(p, AB_PROF_STAGE_INIT);
    if (staged_path(p)) {
        unsigned why = dgemm_staged(p, ta, tb, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
        if (!why) {
            _ab_last_path = "gpu";
            return;
        }
        // As for ZGEMM.
        if (why & 1u << AB_PROF_STAGED_RANGE) {
            note_gpu_fallback(p, "dgemm", M, N, K, AB_PROF_FB_RANGE, AB_RANGE_WHY);
            ab_internal_dgemm_cpu(ta, tb, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
            return;
        }
    }

    // The host path: supports arbitrary alpha/beta and transpose via scaled kernel.
    // Rows of op(A) and columns of op(B) outside the window are scaled by
    // powers of two on upload, alpha and beta outside theirs give their
    // exponents to C's scale, and C is scaled back here (see "Exponent range").
    bool ok = false;
    ab_prof_fallback_t fallback = AB_PROF_FB_GPU_FAILED;
    double *A_row = NULL, *B_row = NULL, *C_row = NULL;
    ab_line_stats_t sa = {0}, sb = {0};
    ABMatrix mA = ab_matrix_create(M, K);
    ABMatrix mB = ab_matrix_create(K, N);
    ABMatrix mC = ab_matrix_create(M, N);
    if (!mA || !mB || !mC) goto cleanup;
    if (!ab_line_stats_init(&sa, M, false) || !ab_line_stats_init(&sb, N, false)) goto cleanup;

    // Upload op(A) as M×K and op(B) as K×N, row-major
    A_row = malloc((size_t)M * K * sizeof(double));
    if (!A_row) goto cleanup;
    prof_stage(p, AB_PROF_STAGE_ALLOC);
    pack_op_real_stats(ta, A, ldA, M, K, A_row, true, &sa);
    const int* ra = ab_line_scales(&sa);
    prof_stage(p, AB_PROF_STAGE_PACK);
    if (ab_internal_upload_scaled(mA, A_row, ra, NULL, 0, NULL) != AB_OK) goto cleanup;
    prof_stage(p, AB_PROF_STAGE_UPLOAD);
    free(A_row);
    A_row = NULL;

    B_row = malloc((size_t)K * N * sizeof(double));
    if (!B_row) goto cleanup;
    prof_stage(p, AB_PROF_STAGE_ALLOC);
    pack_op_real_stats(tb, B, ldB, K, N, B_row, false, &sb);
    const int* cb = ab_line_scales(&sb);
    prof_stage(p, AB_PROF_STAGE_PACK);
    if (ab_internal_upload_scaled(mB, B_row, NULL, cb, 0, NULL) != AB_OK) goto cleanup;
    prof_stage(p, AB_PROF_STAGE_UPLOAD);
    free(B_row);
    B_row = NULL;

    // The GPU multiplies by alpha * 2^-ea and beta * 2^-eb; C_ij's scale is
    // 2^(ra[i] + cb[j] + ea), so C goes up as C * 2^-(ra[i] + cb[j] + ea - eb).
    int ea = ab_scalar_exp(alpha, AB_SCALAR_WINDOW), eb = ab_scalar_exp(beta, AB_SCALAR_WINDOW);
    double alpha_s = ab_scale2(alpha, -ea), beta_s = ab_scale2(beta, -eb);
    if (!isfinite(alpha) || !isfinite(beta) ||
        !ab_range_guard(K, &sa, &sb, ilogb(alpha_s),
                        &(ab_op_view_t){A, ldA, M, K, ta != CblasNoTrans, false},
                        &(ab_op_view_t){B, ldB, K, N, tb != CblasNoTrans, false})) {
        fallback = AB_PROF_FB_RANGE;
        goto cleanup;
    }
    prof_stage(p, AB_PROF_STAGE_PACK);

    // Upload existing C only when beta != 0 (C = alpha*A*B + beta*C); with
    // beta == 0 C is never read, per BLAS (the kernel skips beta*C).
    C_row = malloc((size_t)M * N * sizeof(double));
    if (!C_row) goto cleanup;
    prof_stage(p, AB_PROF_STAGE_ALLOC);
    if (beta != 0.0) {
        ab_range_t cr;
        pack_c(C, ldC, M, N, C_row, &cr);
        prof_stage(p, AB_PROF_STAGE_PACK);
        // Scaled, C is measured again as uploaded.
        bool scaled_c = ra || cb || ea != eb;
        if (ab_internal_upload_scaled(mC, C_row, ra, cb, ea - eb, scaled_c ? &cr : NULL) != AB_OK)
            goto cleanup;
        prof_stage(p, AB_PROF_STAGE_UPLOAD);
        if (!ab_range_guard_c(&sa, &sb, cr, ilogb(beta_s), C, ldC, ea - eb)) {
            fallback = AB_PROF_FB_RANGE;
            goto cleanup;
        }
        prof_stage(p, AB_PROF_STAGE_PACK);
    }

    // Compute C = alpha*A*B + beta*C on GPU
    if (ab_dgemm_scaled(alpha_s, mA, mB, beta_s, mC) != AB_OK) goto cleanup;
    prof_kernel(p);

    // Download result (row-major to column-major); bit for bit
    // (ab_internal.h), each C_ij as the download made it FP64, times
    // 2^(ra[i] + cb[j] + ea) in one rounding (ab_scale2).
    if (ab_matrix_download(mC, C_row, true) != AB_OK) goto cleanup;
    prof_stage(p, AB_PROF_STAGE_DOWNLOAD);
    if (ra || cb || ea) {
        for (size_t j = 0; j < (size_t)N; j++) {
            const int ej = sb.e[j] + ea;
            for (size_t i = 0; i < (size_t)M; i++)
                C[j * ldC + i] = ab_scale2(C_row[i * N + j], sa.e[i] + ej);
        }
    } else {
        for (size_t j = 0; j < (size_t)N; j++)
            for (size_t i = 0; i < (size_t)M; i++)
                C[j * ldC + i] = C_row[i * N + j];
    }
    ok = true;
    prof_stage(p, AB_PROF_STAGE_EPILOGUE);

cleanup:
    free(A_row); free(B_row); free(C_row);
    ab_line_stats_free(&sa); ab_line_stats_free(&sb);
    ab_matrix_destroy(mA);
    ab_matrix_destroy(mB);
    ab_matrix_destroy(mC);
    prof_stage(p, AB_PROF_STAGE_ALLOC);

    // Any failure happened before C was written: redo the whole call on CPU.
    if (!ok) {
        note_gpu_fallback(p, "dgemm", M, N, K, fallback,
                          fallback == AB_PROF_FB_RANGE ? AB_RANGE_WHY : "GPU step failed");
        ab_internal_dgemm_cpu(ta, tb, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
        return;
    }
    _ab_last_path = "gpu";
}

// =============================================================================
// BLAS-ABI entry points
// =============================================================================
// Every BLAS-ABI call ends here: dgemm_, zgemm_ and ab_*gemm_ forward to
// these (fortran_bridge.c). Each call is timed and recorded for AB_PROFILE
// and AB_REPORT, and traced for AB_TRACE.

void ab_dgemm_blas(char transA, char transB, int M, int N, int K,
                   double alpha, const double* A, int ldA,
                   const double* B, int ldB,
                   double beta, double* C, int ldC) {
    CallProf prof, *p = prof_begin(&prof);
    if (p && p->trace)
        trace_begin(p, "dgemm", transA, transB, M, N, K, (const double[2]){alpha, 0.0},
                    A, ldA, B, ldB, (const double[2]){beta, 0.0}, C, ldC);
    dgemm_blas(p, transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
    prof_end(p, "dgemm", M, N, K);
}

void ab_zgemm_blas(char transA, char transB, int M, int N, int K,
                   double complex alpha, const double complex* A, int ldA,
                   const double complex* B, int ldB,
                   double complex beta, double complex* C, int ldC) {
    CallProf prof, *p = prof_begin(&prof);
    if (p && p->trace)
        trace_begin(p, "zgemm", transA, transB, M, N, K,
                    (const double[2]){creal(alpha), cimag(alpha)}, A, ldA, B, ldB,
                    (const double[2]){creal(beta), cimag(beta)}, C, ldC);
    zgemm_blas(p, transA, transB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
    prof_end(p, "zgemm", M, N, K);
}

// Fortran-callable wrappers (ab_zgemm_, ab_dgemm_) are in fortran_bridge.c
