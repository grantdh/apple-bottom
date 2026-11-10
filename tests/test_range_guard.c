// =============================================================================
// test_range_guard.c — stage 2 of the exponent-range guards against the plain
// loops they replace
// =============================================================================
//
// Host only, no GPU. ab_range_guard's stage 2 and ab_range_guard_c's stage 2
// (src/blas_wrapper.c) find their probe elements by one-pass and sweeping
// argmaxes, work out the need's terms of each line once, and pass rows by a
// bound before looking at their pairs. This file keeps the plain loops they
// replaced as the oracle (those of 9e88907, fix/guard-subnormal-stats:
// 44873b6's loops on ed9a27b's statistics, whose minima count a value below
// 2^-1042; copied verbatim below, with the static helpers they use) and
// requires the library's decision to equal the oracle's on every case: random
// and crafted data, real and complex, every transpose combination, leading
// dimensions above the minimum (with poisoned padding), entry-floor and
// product-floor cases, ties for the largest element, lines scaled out of the
// window, values below 2^-1042 (lines made of them too), zero lines and
// parts, and calls the guard refuses. Each suite also requires both outcomes
// to occur in stage 2 often enough that the comparison means something, and
// the crafted cases their known outcome. A last test holds a ZGEMM's
// epilogue bounds (step 5) to their edges and to the claim behind them.
// =============================================================================

#include "ab_internal.h"
#include "test_blas_check.h"
#include "test_registry.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int tests_passed = 0;
static int tests_failed = 0;
static char first_failure[512];

#define PASS() do { printf("✓ PASS\n"); tests_passed++; } while (0)
#define FAIL(msg) do { printf("✗ FAIL (%s)\n", msg); tests_failed++; } while (0)

static bool expect(bool cond, const char *fmt, ...) {
    if (!cond && !first_failure[0]) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(first_failure, sizeof first_failure, fmt, ap);
        va_end(ap);
    }
    return cond;
}

static void begin(void) { first_failure[0] = '\0'; }
static void finish(void) { if (first_failure[0]) FAIL(first_failure); else PASS(); }

// =============================================================================
// The oracle: 9e88907's stage 2 (src/blas_wrapper.c), verbatim but for names
// =============================================================================

static inline int o_line_min_exp(const ab_line_stats_t* s, int i) {
    return ab_hiword_min_ilogb(s->min[i]) - s->e[i];
}

static inline int o_line_max_exp(const ab_line_stats_t* s, int i) {
    return ab_hiword_ilogb(s->max[i]) - s->e[i];
}

static inline size_t o_op_at(const ab_op_view_t* v, int i, int k) {
    return v->trans ? (size_t)i * v->ld + k : (size_t)k * v->ld + i;
}

static inline double o_op_mag(const ab_op_view_t* v, int i, int k) {
    const size_t x = o_op_at(v, i, k);
    if (!v->complex_pairs) return fabs(v->x[x]);
    return fmax(fabs(v->x[2 * x]), fabs(v->x[2 * x + 1]));
}

static inline void o_op_parts(const ab_op_view_t* v, int i, int k, int e, double* re,
                              double* im) {
    const size_t x = o_op_at(v, i, k);
    *re = ab_scale2(fabs(v->complex_pairs ? v->x[2 * x] : v->x[x]), -e);
    *im = v->complex_pairs ? ab_scale2(fabs(v->x[2 * x + 1]), -e) : 0.0;
}

static int o_line_argmax(const ab_op_view_t* v, int l, bool by_row) {
    const int len = by_row ? v->cols : v->rows;
    double best = -1.0;
    int at = 0;
    for (int k = 0; k < len; k++) {
        double m = by_row ? o_op_mag(v, l, k) : o_op_mag(v, k, l);
        if (m > best) { best = m; at = k; }
    }
    return at;
}

static inline int o_imax2(int a, int b) { return a > b ? a : b; }
static inline int o_imin2(int a, int b) { return a < b ? a : b; }

static bool o_lines_extremes(const ab_line_stats_t* s, int* hi, int* lo) {
    *hi = INT_MIN;
    *lo = INT_MAX;
    for (int i = 0; i < s->n; i++) {
        if (s->max[i] >= AB_HIWORD_NONFINITE) return false;
        if (s->max[i] == 0) continue;
        *hi = o_imax2(*hi, ab_hiword_ilogb(s->max[i]) - s->e[i]);
        *lo = o_imin2(*lo, o_line_min_exp(s, i));
    }
    return true;
}

static inline int o_stage1_floor(int alpha_exp) { return AB_PRODUCT_MIN_EXP - o_imin2(alpha_exp, 0); }

static bool o_part_has_products(const ab_line_stats_t* sa, const ab_line_stats_t* sb, int i, int j,
                                bool complex_pairs, bool re) {
    if (!complex_pairs) return re;
    const bool ar = sa->rmax[i], ai = sa->imax[i], br = sb->rmax[j], bi = sb->imax[j];
    return re ? (ar && br) || (ai && bi) : (ar && bi) || (ai && br);
}

static inline void o_k_part_scales(double ar, double ai, double br, double bi, double* re,
                                   double* im) {
    *re = ar * br + ai * bi;
    *im = ar * bi + ai * br;
}

static bool oracle_range_guard(int K, const ab_line_stats_t* sa, const ab_line_stats_t* sb,
                               int alpha_exp, const ab_op_view_t* a, const ab_op_view_t* b) {
    ab_guard_t g = ab_range_guard_stage1(K, sa, sb, alpha_exp, a->complex_pairs);
    if (g != AB_GUARD_STAGE2) return g == AB_GUARD_CERTIFIED;

    const int M = sa->n, N = sb->n, lo = o_stage1_floor(alpha_exp);
    const bool z = a->complex_pairs;
    int amax, amin, bmax, bmin;
    o_lines_extremes(sa, &amax, &amin);
    o_lines_extremes(sb, &bmax, &bmin);
    const double floor_ij = fmax(ldexp((double)K, AB_PRODUCT_MIN_EXP),
                                 ldexp(1.0, AB_PRODUCT_MIN_EXP - alpha_exp));
    const double entry_k = (double)K * (z ? 2 : 1);
    int *rows = malloc((size_t)M * sizeof *rows), *ka = malloc((size_t)M * sizeof *ka);
    int *cols = malloc((size_t)N * sizeof *cols), *kb = malloc((size_t)N * sizeof *kb);
    double *ap = malloc((size_t)M * 2 * sizeof *ap), *bp = malloc((size_t)N * 2 * sizeof *bp);
    bool ok = rows && ka && cols && kb && ap && bp;
    const bool entry_a = amin < AB_ENTRY_MIN_EXP, entry_b = bmin < AB_ENTRY_MIN_EXP;
    int nr = 0, nc = 0;
    for (int i = 0; ok && i < M; i++)
        if (sa->max[i] != 0 && (entry_b || o_line_min_exp(sa, i) + bmin < lo ||
                                o_line_min_exp(sa, i) < AB_ENTRY_MIN_EXP)) rows[nr++] = i;
    for (int j = 0; ok && j < N; j++)
        if (sb->max[j] != 0 && (entry_a || o_line_min_exp(sb, j) + amin < lo ||
                                o_line_min_exp(sb, j) < AB_ENTRY_MIN_EXP)) cols[nc++] = j;
    for (int r = 0; r < nr; r++) {
        ka[r] = o_line_argmax(a, rows[r], true);
        o_op_parts(a, rows[r], ka[r], sa->e[rows[r]], &ap[2 * r], &ap[2 * r + 1]);
    }
    for (int c = 0; c < nc; c++) {
        kb[c] = o_line_argmax(b, cols[c], false);
        o_op_parts(b, kb[c], cols[c], sb->e[cols[c]], &bp[2 * c], &bp[2 * c + 1]);
    }
    for (int r = 0; ok && r < nr; r++) {
        const int i = rows[r];
        for (int c = 0; ok && c < nc; c++) {
            const int j = cols[c];
            double need = o_line_min_exp(sa, i) + o_line_min_exp(sb, j) < lo ? floor_ij : 0.0;
            if (o_line_min_exp(sa, i) < AB_ENTRY_MIN_EXP)
                need = fmax(need, ldexp(entry_k, AB_ENTRY_MIN_EXP + o_line_max_exp(sb, j) + 1));
            if (o_line_min_exp(sb, j) < AB_ENTRY_MIN_EXP)
                need = fmax(need, ldexp(entry_k, AB_ENTRY_MIN_EXP + o_line_max_exp(sa, i) + 1));
            if (need == 0.0) continue;
            const double nre = o_part_has_products(sa, sb, i, j, z, true) ? need : 0.0;
            const double nim = o_part_has_products(sa, sb, i, j, z, false) ? need : 0.0;
            double ar, ai, br, bi, re, im, re2, im2;
            o_op_parts(b, ka[r], j, sb->e[j], &br, &bi);
            o_k_part_scales(ap[2 * r], ap[2 * r + 1], br, bi, &re, &im);
            if (re < nre || im < nim) {
                o_op_parts(a, i, kb[c], sa->e[i], &ar, &ai);
                o_k_part_scales(ar, ai, bp[2 * c], bp[2 * c + 1], &re2, &im2);
                re = fmax(re, re2);
                im = fmax(im, im2);
            }
            ok = re >= nre && im >= nim;
        }
    }
    free(rows); free(ka); free(cols); free(kb); free(ap); free(bp);
    return ok;
}

static inline bool o_c_floors_hold(int c_exp, int beta_exp) {
    return c_exp >= AB_ENTRY_MIN_EXP && c_exp + beta_exp >= AB_PRODUCT_MIN_EXP;
}

static bool oracle_range_guard_c(const ab_line_stats_t* sa, const ab_line_stats_t* sb,
                                 ab_range_t r, double alpha, double beta, const double* C,
                                 int ldC, int shift, const ab_op_view_t* a,
                                 const ab_op_view_t* b) {
    ab_guard_t g = ab_range_guard_c_stage1(r, beta);
    if (g != AB_GUARD_STAGE2) return g == AB_GUARD_CERTIFIED;
    const int M = sa->n, N = sb->n, beta_exp = ilogb(beta);
    const double need = ldexp(fmax(1.0, fabs(beta)), AB_PRODUCT_MIN_EXP), al = fabs(alpha);
    int *ka = malloc((size_t)M * sizeof *ka), *kb = malloc((size_t)N * sizeof *kb);
    double *av = malloc((size_t)M * sizeof *av), *bv = malloc((size_t)N * sizeof *bv);
    bool ok = ka && kb && av && bv;
    for (int i = 0; ok && i < M; i++) ka[i] = -1;
    for (int j = 0; ok && j < N; j++) kb[j] = -1;
    for (int j = 0; ok && j < N; j++)
        for (int i = 0; ok && i < M; i++) {
            const double c = C[(size_t)j * ldC + i];
            const int e = sa->e[i] + sb->e[j] + shift;
            if (c == 0.0 || o_c_floors_hold(ab_hiword_ilogb(ab_abs_hiword(c)) - e, beta_exp))
                continue;
            if (sa->max[i] == 0 || sb->max[j] == 0) { ok = false; break; }
            if (ka[i] < 0) {
                ka[i] = o_line_argmax(a, i, true);
                av[i] = ab_scale2(o_op_mag(a, i, ka[i]), -sa->e[i]);
            }
            ok = al * av[i] * ab_scale2(o_op_mag(b, ka[i], j), -sb->e[j]) >= need;
            if (ok) continue;
            if (kb[j] < 0) {
                kb[j] = o_line_argmax(b, j, false);
                bv[j] = ab_scale2(o_op_mag(b, kb[j], j), -sb->e[j]);
            }
            ok = al * ab_scale2(o_op_mag(a, i, kb[j]), -sa->e[i]) * bv[j] >= need;
        }
    free(ka); free(kb); free(av); free(bv);
    return ok;
}

// =============================================================================
// Data
// =============================================================================

static uint64_t rng_state;
static uint64_t rnd64(void) {
    const uint64_t s = rng_state;
    rng_state += 0x9E3779B97F4A7C15ULL;
    return splitmix(s);
}
static int imin(int a, int b) { return a < b ? a : b; }
static double rnd_unit(void) { return (double)(rnd64() >> 11) * 0x1p-53; }
static int rnd_int(int lo, int hi) { return lo + (int)(rnd64() % (uint64_t)(hi - lo + 1)); }
static bool rnd_bool(double p) { return rnd_unit() < p; }
static double rnd_sign(void) { return rnd_bool(0.5) ? -1.0 : 1.0; }
// +-[1, 2) 2^e (subnormal or zero far down, as ldexp rounds)
static double rnd_val(int e) { return rnd_sign() * ldexp(1.0 + rnd_unit(), e); }

// A GEMM's operands as the BLAS ABI passes them, logically op(A) (M x K) and
// op(B) (K x N), elements of each part (re, im; im 0 for real data).
typedef struct {
    bool z;
    char ta, tb;            // 'N', 'T' or 'C' (the view's trans: != 'N')
    int M, N, K;
    Layout la, lb;          // A's and B's storage (test_blas_check.h)
    double *A, *B;          // storage, (re, im) pairs if z, padding poisoned
    int alpha_exp;          // for a DGEMM; 0 for a ZGEMM
} Gemm;

#define POISON 0x1p900      // far above every element: an argmax that reads padding shows

static void gemm_alloc(Gemm *g, bool z, char ta, char tb, int M, int N, int K, int pad_a,
                       int pad_b) {
    *g = (Gemm){.z = z, .ta = ta, .tb = tb, .M = M, .N = N, .K = K,
                .la = stored(ta, M, K, pad_a), .lb = stored(tb, K, N, pad_b)};
    const size_t na = nelem(g->la) * (z ? 2 : 1), nb = nelem(g->lb) * (z ? 2 : 1);
    g->A = malloc(na * sizeof(double));
    g->B = malloc(nb * sizeof(double));
    for (size_t x = 0; x < na; x++) g->A[x] = POISON;
    for (size_t x = 0; x < nb; x++) g->B[x] = POISON;
}

static void gemm_free(Gemm *g) { free(g->A); free(g->B); }

static void set_a(Gemm *g, int i, int k, double re, double im) {
    const size_t x = op_index(g->ta, g->la, i, k);
    if (g->z) { g->A[2 * x] = re; g->A[2 * x + 1] = im; } else g->A[x] = re;
}
static void set_b(Gemm *g, int k, int j, double re, double im) {
    const size_t x = op_index(g->tb, g->lb, k, j);
    if (g->z) { g->B[2 * x] = re; g->B[2 * x + 1] = im; } else g->B[x] = re;
}
static ab_op_view_t view_a(const Gemm *g) {
    return (ab_op_view_t){g->A, g->la.ld, g->M, g->K, !is_notrans(g->ta), g->z};
}
static ab_op_view_t view_b(const Gemm *g) {
    return (ab_op_view_t){g->B, g->lb.ld, g->K, g->N, !is_notrans(g->tb), g->z};
}

// The knobs of a random case.
typedef struct {
    int span;               // a wide line's elements: exponents down to span below its base
    int narrow;             // other lines': down to this
    double p_wide;          // a line is wide
    double p_shift;         // a line's base lies outside the scale window
    double p_zero;          // an element is 0
    double p_zero_part;     // a complex element's part is 0
    double p_zero_line;
    double p_real_line;     // a complex line has real parts only
    double p_imag_line;     // ... imaginary parts only
    double p_tie;           // a line repeats its largest magnitude at another k
    double p_tiny;          // an element lies below 2^-1042 (a high word of 0)
} Knobs;

static Knobs rnd_knobs(void) {
    static const int spans[] = {10, 40, 70, 85, 100, 115, 140, 200};
    static const double ps[] = {0.0, 0.02, 0.1, 0.3};
    Knobs k = {
        .span = spans[rnd_int(0, 7)], .narrow = rnd_int(0, 12),
        .p_wide = (const double[]){0.03, 0.15, 0.5, 1.0}[rnd_int(0, 3)],
        .p_shift = ps[rnd_int(0, 3)], .p_zero = ps[rnd_int(0, 3)],
        .p_zero_part = ps[rnd_int(0, 3)], .p_zero_line = ps[rnd_int(0, 2)] / 2,
        .p_real_line = ps[rnd_int(0, 3)], .p_imag_line = ps[rnd_int(0, 2)],
        .p_tie = ps[rnd_int(0, 3)], .p_tiny = ps[rnd_int(0, 2)] / 4,
    };
    return k;
}

// Knobs for QE-like operands: no zeros or values below 2^-1042, lines near
// their base, a share of them (p_wide) spread down to span below it.
static Knobs benign_knobs(double p_wide, int span) {
    return (Knobs){.span = span, .narrow = rnd_int(0, 8), .p_wide = p_wide, .p_shift = 0.02};
}

// One line of n elements (re, im) into x, from knobs.
static void rnd_line(const Knobs *kn, bool z, int n, double *re, double *im) {
    const int base = rnd_bool(kn->p_shift) ? rnd_sign() * rnd_int(21, 400) : rnd_int(-12, 12);
    const bool wide = rnd_bool(kn->p_wide), zero = rnd_bool(kn->p_zero_line);
    const bool real_only = z && rnd_bool(kn->p_real_line);
    const bool imag_only = z && !real_only && rnd_bool(kn->p_imag_line);
    const int span = wide ? kn->span : kn->narrow;
    for (int k = 0; k < n; k++) {
        re[k] = im[k] = 0.0;
        if (zero || rnd_bool(kn->p_zero)) continue;
        // most elements near the line's base, the rest spread down to span
        int e = base - (rnd_bool(0.6) ? rnd_int(0, imin(span, 6)) : rnd_int(0, span));
        if (rnd_bool(kn->p_tiny)) e = rnd_int(-1100, -1043);
        re[k] = rnd_val(e);
        if (z) {
            int e2 = base - (rnd_bool(0.6) ? rnd_int(0, imin(span, 6)) : rnd_int(0, span));
            im[k] = rnd_val(e2);
            if (rnd_bool(kn->p_zero_part)) { if (rnd_bool(0.5)) re[k] = 0.0; else im[k] = 0.0; }
        }
        if (real_only) im[k] = 0.0;
        if (imag_only) re[k] = 0.0;
    }
    if (!zero && n > 1 && rnd_bool(kn->p_tie)) {
        // the largest part's magnitude again, at another k, in either part
        int at = 0;
        double best = -1.0;
        for (int k = 0; k < n; k++) {
            double m = fmax(fabs(re[k]), fabs(im[k]));
            if (m > best) { best = m; at = k; }
        }
        int k2 = rnd_int(0, n - 1);
        if (k2 == at) k2 = (at + 1) % n;
        const double small = best * 0x1p-30 * rnd_unit();
        if (!z || real_only || (!imag_only && rnd_bool(0.5))) {
            re[k2] = rnd_sign() * best;
            im[k2] = z && !real_only ? small : 0.0;
        } else {
            im[k2] = rnd_sign() * best;
            re[k2] = imag_only ? 0.0 : small;
        }
    }
}

// Logical operands: op(A) (M x K) and op(B) (K x N), row-major, parts apart.
typedef struct {
    bool z;
    int M, N, K;
    double *ar, *ai, *br, *bi;      // ar[i K + k], br[k N + j]
} Logical;

static void logical_alloc(Logical *L, bool z, int M, int N, int K) {
    *L = (Logical){.z = z, .M = M, .N = N, .K = K};
    L->ar = calloc((size_t)M * K, sizeof(double));
    L->ai = calloc((size_t)M * K, sizeof(double));
    L->br = calloc((size_t)K * N, sizeof(double));
    L->bi = calloc((size_t)K * N, sizeof(double));
}

static void logical_free(Logical *L) { free(L->ar); free(L->ai); free(L->br); free(L->bi); }

static void logical_rnd(Logical *L, const Knobs *ka, const Knobs *kb) {
    double *re = malloc((size_t)L->K * sizeof *re), *im = malloc((size_t)L->K * sizeof *im);
    for (int i = 0; i < L->M; i++) {
        rnd_line(ka, L->z, L->K, re, im);
        for (int k = 0; k < L->K; k++) {
            L->ar[(size_t)i * L->K + k] = re[k];
            L->ai[(size_t)i * L->K + k] = im[k];
        }
    }
    for (int j = 0; j < L->N; j++) {
        rnd_line(kb, L->z, L->K, re, im);
        for (int k = 0; k < L->K; k++) {
            L->br[(size_t)k * L->N + j] = re[k];
            L->bi[(size_t)k * L->N + j] = im[k];
        }
    }
    free(re);
    free(im);
}

// L laid out as the BLAS ABI would pass it with transposes ta, tb.
static void gemm_from(Gemm *g, const Logical *L, char ta, char tb, int pad_a, int pad_b) {
    gemm_alloc(g, L->z, ta, tb, L->M, L->N, L->K, pad_a, pad_b);
    for (int i = 0; i < L->M; i++)
        for (int k = 0; k < L->K; k++)
            set_a(g, i, k, L->ar[(size_t)i * L->K + k], L->ai[(size_t)i * L->K + k]);
    for (int k = 0; k < L->K; k++)
        for (int j = 0; j < L->N; j++)
            set_b(g, k, j, L->br[(size_t)k * L->N + j], L->bi[(size_t)k * L->N + j]);
}

static bool stats_of(const Gemm *g, ab_line_stats_t *sa, ab_line_stats_t *sb) {
    const ab_op_view_t va = view_a(g), vb = view_b(g);
    if (!ab_line_stats_init(sa, g->M, g->z) || !ab_line_stats_init(sb, g->N, g->z)) return false;
    ab_line_stats_scan(&va, true, sa);
    ab_line_stats_scan(&vb, false, sb);
    ab_line_scales(sa);
    ab_line_scales(sb);
    return true;
}

// Flush to zero (FPCR.FZ) around the guards' calls, for the suite that asks.
static bool g_flush_to_zero;
#if defined(__aarch64__)
static uint64_t fpcr_get(void) { uint64_t v; __asm__ volatile("mrs %0, fpcr" : "=r"(v) : : "memory"); return v; }
static void fpcr_set(uint64_t v) { __asm__ volatile("msr fpcr, %0" : : "r"(v) : "memory"); }
#endif

typedef struct {
    long cases, stage2, accepted, refused, mismatches;
} Tally;

static void tally(Tally *t, bool stage2, bool want, bool got) {
    t->cases++;
    if (stage2) { t->stage2++; if (want) t->accepted++; else t->refused++; }
    if (got != want) t->mismatches++;
}

static void show(const char *what, const Tally *t) {
    printf("\n      %-28s %7ld cases, %6ld in stage 2 (%ld GPU, %ld Accelerate), %ld differ ",
           what, t->cases, t->stage2, t->accepted, t->refused, t->mismatches);
}

typedef bool (*guard_fn)(int, const ab_line_stats_t *, const ab_line_stats_t *, int,
                         const ab_op_view_t *, const ab_op_view_t *);
typedef bool (*guard_c_fn)(const ab_line_stats_t *, const ab_line_stats_t *, ab_range_t, double,
                           double, const double *, int, int, const ab_op_view_t *,
                           const ab_op_view_t *);

// One guard call, under flush to zero if asked (noinline, so that no
// floating-point work moves across the FPCR writes).
__attribute__((noinline)) static bool call_guard(guard_fn f, const Gemm *g, ab_line_stats_t *sa,
                                                 ab_line_stats_t *sb) {
    const ab_op_view_t va = view_a(g), vb = view_b(g);
#if defined(__aarch64__)
    const uint64_t old = fpcr_get();
    if (g_flush_to_zero) fpcr_set(old | (1ULL << 24));
    bool ok = f(g->K, sa, sb, g->alpha_exp, &va, &vb);
    fpcr_set(old);
    return ok;
#else
    return f(g->K, sa, sb, g->alpha_exp, &va, &vb);
#endif
}

// The library against the oracle on g; returns the oracle's decision.
static bool compare(const Gemm *g, Tally *t, const char *what) {
    ab_line_stats_t sa = {0}, sb = {0};
    bool want = false;
    if (!stats_of(g, &sa, &sb)) {
        expect(false, "%s: out of memory", what);
    } else {
        const ab_guard_t s1 = ab_range_guard_stage1(g->K, &sa, &sb, g->alpha_exp, g->z);
        want = call_guard(oracle_range_guard, g, &sa, &sb);
        const bool got = call_guard(ab_range_guard, g, &sa, &sb);
        tally(t, s1 == AB_GUARD_STAGE2, want, got);
        expect(got == want, "%s: %s %c%c %dx%dx%d, ld %d/%d, alpha_exp %d: library %s, oracle %s",
               what, g->z ? "ZGEMM" : "DGEMM", g->ta, g->tb, g->M, g->N, g->K, g->la.ld, g->lb.ld,
               g->alpha_exp, got ? "GPU" : "Accelerate", want ? "GPU" : "Accelerate");
    }
    ab_line_stats_free(&sa);
    ab_line_stats_free(&sb);
    return want;
}

static const char TRANS_Z[] = {'N', 'T', 'C'};
static const char TRANS_D[] = {'N', 'T'};

// Random logical operands, each compared in every transpose combination
// (whose oracle decisions must agree too: the guard reads op(A), op(B)).
// L compared in every transpose combination, with random padding (whose
// oracle decisions must agree too: the guard reads op(A), op(B)).
static void compare_layouts(const Logical *L, int alpha_exp, Tally *t, const char *what) {
    const int nt = L->z ? 3 : 2;
    const char *tr = L->z ? TRANS_Z : TRANS_D;
    int first = -1;
    for (int x = 0; x < nt; x++)
        for (int y = 0; y < nt; y++) {
            Gemm g;
            gemm_from(&g, L, tr[x], tr[y], rnd_bool(0.5) ? rnd_int(1, 3) : 0,
                      rnd_bool(0.5) ? rnd_int(1, 3) : 0);
            g.alpha_exp = alpha_exp;
            bool want = compare(&g, t, what);
            if (first < 0) first = want;
            expect(want == (first == 1), "%s: the oracle's decision changed with the layout", what);
            gemm_free(&g);
        }
}

// Random logical operands, each compared in every layout.
static void random_layouts(bool z, int count, int max_mn, int max_k, Tally *t, const char *what) {
    for (int n = 0; n < count; n++) {
        Logical L;
        logical_alloc(&L, z, rnd_int(1, max_mn), rnd_int(1, max_mn), rnd_int(1, max_k));
        const Knobs ka = rnd_knobs(), kb = rnd_knobs();
        logical_rnd(&L, &ka, &kb);
        compare_layouts(&L, z ? 0 : rnd_int(-20, 20), t, what);
        logical_free(&L);
    }
}

static void need_outcomes(const Tally *t, const char *what, long min_each) {
    expect(t->mismatches == 0, "%s: %ld decisions differ", what, t->mismatches);
    expect(t->accepted >= min_each && t->refused >= min_each,
           "%s: too few stage-2 outcomes to compare (%ld GPU, %ld Accelerate; want %ld each)",
           what, t->accepted, t->refused, min_each);
}

// =============================================================================
// Suites
// =============================================================================

static void test_guard_random_real(void) {
    begin();
    rng_state = 0x5EED0001;
    Tally t = {0};
    random_layouts(false, 3000, 24, 40, &t, "real, small");
    show("real, small", &t);
    need_outcomes(&t, "real, small", 500);
    finish();
}

static void test_guard_random_complex(void) {
    begin();
    rng_state = 0x5EED0002;
    Tally t = {0};
    random_layouts(true, 1500, 24, 40, &t, "complex, small");
    show("complex, small", &t);
    need_outcomes(&t, "complex, small", 500);
    finish();
}

// QE-like shapes, large enough for the sweeps (and the per-line scans below
// their share), first_absmax's blocks and the row groups: projector
// products (tall op(A) rows with tiny parts), their adjoints (long
// contiguous rows), rotations, and lines that are mostly narrow.
static void test_guard_large(void) {
    begin();
    rng_state = 0x5EED0003;
    Tally t = {0};
    static const struct { int M, N, K; } S[] = {
        {1500, 40, 96}, {64, 40, 1500}, {1200, 60, 60}, {900, 33, 200}, {37, 700, 130},
        {2000, 5, 64}, {300, 300, 300},
    };
    for (int rep = 0; rep < 4; rep++)
        for (size_t s = 0; s < TEST_REGISTRY_SIZE(S); s++)
            for (int zi = 0; zi < 2; zi++) {
                const bool z = zi;
                Logical L;
                logical_alloc(&L, z, S[s].M, S[s].N, S[s].K);
                Knobs ka = rnd_knobs(), kb = rnd_knobs();
                if (rep == 0) { ka = benign_knobs(1.0, 170); kb = benign_knobs(0.0, 0); }   // vkb-like
                if (rep == 1) { ka = benign_knobs(0.05, 100); kb = benign_knobs(0.05, 100); }  // few lines
                if (rep == 2) { ka = benign_knobs(0.3, 120); kb = benign_knobs(0.0, 0); }
                logical_rnd(&L, &ka, &kb);
                const int nt = z ? 3 : 2;
                const char *tr = z ? TRANS_Z : TRANS_D;
                for (int x = 0; x < nt; x++)
                    for (int y = 0; y < nt; y++) {
                        Gemm g;
                        gemm_from(&g, &L, tr[x], tr[y], rnd_int(0, 2), rnd_int(0, 2));
                        g.alpha_exp = z ? 0 : rnd_int(-20, 20);
                        compare(&g, &t, "large");
                        gemm_free(&g);
                    }
                logical_free(&L);
            }
    show("large", &t);
    need_outcomes(&t, "large", 20);
    finish();
}

// ---- Crafted cases, each with its known outcome ------------------------------

// A crafted case in every transpose combination: the library and the oracle
// must both give `expected`.
static void crafted(const Logical *L, bool expected, Tally *t, const char *what) {
    const int nt = L->z ? 3 : 2;
    const char *tr = L->z ? TRANS_Z : TRANS_D;
    for (int x = 0; x < nt; x++)
        for (int y = 0; y < nt; y++)
            for (int pad = 0; pad < 2; pad++) {
                Gemm g;
                gemm_from(&g, L, tr[x], tr[y], pad, 2 * pad);
                const bool want = compare(&g, t, what);
                expect(want == expected, "%s (%c%c): the oracle says %s, the construction %s", what,
                       tr[x], tr[y], want ? "GPU" : "Accelerate", expected ? "GPU" : "Accelerate");
                gemm_free(&g);
            }
}

static void la_set(Logical *L, int i, int k, double re, double im) {
    L->ar[(size_t)i * L->K + k] = re;
    L->ai[(size_t)i * L->K + k] = L->z ? im : 0.0;
}
static void lb_set(Logical *L, int k, int j, double re, double im) {
    L->br[(size_t)k * L->N + j] = re;
    L->bi[(size_t)k * L->N + j] = L->z ? im : 0.0;
}

// Row 0 of op(A) holds its largest magnitude 1 twice, at k1 < k2, and a part
// of 2^-100 at k3 (below the entry floor, so every column pairs with it).
// Column 0 of op(B) holds 2 at k3 (its largest), so the second probe gives
// 2^-99 and fails; the first probe, through k1, decides: B(k1, 0) = 1
// passes and 2^-75 fails (the need is about K 2^-76). With first_good the
// call runs on the GPU only if k1 (not k2) is taken. Other rows (nice > 1)
// sit in [0.5, 1) and pair with nothing.
static void tie_in_row(bool z, int K, int k1, int k2, int k3, int nice, bool first_good,
                       Tally *t) {
    Logical L;
    logical_alloc(&L, z, 1 + nice, 1, K);
    for (int k = 0; k < K; k++) {
        la_set(&L, 0, k, 0x1p-10, -0x1p-10);
        lb_set(&L, k, 0, 0.5, 0.5);
        for (int i = 1; i <= nice; i++) la_set(&L, i, k, 0.5 + 0.4 * rnd_unit(), 0.75);
    }
    la_set(&L, 0, k1, 1.0, 0x1p-12);
    la_set(&L, 0, k2, z ? -0x1p-12 : -1.0, -1.0);           // complex: the largest in Im
    la_set(&L, 0, k3, 0x1p-100, 0.0);
    lb_set(&L, k1, 0, first_good ? 1.0 : 0x1p-75, first_good ? 1.0 : 0x1p-75);
    lb_set(&L, k2, 0, first_good ? 0x1p-75 : 1.0, first_good ? 0x1p-75 : 1.0);
    lb_set(&L, k3, 0, 2.0, 2.0);
    char what[96];
    snprintf(what, sizeof what, "tie in a row, %s K=%d k=%d,%d", z ? "complex" : "real", K, k1, k2);
    crafted(&L, first_good, t, what);
    logical_free(&L);
}

// Column 0 of op(B) holds its largest magnitude 2 twice, at k1 < k2; the
// first probe fails (row 0's largest, at k0, meets 2^-75), so the second,
// through k1, decides: A(0, k1) = 2^-10 passes, the 2^-100 fails.
static void tie_in_col(bool z, int K, int k0, int k1, int k2, int nice, bool first_good,
                       Tally *t) {
    Logical L;
    logical_alloc(&L, z, 1, 1 + nice, K);
    for (int k = 0; k < K; k++) {
        la_set(&L, 0, k, 0x1p-10, 0x1p-10);
        lb_set(&L, k, 0, 0.5, -0.5);
        for (int j = 1; j <= nice; j++) lb_set(&L, k, j, 0.5 + 0.4 * rnd_unit(), 0.75);
    }
    la_set(&L, 0, k0, 1.0, 1.0);
    lb_set(&L, k0, 0, 0x1p-75, 0x1p-75);
    for (int j = 1; j <= nice; j++) lb_set(&L, k0, j, 0.5, 0.5);
    la_set(&L, 0, first_good ? k2 : k1, 0x1p-100, 0.0);
    lb_set(&L, k1, 0, 2.0, z ? 0x1p-12 : 0.0);
    lb_set(&L, k2, 0, z ? 0x1p-12 : -2.0, z ? -2.0 : 0.0);  // complex: the largest in Im
    char what[96];
    snprintf(what, sizeof what, "tie in a column, %s K=%d k=%d,%d", z ? "complex" : "real", K, k1, k2);
    crafted(&L, first_good, t, what);
    logical_free(&L);
}

// Every pair fails its first probe (op(A)'s rows peak at k = 0, where op(B)
// is 2^-75) and passes its second (column j peaks at k = 1 + j % (K - 2),
// where op(A) is 2^-10), but for column bad (if >= 0), which peaks where
// the rows hold 2^-100: so every column's probe element is wanted, and with
// many columns, strided ones go to a sweep.
static void second_probes(bool z, int M, int N, int K, int bad, Tally *t) {
    Logical L;
    logical_alloc(&L, z, M, N, K);
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) la_set(&L, i, k, k == 0 ? 1.0 : k == K - 1 ? 0x1p-100 : 0x1p-10,
                                           k == 0 ? -1.0 : k == K - 1 ? 0.0 : 0x1p-11);
    for (int j = 0; j < N; j++)
        for (int k = 0; k < K; k++) {
            const int peak = j == bad ? K - 1 : 1 + j % (K - 2);
            lb_set(&L, k, j, k == 0 ? 0x1p-75 : k == peak ? 2.0 : 0.25, k == 0 ? 0x1p-75 : 0.25);
        }
    char what[96];
    snprintf(what, sizeof what, "second probes, %s %dx%dx%d%s", z ? "complex" : "real", M, N, K,
             bad >= 0 ? ", one fails" : "");
    crafted(&L, bad < 0, t, what);
    logical_free(&L);
}

// Complex parts: row 0 has real parts only (one of 2^-100), column 0
// imaginary parts only, so C_00's real part has no products (no need) and
// its imaginary part is checked: through Re A(0, k) Im B(k, 0).
static void parts_case(bool im_ok, Tally *t) {
    Logical L;
    const int K = 9;
    logical_alloc(&L, true, 1, 1, K);
    for (int k = 0; k < K; k++) {
        la_set(&L, 0, k, k == 4 ? 1.0 : k == 7 ? 0x1p-100 : 0x1p-8, 0.0);
        lb_set(&L, k, 0, 0.0, k == 4 ? (im_ok ? 1.0 : 0x1p-75) : k == 7 ? 2.0 : 0.5);
    }
    crafted(&L, im_ok, t, im_ok ? "parts, imaginary part passes" : "parts, imaginary part fails");
    logical_free(&L);
}

// A row whose bound only just meets its need (exact powers of two): the
// row's pairs are checked one by one, and pass.
static void tight_bound(bool z, Tally *t) {
    Logical L;
    const int K = 2, N = 3;
    logical_alloc(&L, z, 1, N, K);
    la_set(&L, 0, 0, 1.0, 0.0);
    la_set(&L, 0, 1, 0x1p-100, 0.0);
    for (int j = 0; j < N; j++) {
        // need: rows below the entry floor ask K' 2^-77 (K' = K, 2K for complex)
        lb_set(&L, 0, j, j == 0 ? (z ? 0x1p-75 : 0x1p-76) : 1.0, 0.0);
        lb_set(&L, 1, j, j == 0 ? 1.0 : 0.5, 0.0);
    }
    crafted(&L, true, t, z ? "tight bound, complex" : "tight bound, real");
    logical_free(&L);
}

static void test_guard_crafted(void) {
    begin();
    rng_state = 0x5EED0004;
    Tally t = {0};
    static const struct { int K, k1, k2, k3; } T[] = {
        {6, 1, 4, 5}, {6, 2, 5, 0}, {40, 0, 39, 20}, {40, 3, 9, 30}, {40, 17, 18, 2},
        {100, 15, 16, 99}, {100, 16, 80, 0}, {100, 47, 48, 3}, {100, 98, 99, 50}, {33, 31, 32, 0},
    };
    for (size_t x = 0; x < TEST_REGISTRY_SIZE(T); x++)
        for (int zi = 0; zi < 2; zi++)
            for (int good = 0; good < 2; good++)
                for (int nice = 0; nice <= 20; nice += 20) {
                    tie_in_row(zi, T[x].K, T[x].k1, T[x].k2, T[x].k3, nice, good, &t);
                    tie_in_col(zi, T[x].K, T[x].k3, T[x].k1, T[x].k2, nice, good, &t);
                }
    for (int zi = 0; zi < 2; zi++) {
        second_probes(zi, 3, 40, 12, -1, &t);
        second_probes(zi, 3, 40, 12, 17, &t);
        second_probes(zi, 2, 5, 70, -1, &t);
        second_probes(zi, 2, 5, 70, 4, &t);
        tight_bound(zi, &t);
    }
    parts_case(true, &t);
    parts_case(false, &t);
    show("crafted", &t);
    expect(t.mismatches == 0, "crafted: %ld decisions differ", t.mismatches);
    finish();
}

// ---- Values below 2^-1042 --------------------------------------------------------

// Nonzero values below 2^-1042 have a high word of 0, which the statistics
// count as AB_HIWORD_TINY (ed9a27b): smallest exponent -1074, so their lines
// fall below the entry floor, and a line of nothing else is scaled by about
// 2^1072. In QE-like or random data (benign) with one such line (kind 0: a
// row of op(A) made of them; 1: such a column of op(B); 2: a row with every
// third element one; 3: a row whose imaginary parts all are, real data: all
// its elements), the library must decide as the oracle, in every layout,
// with and without flush to zero.
static void tiny_case(bool z, int M, int N, int K, int kind, bool benign, Tally *t) {
    Logical L;
    logical_alloc(&L, z, M, N, K);
    const Knobs ka = benign ? benign_knobs(0.3, 40) : rnd_knobs();
    const Knobs kb = benign ? benign_knobs(0.3, 40) : rnd_knobs();
    logical_rnd(&L, &ka, &kb);
    const int i0 = rnd_int(0, M - 1), j0 = rnd_int(0, N - 1);
    for (int k = 0; k < K; k++) {
        const double t1 = rnd_val(rnd_int(-1074, -1043)), t2 = rnd_val(rnd_int(-1074, -1043));
        if (kind == 0) la_set(&L, i0, k, t1, t2);
        if (kind == 1) lb_set(&L, k, j0, t1, t2);
        if (kind == 2 && k % 3 == 1) la_set(&L, i0, k, t1, t2);
        if (kind == 3) {
            if (z) L.ai[(size_t)i0 * K + k] = t2;
            else la_set(&L, i0, k, t1, 0.0);
        }
    }
    char what[80];
    snprintf(what, sizeof what, "below 2^-1042, kind %d, %s %dx%dx%d", kind,
             z ? "complex" : "real", M, N, K);
    for (int fz = 0; fz < 2; fz++) {
        g_flush_to_zero = fz;
        compare_layouts(&L, z ? 0 : rnd_int(-20, 20), t, what);
    }
    g_flush_to_zero = false;
    logical_free(&L);
}

static void test_guard_tiny_values(void) {
    begin();
    rng_state = 0x5EED0007;
    Tally t = {0};
    static const struct { int M, N, K; } S[] = {{1, 1, 1}, {3, 2, 5}, {12, 9, 40}, {40, 7, 33},
                                               {7, 30, 64}, {300, 20, 50}};
    for (size_t s = 0; s < TEST_REGISTRY_SIZE(S); s++)
        for (int zi = 0; zi < 2; zi++)
            for (int kind = 0; kind < 4; kind++)
                for (int rep = 0; rep < 4; rep++)
                    tiny_case(zi, S[s].M, S[s].N, S[s].K, kind, rep % 2, &t);
    show("below 2^-1042", &t);
    need_outcomes(&t, "below 2^-1042", 20);
    finish();
}

// ---- The DGEMM's C check (ab_range_guard_c) -------------------------------------

__attribute__((noinline)) static bool call_guard_c(guard_c_fn f, const Gemm *g, ab_line_stats_t *sa,
                                                   ab_line_stats_t *sb, ab_range_t r, double alpha,
                                                   double beta, const double *C, int ldc, int shift) {
    const ab_op_view_t va = view_a(g), vb = view_b(g);
#if defined(__aarch64__)
    const uint64_t old = fpcr_get();
    if (g_flush_to_zero) fpcr_set(old | (1ULL << 24));
    bool ok = f(sa, sb, r, alpha, beta, C, ldc, shift, &va, &vb);
    fpcr_set(old);
    return ok;
#else
    return f(sa, sb, r, alpha, beta, C, ldc, shift, &va, &vb);
#endif
}

// Random DGEMMs with beta != 0 and a C whose entries reach below the floors
// once scaled as uploaded (some far below, some zero), in every transpose
// combination.
static void random_c(int count, int max_mn, int max_k, double p_small, Tally *t, const char *what) {
    for (int n = 0; n < count; n++) {
        Logical L;
        logical_alloc(&L, false, rnd_int(1, max_mn), rnd_int(1, max_mn), rnd_int(1, max_k));
        const bool benign = n % 2;      // every other case QE-like, so that large ones pass too
        const Knobs ka = benign ? benign_knobs(0.2, 60) : rnd_knobs();
        const Knobs kb = benign ? benign_knobs(0.2, 60) : rnd_knobs();
        logical_rnd(&L, &ka, &kb);
        const double alpha = rnd_val(benign ? rnd_int(-10, 10) : rnd_int(-45, 20));
        const double beta = rnd_val(benign ? rnd_int(-10, 10) : rnd_int(-30, 30));
        const int shift = rnd_bool(0.5) ? 0 : rnd_int(-25, 25);
        const int M = L.M, N = L.N, ldc = M + rnd_int(0, 2);
        double *C = malloc((size_t)ldc * N * sizeof *C);
        ab_range_t r = AB_RANGE_EMPTY;
        for (int x = 0; x < 2; x++)
            for (int y = 0; y < 2; y++) {
                Gemm g;
                gemm_from(&g, &L, TRANS_D[x], TRANS_D[y], rnd_int(0, 2), rnd_int(0, 2));
                ab_line_stats_t sa = {0}, sb = {0};
                if (!stats_of(&g, &sa, &sb)) { expect(false, "out of memory"); goto next; }
                // C and the range of C as uploaded, C_ij 2^-e, counted as the
                // upload counts it: fixed by the first layout's scales, which
                // every layout shares
                if (x == 0 && y == 0) {
                    for (int j = 0; j < N; j++)
                        for (int i = 0; i < ldc; i++) {
                            const int e = sa.e[imin(i, M - 1)] + sb.e[j] + shift;
                            double c = rnd_val(e + rnd_int(-30, 10));
                            if (rnd_bool(p_small)) c = rnd_val(e - rnd_int(70, 200));
                            if (rnd_bool(0.1)) c = 0.0;
                            C[(size_t)j * ldc + i] = i < M ? c : POISON;
                        }
                    for (int j = 0; j < N; j++)
                        for (int i = 0; i < M; i++) {
                            const double c = C[(size_t)j * ldc + i];
                            const double v = ab_scale2(c, -(sa.e[i] + sb.e[j] + shift));
                            AB_HIWORD_ADD(ab_stat_hiword_scaled(v, c), r.max, r.min);
                        }
                }
                const bool stage2 = ab_range_guard_c_stage1(r, beta) == AB_GUARD_STAGE2;
                const bool want = call_guard_c(oracle_range_guard_c, &g, &sa, &sb, r, alpha, beta, C, ldc, shift);
                const bool got = call_guard_c(ab_range_guard_c, &g, &sa, &sb, r, alpha, beta, C, ldc, shift);
                tally(t, stage2, want, got);
                expect(got == want, "%s: DGEMM %c%c %dx%dx%d, alpha %a beta %a shift %d: library %s, oracle %s",
                       what, g.ta, g.tb, g.M, g.N, g.K, alpha, beta, shift, got ? "GPU" : "Accelerate",
                       want ? "GPU" : "Accelerate");
            next:
                ab_line_stats_free(&sa);
                ab_line_stats_free(&sb);
                gemm_free(&g);
            }
        free(C);
        logical_free(&L);
    }
}

static void test_guard_c_random(void) {
    begin();
    rng_state = 0x5EED0005;
    Tally t = {0}, u = {0};
    random_c(2500, 24, 40, 0.05, &t, "C check, small");
    random_c(40, 900, 120, 0.3, &u, "C check, large");
    show("C check, small", &t);
    show("C check, large", &u);
    need_outcomes(&t, "C check, small", 300);
    need_outcomes(&u, "C check, large", 3);
    finish();
}

// The random suites again with the CPU flushing subnormals to zero (FPCR.FZ):
// the host path still runs the guard then (staged_path), and comparisons
// and maxima flush alike in both implementations.
static void test_guard_flush_to_zero(void) {
    begin();
#if defined(__aarch64__)
    rng_state = 0x5EED0006;
    Tally t = {0}, u = {0}, v = {0};
    g_flush_to_zero = true;
    random_layouts(false, 600, 24, 40, &t, "real, FZ");
    random_layouts(true, 300, 24, 40, &u, "complex, FZ");
    random_c(400, 24, 40, 0.05, &v, "C check, FZ");
    g_flush_to_zero = false;
    show("real, FZ", &t);
    show("complex, FZ", &u);
    show("C check, FZ", &v);
    need_outcomes(&t, "real, FZ", 50);
    need_outcomes(&u, "complex, FZ", 50);
    need_outcomes(&v, "C check, FZ", 30);
#else
    printf("(not aarch64: FPCR not set) ");
#endif
    finish();
}

// Step 5, a ZGEMM's FP64 epilogue (ab_zgemm_product_fits, ab_zgemm_c_fits):
// each bound on both sides of its edge; the GPU epilogue's flag on C beside
// beta = 1 (ab_f64_top: |c| >= 2^1022, Inf and NaN included) where c_fits
// says no; and the bound's claim, that wherever both pass, the epilogue's
// FP64 operations on the largest parts they allow, with the worst signs, stay
// below 2^1023 (1 + 2^-50).

// The P bound with a and b the largest parts of op(A) and op(B).
static bool epi_product(int K, double a, double b, double ar, double ai) {
    return ab_zgemm_product_fits(K, ab_abs_hiword(a), ab_abs_hiword(b), ar, ai);
}

// The epilogue's real part (as zgemm_epilogue forms it) on p = |P|'s parts
// and c = |C|'s, signed so that every term adds: also its imaginary part's
// worst case, with the signs moved.
static double epi_worst(double ar, double ai, double br, double bi, double p, double c) {
    double pr = copysign(p, ar), pi = copysign(p, -ai), cr = copysign(c, br), ci = copysign(c, -bi);
    double r = ar * pr - ai * pi;
    r += br * cr - bi * ci;
    pr = copysign(p, ai);
    pi = copysign(p, ar);
    cr = copysign(c, bi);
    ci = copysign(c, br);
    double i = ar * pi + ai * pr;
    i += br * ci + bi * cr;
    return fmax(fabs(r), fabs(i));
}

static void test_zgemm_epilogue_bounds(void) {
    begin();
    const double two = 2.0, top = 0x1p1023 * (1 + 0x1p-50);
    // The product: E = ilogb a + ilogb b + ilogb K + 5.
    const double a505 = 0.75 * 0x1p506, a506 = 0.75 * 0x1p507, a507 = 0.75 * 0x1p508;
    expect(epi_product(72, a505, a505, two, 0), "E = 1021, alpha 2 refused");
    expect(!epi_product(72, a505, a505, nextafter(two, 3), 0), "E = 1021, alpha 2 + ulp passed");
    expect(epi_product(72, a505, a505, 1, -1), "E = 1021, alpha 1 - i refused");
    expect(!epi_product(72, a505, a505, 1, -(1 + 0x1p-51)), "E = 1021, alpha 1 - (1 + 2^-51)i passed");
    expect(epi_product(127, a505, a505, two, 0), "K = 127 (E = 1021), alpha 2 refused");
    expect(!epi_product(128, a505, a505, two, 0), "K = 128 (E = 1022), alpha 2 passed");
    expect(epi_product(72, a506, a506, 0.5, 0), "E = 1023, alpha 0.5 refused");
    expect(!epi_product(72, a506, a506, 0.5 + 0x1p-53, 0), "E = 1023, alpha 0.5 + ulp passed");
    expect(epi_product(72, a506, a506, 0x1p-1074, 0), "E = 1023, alpha 2^-1074 refused");
    expect(!epi_product(72, a507, a506, 0x1p-1000, 0), "E = 1024 passed (P itself overflows)");
    expect(epi_product(1 << 30, 1.0, 1.0, 0x1p900, 0), "lines at 1, K = 2^30, alpha 2^900 refused");
    expect(epi_product(72, 0.0, a507, 0x1p1000, 0), "op(A) zero, alpha 2^1000 refused");
    expect(!epi_product(72, 0.0, a507, INFINITY, 0), "op(A) zero, alpha Inf passed");
    expect(!epi_product(72, 1.0, 1.0, 1.0, NAN), "alpha 1 + NaN i passed");
    expect(!epi_product(72, 1.0, NAN, 1.0, 0), "a NaN line passed");
    expect(!epi_product(72, INFINITY, 1.0, 1.0, 0), "an Inf line passed");
    // beta C: (|Re beta| + |Im beta|) 2^(ilogb c + 1) <= 2^1022.
    const double below = nextafter(0x1p1022, 0);
    expect(ab_zgemm_c_fits(0.0, -0.0, ab_abs_hiword(NAN)), "beta 0 with NaN in C refused");
    expect(ab_zgemm_c_fits(1.0, 0.0, ab_abs_hiword(below)), "beta 1, |c| below 2^1022 refused");
    expect(!ab_zgemm_c_fits(1.0, 0.0, ab_abs_hiword(0x1p1022)), "beta 1, |c| = 2^1022 passed");
    expect(!ab_zgemm_c_fits(nextafter(1, 2), 0.0, ab_abs_hiword(0x1p1021)),
           "beta 1 + ulp, |c| = 2^1021 passed");
    expect(ab_zgemm_c_fits(0.5, -0.5, ab_abs_hiword(0x1p1021)), "beta 0.5 - 0.5i, |c| = 2^1021 refused");
    expect(ab_zgemm_c_fits(5.0, 0.0, 0), "beta 5, C zero refused");
    expect(!ab_zgemm_c_fits(INFINITY, 0.0, 0), "beta Inf, C zero passed");
    expect(!ab_zgemm_c_fits(NAN, 0.0, ab_abs_hiword(1.0)), "beta NaN passed");
    expect(!ab_zgemm_c_fits(1e-300, 0.0, ab_abs_hiword(INFINITY)), "an Inf in C passed");
    // The GPU epilogue's flag beside beta = 1 refuses exactly where c_fits does.
    const double cs[] = {0x1p1021, below, 0x1p1022, -0x1p1022, nextafter(0x1p1022, 0x1p1023), 0x1p1023,
                         DBL_MAX, INFINITY, -INFINITY, NAN, 1.0, 0x1p-1060, 0.0};
    for (size_t i = 0; i < TEST_REGISTRY_SIZE(cs); i++) {
        uint64_t u;
        memcpy(&u, &cs[i], sizeof u);
        bool flagged = (u & ~(1ULL << 63)) >= 0x7FD0000000000000ULL;
        expect(ab_zgemm_c_fits(1.0, 0.0, ab_abs_hiword(cs[i])) == !flagged,
               "c = %a: c_fits and the GPU flag disagree", cs[i]);
    }
    // The claim, on random calls near the edges: wherever both pass, the
    // worst case stays below 2^1023 (1 + 2^-50).
    rng_state = 0x5EED0008;
    int passed = 0, refused = 0;
    for (int t = 0; t < 200000; t++) {
        const int K = rnd_int(1, 1 << 20);
        const int ea = rnd_int(400, 620), eb = 1013 - ilogb((double)K) - ea + rnd_int(0, 11);
        const double a = fabs(rnd_val(ea)), b = fabs(rnd_val(eb));
        const double ar = rnd_val(rnd_int(-8, 0)), ai = rnd_bool(0.67) ? rnd_val(rnd_int(-8, 0)) : 0.0;
        const double br = rnd_val(rnd_int(-4, 3)), bi = rnd_bool(0.67) ? rnd_val(rnd_int(-4, 1)) : 0.0;
        const double c = fabs(rnd_val(rnd_int(1016, 1023)));
        if (!epi_product(K, a, b, ar, ai) || !ab_zgemm_c_fits(br, bi, ab_abs_hiword(c))) {
            refused++;
            continue;
        }
        passed++;
        const int E = ilogb(a) + ilogb(b) + ilogb((double)K) + 5;
        const double w = epi_worst(ar, ai, br, bi, nextafter(ldexp(1.0, E), 0),
                                   nextafter(ldexp(1.0, ilogb(c) + 1), 0));
        if (!expect(isfinite(w) && w <= top, "K %d, |a| %a, |b| %a, alpha %a%+ai, beta %a%+ai, |c| %a: "
                    "worst %a", K, a, b, ar, ai, br, bi, c, w))
            break;
    }
    printf("(%d near the edges pass, %d refused) ", passed, refused);
    expect(passed > 1000 && refused > 1000, "only %d passed and %d refused near the edges", passed, refused);
    finish();
}

static const TestCase TESTS[] = {
    {"test_guard_random_real", test_guard_random_real},
    {"test_guard_random_complex", test_guard_random_complex},
    {"test_guard_large", test_guard_large},
    {"test_guard_crafted", test_guard_crafted},
    {"test_guard_tiny_values", test_guard_tiny_values},
    {"test_guard_c_random", test_guard_c_random},
    {"test_guard_flush_to_zero", test_guard_flush_to_zero},
    {"test_zgemm_epilogue_bounds", test_zgemm_epilogue_bounds},
};
static const int N_TESTS = (int)TEST_REGISTRY_SIZE(TESTS);

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  apple-bottom exponent-range guard, stage 2 vs its oracle (host) ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    for (int i = 0; i < N_TESTS; i++) {
        printf("[%d/%d] %-30s ... ", i + 1, N_TESTS, TESTS[i].name);
        fflush(stdout);
        int pre = tests_passed + tests_failed;
        TESTS[i].fn();
        int post = tests_passed + tests_failed;
        if (post == pre) {
            fprintf(stderr, "\nFATAL: ghost test — '%s' reported no result\n", TESTS[i].name);
            abort();
        }
    }
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    if (tests_failed == 0) {
        printf("✓ All %d tests PASSED\n", tests_passed);
        return 0;
    }
    printf("✗ Some tests FAILED\n");
    return 1;
}
