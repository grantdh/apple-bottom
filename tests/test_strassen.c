// =============================================================================
// test_strassen.c — the opt-in Strassen-Winograd products (src/strassen.m)
// =============================================================================
// ab_dgemm_strassen and ab_zgemm_strassen against a compensated FP64
// reference (Dot2, test_dot2.h) over odd, tiny and rectangular shapes,
// alpha and beta, and badly scaled rows, within the bounds of
// docs/STRASSEN.md:
//   - normwise, the bound of one level of Winograd's variant (Higham, Accuracy
//     and Stability of Numerical Algorithms, 2nd ed., Thm 23.3 with n0 = K/2):
//       max|Ĉ - C| <= (4.5 K^2 + 48 K) u max|A| max|B|,  u = 2^-45;
//   - and, as a result rounded to FP32 can meet that worst case on uniform
//     data from K around 8192, the Frobenius error relative to |A||B| against
//     FRO_TOL, which any loss of the DD low words would exceed.
// The classical GEMM is checked against its componentwise bound
// |Ĉ - C| <= (K + 2) u |A||B| on the same data, the bound Strassen does not
// keep: for badly scaled rows the test prints both componentwise errors.
// Strassen-Winograd takes the place of each real product inside the
// library's GEMMs, so what they do around the product holds: data far
// outside FP32's range (a power-of-two scale per matrix) gives the unscaled
// data's result times the scales, bit for bit, and a ZGEMM whose imaginary
// parts are 2^-24 of the real ones runs as 4M, its imaginary part within the
// bound of its own size, which Gauss-3M misses.
// Then argument checks, determinism across workspace reuse and ab_shutdown(),
// and routing, in child processes with their own environment: with
// AB_STRASSEN unset or 0, ab_dgemm, ab_dgemm_scaled and ab_zgemm are bitwise
// the classical GEMMs even with AB_STRASSEN_MIN=1; with AB_STRASSEN=1 a call
// with M, N and K >= AB_STRASSEN_MIN is bitwise the Strassen one natively,
// through dgemm_/zgemm_ on the staged path and on the host path
// (AB_HOST_PACK=1), balanced and imbalanced, and through ab_dev_dgemm and
// ab_dev_zgemm; a smaller one is the classical one.
// =============================================================================

#include "apple_bottom.h"
#include "apple_bottom_device.h"
#include "gpu_stage.h"         // ab_internal_last_staged
#include "test_blas_check.h"
#include "test_dot2.h"
#include "test_registry.h"
#include "test_spawn.h"
#include <Accelerate/Accelerate.h>
#include <float.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define U_DD 0x1p-45           // unit roundoff of the DD arithmetic in the bounds
#define FRO_TOL 0x1p-40        // Frobenius error / || |A||B| ||_F accepted
#define CHILD_TIMEOUT_S 120

static int tests_passed = 0, tests_failed = 0;
static char self_path[PATH_MAX];
static char g_why[512];        // the current test's first failure

// Records the first failure of the current test; returns cond.
static bool check(bool cond, const char *fmt, ...) {
    if (!cond && !g_why[0]) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(g_why, sizeof g_why, fmt, ap);
        va_end(ap);
    }
    return cond;
}

static void finish(void) {
    if (g_why[0]) {
        printf("✗ FAIL (%s)\n", g_why);
        tests_failed++;
        g_why[0] = 0;
    } else {
        printf("✓ PASS\n");
        tests_passed++;
    }
}

// =============================================================================
// Data, references and errors (row-major)
// =============================================================================

// n values uniform in [lo, hi), the same for the same seed.
static double *random_matrix(size_t n, double lo, double hi, uint64_t seed) {
    double *x = malloc(n * sizeof *x);
    for (size_t i = 0; i < n; i++)
        x[i] = lo + (hi - lo) * (double)(splitmix(seed + i * 0x9E3779B97F4A7C15ULL) >> 11) *
                   0x1p-53;
    return x;
}

static ABMatrix upload(int rows, int cols, const double *x) {
    ABMatrix m = ab_matrix_create(rows, cols);
    if (m && x) ab_matrix_upload(m, x, true);
    return m;
}

static double *download(ABMatrix m) {
    double *x = malloc(ab_matrix_count(m) * sizeof *x);
    ab_matrix_download(m, x, true);
    return x;
}

static bool same_bits(const double *x, const double *y, size_t n) {
    return !memcmp(x, y, n * sizeof *x);
}

static double max_abs(const double *x, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; i++) m = fmax(m, fabs(x[i]));
    return m;
}

// S + E += sign A B by Dot2, for the M x N product of M x K and K x N.
typedef struct { double *S, *E; } Acc;

static Acc acc_new(size_t n) {
    return (Acc){calloc(n, sizeof(double)), calloc(n, sizeof(double))};
}

// The accumulated value, rounded once, in place of S; frees E.
static double *acc_round(Acc a, size_t n) {
    for (size_t i = 0; i < n; i++) a.S[i] += a.E[i];
    free(a.E);
    return a.S;
}

typedef struct {
    double nw;    // max|X - R| / scale, scale = max|A| max|B|
    double fro;   // ||X - R||_F / || |A||B| ||_F
    double cw;    // max_ij |X - R|_ij / (|A||B|)_ij
} Err;

// Errors of X against R over n values; NaN if X has one.
static Err errors(const double *X, const double *R, const double *absAB, size_t n, double scale) {
    double worst = 0, num = 0, den = 0, cw = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs(X[i] - R[i]);
        if (isnan(d)) return (Err){NAN, NAN, NAN};
        worst = fmax(worst, d);
        num += d * d;
        den += absAB[i] * absAB[i];
        cw = fmax(cw, d / (absAB[i] + DBL_MIN));
    }
    return (Err){worst / scale, sqrt(num / fmax(den, DBL_MIN)), cw};
}

// The worst-case normwise bound of one level of Strassen-Winograd and the
// componentwise bound of the classical GEMM, for inner dimension K.
static double winograd_bound(int K) { return (4.5 * K * K + 48.0 * K) * U_DD; }
static double classical_cw_bound(int K) { return (K + 2.0) * U_DD; }

// =============================================================================
// DGEMM
// =============================================================================

typedef struct {
    double *A, *B, *R, *absAB;   // R = A B by Dot2
    int M, N, K;
} Problem;

static Problem problem(int M, int N, int K, double *A, double *B) {
    size_t mn = (size_t)M * N;
    Acc acc = acc_new(mn);
    dot2_gemm_acc(M, N, K, 1.0, A, B, acc.S, acc.E);
    double *absAB = calloc(mn, sizeof *absAB);
    abs_gemm_acc(M, N, K, A, NULL, B, NULL, absAB);
    return (Problem){A, B, acc_round(acc, mn), absAB, M, N, K};
}

static void problem_free(Problem p) {
    free(p.A); free(p.B); free(p.R); free(p.absAB);
}

// C = A B on the GPU, by Strassen or by the classical kernel.
static double *gpu_dgemm(const Problem *p, bool strassen) {
    ABMatrix A = upload(p->M, p->K, p->A), B = upload(p->K, p->N, p->B);
    ABMatrix C = upload(p->M, p->N, NULL);
    ABStatus s = strassen ? ab_dgemm_strassen(1.0, A, B, 0.0, C)
                          : ab_dgemm_scaled(1.0, A, B, 0.0, C);
    double *x = s == AB_OK ? download(C) : NULL;
    check(s == AB_OK, "%s %dx%dx%d: %s", strassen ? "Strassen" : "classical", p->M, p->K,
          p->N, ab_status_string(s));
    ab_matrix_destroy(A); ab_matrix_destroy(B); ab_matrix_destroy(C);
    return x;
}

// Strassen within its normwise bounds and the classical GEMM within its
// componentwise one; *s and *c get their errors.
static void check_dgemm(const Problem *p, Err *s, Err *c) {
    double scale = max_abs(p->A, (size_t)p->M * p->K) * max_abs(p->B, (size_t)p->K * p->N);
    size_t mn = (size_t)p->M * p->N;
    double *xs = gpu_dgemm(p, true), *xc = gpu_dgemm(p, false);
    if (!xs || !xc) { free(xs); free(xc); return; }
    *s = errors(xs, p->R, p->absAB, mn, scale);
    *c = errors(xc, p->R, p->absAB, mn, scale);
    char at[64];
    snprintf(at, sizeof at, "%dx%dx%d", p->M, p->K, p->N);
    check(s->nw <= winograd_bound(p->K), "%s: Strassen normwise %.3g > bound %.3g", at, s->nw,
          winograd_bound(p->K));
    check(s->fro <= FRO_TOL, "%s: Strassen Frobenius %.3g > %.3g", at, s->fro, FRO_TOL);
    check(c->cw <= classical_cw_bound(p->K), "%s: classical componentwise %.3g > bound %.3g",
          at, c->cw, classical_cw_bound(p->K));
    check(c->fro <= FRO_TOL, "%s: classical Frobenius %.3g > %.3g", at, c->fro, FRO_TOL);
    free(xs);
    free(xc);
}

static void test_dgemm_shapes(void) {
    // M, K, N: even, odd, a lone odd dimension, tiny, short and long K, and a
    // tall product whose half-size GEMMs take the tall-skinny kernel.
    static const int shapes[][3] = {
        {512, 512, 512}, {257, 511, 129}, {256, 129, 384}, {1, 1, 1},  {1, 2, 3},
        {3, 1, 2},       {2, 3, 1},       {5, 7, 3},       {600, 40, 700}, {64, 1500, 96},
        {2048, 256, 256},
    };
    for (size_t t = 0; t < sizeof shapes / sizeof shapes[0]; t++) {
        int M = shapes[t][0], K = shapes[t][1], N = shapes[t][2];
        Problem p = problem(M, N, K, random_matrix((size_t)M * K, -1, 1, 11 + t),
                            random_matrix((size_t)K * N, -1, 1, 101 + t));
        Err s, c;
        check_dgemm(&p, &s, &c);
        problem_free(p);
    }
    finish();
}

// Rows of A scaled by 2^e, e in [-20, 20]: the normwise bound still holds,
// the componentwise one only for the classical GEMM.
static void test_dgemm_scaled_rows(void) {
    const int M = 512, K = 512, N = 512;
    double *A = random_matrix((size_t)M * K, -1, 1, 7), *B = random_matrix((size_t)K * N, -1, 1, 8);
    for (int i = 0; i < M; i++) {
        double scale = ldexp(1.0, (int)(splitmix(1000 + i) % 41) - 20);
        for (int k = 0; k < K; k++) A[(size_t)i * K + k] *= scale;
    }
    Problem p = problem(M, N, K, A, B);
    Err s = {0}, c = {0};
    check_dgemm(&p, &s, &c);
    printf("(componentwise: classical %.1e, Strassen %.1e) ", c.cw, s.cw);
    problem_free(p);
    finish();
}

// alpha A B + beta C0 is alpha U + beta C0 for U = A B by the same algorithm,
// to DD precision; with beta = 0 the NaNs in C are never read.
static void test_dgemm_epilogue(void) {
    const int M = 300, K = 200, N = 250;
    const size_t mn = (size_t)M * N;
    double *a = random_matrix((size_t)M * K, -1, 1, 21);
    double *b = random_matrix((size_t)K * N, -1, 1, 22);
    double *c0 = random_matrix(mn, -1, 1, 23), *nan_c = malloc(mn * sizeof *nan_c);
    for (size_t i = 0; i < mn; i++) nan_c[i] = NAN;
    ABMatrix A = upload(M, K, a), B = upload(K, N, b), C = upload(M, N, NULL);

    check(ab_dgemm_strassen(1.0, A, B, 0.0, C) == AB_OK, "alpha 1, beta 0");
    double *u = download(C);
    static const double ab[][2] = {{-0.75, 1.25}, {2.5, 0.0}, {1.0, -1.0}};
    for (int t = 0; t < 3; t++) {
        double alpha = ab[t][0], beta = ab[t][1];
        ab_matrix_upload(C, beta != 0.0 ? c0 : nan_c, true);
        check(ab_dgemm_strassen(alpha, A, B, beta, C) == AB_OK, "alpha %g, beta %g", alpha, beta);
        double *x = download(C), worst = 0;
        for (size_t i = 0; i < mn; i++) {
            double want = alpha * u[i] + (beta != 0.0 ? beta * c0[i] : 0.0);
            double tol = 0x1p-44 * (fabs(alpha * u[i]) + (beta != 0.0 ? fabs(beta * c0[i]) : 0.0));
            double d = fabs(x[i] - want) / (tol + DBL_MIN);
            worst = isnan(d) ? INFINITY : fmax(worst, d);
        }
        check(worst <= 1.0, "alpha %g, beta %g: off by %.3g x the DD tolerance", alpha, beta,
              worst);
        free(x);
    }
    free(u); free(a); free(b); free(c0); free(nan_c);
    ab_matrix_destroy(A); ab_matrix_destroy(B); ab_matrix_destroy(C);
    finish();
}

// =============================================================================
// ZGEMM
// =============================================================================

// A ZGEMM problem: row-major parts with imaginary parts imag times uniform
// [-1, 1) data, the Dot2 references of Cr = Ar Br - Ai Bi and
// Ci = Ar Bi + Ai Br, and their |.||.| terms.
typedef struct {
    int M, K, N;
    double *ar, *ai, *br, *bi, *rr, *ri, *abs_r, *abs_i;
} ZProblem;

// The parts alone (no references), Ai imag_a and Bi imag_b times the size
// of uniform [-1, 1) data.
static ZProblem zparts(int M, int K, int N, double imag_a, double imag_b, uint64_t seed) {
    size_t mk = (size_t)M * K, kn = (size_t)K * N;
    ZProblem p = {M, K, N, random_matrix(mk, -1, 1, seed), random_matrix(mk, -1, 1, seed + 10),
                  random_matrix(kn, -1, 1, seed + 20), random_matrix(kn, -1, 1, seed + 30),
                  NULL, NULL, NULL, NULL};
    for (size_t i = 0; i < mk; i++) p.ai[i] *= imag_a;
    for (size_t i = 0; i < kn; i++) p.bi[i] *= imag_b;
    return p;
}

// p's references: Cr and Ci by Dot2, and their |.||.| terms.
static void zrefs(ZProblem *p) {
    int M = p->M, K = p->K, N = p->N;
    size_t mn = (size_t)M * N;
    Acc re = acc_new(mn), im = acc_new(mn);
    dot2_zgemm_acc(M, N, K, p->ar, p->ai, p->br, p->bi, (double *const[]){re.S, im.S},
                   (double *const[]){re.E, im.E});
    p->rr = acc_round(re, mn);
    p->ri = acc_round(im, mn);
    p->abs_r = calloc(mn, sizeof(double));
    p->abs_i = calloc(mn, sizeof(double));
    abs_gemm_acc(M, N, K, p->ar, NULL, p->br, NULL, p->abs_r);
    abs_gemm_acc(M, N, K, p->ai, NULL, p->bi, NULL, p->abs_r);
    abs_gemm_acc(M, N, K, p->ar, NULL, p->bi, NULL, p->abs_i);
    abs_gemm_acc(M, N, K, p->ai, NULL, p->br, NULL, p->abs_i);
}

static ZProblem zproblem(int M, int K, int N, double imag, uint64_t seed) {
    ZProblem p = zparts(M, K, N, imag, imag, seed);
    zrefs(&p);
    return p;
}

static void zproblem_free(ZProblem p) {
    free(p.ar); free(p.ai); free(p.br); free(p.bi);
    free(p.rr); free(p.ri); free(p.abs_r); free(p.abs_i);
}

// Cr and Ci by ab_zgemm_strassen or ab_zgemm into *xr, *xi; false on failure.
static bool gpu_zgemm(const ZProblem *p, bool strassen, double **xr, double **xi) {
    int M = p->M, K = p->K, N = p->N;
    ABMatrix Ar = upload(M, K, p->ar), Ai = upload(M, K, p->ai), Br = upload(K, N, p->br),
             Bi = upload(K, N, p->bi), Cr = upload(M, N, NULL), Ci = upload(M, N, NULL);
    ABStatus st = strassen ? ab_zgemm_strassen(Ar, Ai, Br, Bi, Cr, Ci)
                           : ab_zgemm(Ar, Ai, Br, Bi, Cr, Ci);
    bool ok = check(st == AB_OK, "%s ZGEMM %dx%dx%d: %s", strassen ? "Strassen" : "classical",
                    M, K, N, ab_status_string(st));
    *xr = ok ? download(Cr) : NULL;
    *xi = ok ? download(Ci) : NULL;
    ab_matrix_destroy(Ar); ab_matrix_destroy(Ai); ab_matrix_destroy(Br);
    ab_matrix_destroy(Bi); ab_matrix_destroy(Cr); ab_matrix_destroy(Ci);
    return ok;
}

static void test_zgemm(void) {
    static const int shapes[][3] = {{384, 256, 320}, {255, 129, 77}, {3, 1, 2}};
    for (int t = 0; t < 3; t++) {
        ZProblem p = zproblem(shapes[t][0], shapes[t][1], shapes[t][2], 1.0, 31 + t);
        size_t mk = (size_t)p.M * p.K, kn = (size_t)p.K * p.N, mn = (size_t)p.M * p.N;
        double scale = fmax(max_abs(p.ar, mk), max_abs(p.ai, mk)) *
                       fmax(max_abs(p.br, kn), max_abs(p.bi, kn));
        for (int strassen = 0; strassen < 2; strassen++) {
            double *xr, *xi;
            if (!gpu_zgemm(&p, strassen, &xr, &xi)) break;
            Err er = errors(xr, p.rr, p.abs_r, mn, scale),
                ei = errors(xi, p.ri, p.abs_i, mn, scale);
            const char *who = strassen ? "Strassen" : "classical";
            // Ci = (Ar + Ai)(Br + Bi) - Ar Br - Ai Bi: the first product's
            // operands are up to twice as large, so 4 + 1 + 1 times the bound.
            check(fmax(er.nw, ei.nw) <= 6.0 * winograd_bound(p.K),
                  "%s ZGEMM %dx%dx%d normwise %.3g > %.3g", who, p.M, p.K, p.N,
                  fmax(er.nw, ei.nw), 6.0 * winograd_bound(p.K));
            check(fmax(er.fro, ei.fro) <= FRO_TOL, "%s ZGEMM %dx%dx%d Frobenius %.3g > %.3g", who,
                  p.M, p.K, p.N, fmax(er.fro, ei.fro), FRO_TOL);
            free(xr);
            free(xi);
        }
        zproblem_free(p);
    }
    finish();
}

// Imaginary parts 2^-24 of the real ones give Gauss-3M a rho near 2^23, so
// the ZGEMM runs as 4M, by Strassen-Winograd too: each part within twice the
// bound for its own two products (Ci's scale is max|Ar| max|Bi| +
// max|Ai| max|Br|, about 2^-23). Gauss-3M's Ci would carry the rounding of
// (Ar + Ai)(Br + Bi), about 2^23 times that scale's share.
static void test_zgemm_imbalanced(void) {
    ZProblem p = zproblem(384, 256, 320, 0x1p-24, 131);
    size_t mk = (size_t)p.M * p.K, kn = (size_t)p.K * p.N, mn = (size_t)p.M * p.N;
    double sr = max_abs(p.ar, mk) * max_abs(p.br, kn) + max_abs(p.ai, mk) * max_abs(p.bi, kn);
    double si = max_abs(p.ar, mk) * max_abs(p.bi, kn) + max_abs(p.ai, mk) * max_abs(p.br, kn);
    double *xr, *xi;
    if (gpu_zgemm(&p, true, &xr, &xi)) {
        Err er = errors(xr, p.rr, p.abs_r, mn, sr), ei = errors(xi, p.ri, p.abs_i, mn, si);
        check(er.nw <= 2.0 * winograd_bound(p.K), "Cr normwise %.3g > %.3g", er.nw,
              2.0 * winograd_bound(p.K));
        check(ei.nw <= 2.0 * winograd_bound(p.K), "Ci normwise %.3g > %.3g (3M, not 4M?)", ei.nw,
              2.0 * winograd_bound(p.K));
        check(fmax(er.fro, ei.fro) <= FRO_TOL, "Frobenius %.3g > %.3g", fmax(er.fro, ei.fro),
              FRO_TOL);
        printf("(Ci normwise %.1e of its scale) ", ei.nw);
        free(xr);
        free(xi);
    }
    zproblem_free(p);
    finish();
}

// =============================================================================
// Power-of-two scales
// =============================================================================

// x * 2^e for n values (exact: none leaves the normal range).
static double *scaled_copy(const double *x, size_t n, int e) {
    double *y = malloc(n * sizeof *y);
    for (size_t i = 0; i < n; i++) y[i] = ldexp(x[i], e);
    return y;
}

// True if y is x * 2^e bit for bit over n values.
static bool same_bits_scaled(const double *x, const double *y, size_t n, int e) {
    for (size_t i = 0; i < n; i++) {
        double want = ldexp(x[i], e);
        if (memcmp(&want, &y[i], sizeof want)) return false;
    }
    return true;
}

// Data far outside FP32's range uploads with power-of-two scales (one per
// matrix, multiples of 16 near its centre), and FP32 arithmetic on the scaled
// DD values is the unscaled arithmetic times powers of two. So a product of
// A times 2^84 (scale 80) and B times 2^-20 (scale 0) is the unscaled one
// times 2^64, bit for bit, as for the classical GEMM:
//   - DGEMM with C times 2^64 (scale 48) and beta = 2^-20: the result's scale
//     follows the alpha term, so both alpha and beta are rescaled;
//   - ZGEMM by Gauss-3M (Ai 2^-20 of Ar, Bi as large as Br): Ai gets scale
//     48, so Ar and Ai are aligned before their sum;
//   - ZGEMM by 4M (Ai and Bi 2^-20 of the real parts; Bi gets scale -48):
//     Ar Br and Ai Bi, and Ai Br and Ar Bi, are on different scales, so each
//     pair is rescaled before it is combined.
// The unscaled result is checked against its references too, which with the
// comparison makes the scaled one right: the comparison alone is not enough,
// as a part of the unscaled data can be on another scale than its partner
// (its smallest entries set it), and then a missing alignment or rescaling
// would leave both results wrong alike.
static void scaled_zgemm(int M, int K, int N, double imag_b, uint64_t seed, const char *what) {
    const size_t mk = (size_t)M * K, kn = (size_t)K * N, mn = (size_t)M * N;
    ZProblem p = zparts(M, K, N, 0x1p-20, imag_b, seed), q = p;
    q.ar = scaled_copy(p.ar, mk, 84);
    q.ai = scaled_copy(p.ai, mk, 84);
    q.br = scaled_copy(p.br, kn, -20);
    q.bi = scaled_copy(p.bi, kn, -20);
    zrefs(&p);
    double *zr[2] = {NULL, NULL}, *zi[2] = {NULL, NULL};
    if (gpu_zgemm(&p, true, &zr[0], &zi[0]) && gpu_zgemm(&q, true, &zr[1], &zi[1])) {
        // Each part's scale is the size of its two products, as in
        // test_zgemm_imbalanced; 6 times the bound covers Gauss-3M's Ci.
        double sr = max_abs(p.ar, mk) * max_abs(p.br, kn) + max_abs(p.ai, mk) * max_abs(p.bi, kn);
        double si = max_abs(p.ar, mk) * max_abs(p.bi, kn) + max_abs(p.ai, mk) * max_abs(p.br, kn);
        Err er = errors(zr[0], p.rr, p.abs_r, mn, sr), ei = errors(zi[0], p.ri, p.abs_i, mn, si);
        check(fmax(er.nw, ei.nw) <= 6.0 * winograd_bound(K), "ZGEMM (%s) normwise %.3g > %.3g",
              what, fmax(er.nw, ei.nw), 6.0 * winograd_bound(K));
        check(same_bits_scaled(zr[0], zr[1], mn, 64),
              "ZGEMM (%s) Cr on scaled data is not the unscaled result times 2^64", what);
        check(same_bits_scaled(zi[0], zi[1], mn, 64),
              "ZGEMM (%s) Ci on scaled data is not the unscaled result times 2^64", what);
    }
    for (int s = 0; s < 2; s++) { free(zr[s]); free(zi[s]); }
    zproblem_free(p);
    zproblem_free(q);
}

static void test_scaled_data(void) {
    const int M = 257, K = 384, N = 129;
    const size_t mk = (size_t)M * K, kn = (size_t)K * N, mn = (size_t)M * N;
    const double alpha = -0.75, beta = 0x1p-20;
    double *a = random_matrix(mk, -1, 1, 71), *b = random_matrix(kn, -1, 1, 72);
    double *c = random_matrix(mn, -1, 1, 73);
    double *sa = scaled_copy(a, mk, 84), *sb = scaled_copy(b, kn, -20);
    double *sc = scaled_copy(c, mn, 64);
    double *x[2] = {NULL, NULL};
    for (int s = 0; s < 2; s++) {
        ABMatrix A = upload(M, K, s ? sa : a), B = upload(K, N, s ? sb : b);
        ABMatrix C = upload(M, N, s ? sc : c);
        if (check(ab_dgemm_strassen(alpha, A, B, beta, C) == AB_OK, "DGEMM, %s data",
                  s ? "scaled" : "unscaled"))
            x[s] = download(C);
        ab_matrix_destroy(A); ab_matrix_destroy(B); ab_matrix_destroy(C);
    }
    check(x[0] && x[1] && same_bits_scaled(x[0], x[1], mn, 64),
          "DGEMM on scaled data is not the unscaled result times 2^64");
    free(x[0]); free(x[1]); free(a); free(b); free(c); free(sa); free(sb); free(sc);

    scaled_zgemm(M, K, N, 1.0, 81, "3M");
    scaled_zgemm(M, K, N, 0x1p-20, 91, "4M");
    finish();
}

// =============================================================================
// Arguments
// =============================================================================

static void test_arguments(void) {
    ABMatrix A = upload(4, 5, NULL), B = upload(5, 3, NULL), C = upload(4, 3, NULL);
    ABMatrix B6 = upload(6, 3, NULL), C5 = upload(5, 3, NULL);
    check(ab_dgemm_strassen(1, NULL, B, 0, C) == AB_ERROR_INVALID_ARG, "NULL A");
    check(ab_dgemm_strassen(1, A, B, 0, NULL) == AB_ERROR_INVALID_ARG, "NULL C");
    check(ab_dgemm_strassen(1, A, B6, 0, C) == AB_ERROR_DIMENSION_MISMATCH, "inner dims");
    check(ab_dgemm_strassen(1, A, B, 0, C5) == AB_ERROR_DIMENSION_MISMATCH, "C dims");
    check(ab_zgemm_strassen(A, NULL, B, B, C, C) == AB_ERROR_INVALID_ARG, "NULL Ai");
    check(ab_zgemm_strassen(A, A, B, B6, C, C) == AB_ERROR_DIMENSION_MISMATCH, "Bi dims");
    check(ab_zgemm_strassen(A, A, B, B, C, C5) == AB_ERROR_DIMENSION_MISMATCH, "Ci dims");
    ab_matrix_destroy(A); ab_matrix_destroy(B); ab_matrix_destroy(C);
    ab_matrix_destroy(B6); ab_matrix_destroy(C5);
    finish();
}

// =============================================================================
// Determinism, workspace reuse and routing
// =============================================================================

// C after ab_dgemm (alpha 1, beta 0) or ab_dgemm_scaled, the classical batch
// GEMM, or Strassen. The batch GEMM runs the kernel ab_dgemm does for these
// shapes (M < 4N; the tall-skinny one from M >= 4N). For a ZGEMM, also a
// one-element ab_zgemm_batched (VIA_BATCHED).
enum { VIA_API, VIA_BATCH, VIA_STRASSEN, VIA_BATCHED };

static double *dgemm_via(int via, bool scaled, int M, int N, int K, uint64_t seed) {
    const double alpha = scaled ? -0.5 : 1.0, beta = scaled ? 2.0 : 0.0;
    double *a = random_matrix((size_t)M * K, -1, 1, seed);
    double *b = random_matrix((size_t)K * N, -1, 1, seed + 1);
    double *c0 = random_matrix((size_t)M * N, -1, 1, seed + 2);
    ABMatrix A = upload(M, K, a), B = upload(K, N, b), C = upload(M, N, c0);
    ABStatus s = AB_OK;
    if (via == VIA_STRASSEN) {
        s = ab_dgemm_strassen(alpha, A, B, beta, C);
    } else if (via == VIA_API) {
        s = scaled ? ab_dgemm_scaled(alpha, A, B, beta, C) : ab_dgemm(A, B, C);
    } else {
        ABBatch batch = ab_batch_create();
        s = scaled ? ab_batch_dgemm_scaled(batch, alpha, A, B, beta, C)
                   : ab_batch_dgemm(batch, A, B, C);
        if (s == AB_OK) s = ab_batch_commit(batch);
        if (s == AB_OK) s = ab_batch_wait(batch);
        ab_batch_destroy(batch);
    }
    double *x = s == AB_OK ? download(C) : NULL;
    ab_matrix_destroy(A); ab_matrix_destroy(B); ab_matrix_destroy(C);
    free(a); free(b); free(c0);
    return x;
}

static double *zgemm_via(int via, int M, int N, int K, uint64_t seed) {
    ABMatrix m[6];
    const int rows[6] = {M, M, K, K, M, M}, cols[6] = {K, K, N, N, N, N};
    for (int i = 0; i < 6; i++) {
        double *x = i < 4 ? random_matrix((size_t)rows[i] * cols[i], -1, 1, seed + i) : NULL;
        m[i] = upload(rows[i], cols[i], x);
        free(x);
    }
    ABStatus s = AB_OK;
    if (via == VIA_STRASSEN) {
        s = ab_zgemm_strassen(m[0], m[1], m[2], m[3], m[4], m[5]);
    } else if (via == VIA_BATCHED) {
        s = ab_zgemm_batched(1, &m[0], &m[1], &m[2], &m[3], &m[4], &m[5]);
    } else if (via == VIA_API) {
        s = ab_zgemm(m[0], m[1], m[2], m[3], m[4], m[5]);
    } else {
        ABBatch batch = ab_batch_create();
        s = ab_batch_zgemm(batch, m[0], m[1], m[2], m[3], m[4], m[5]);
        if (s == AB_OK) s = ab_batch_commit(batch);
        if (s == AB_OK) s = ab_batch_wait(batch);
        ab_batch_destroy(batch);
    }
    double *x = NULL;
    if (s == AB_OK) {   // Cr then Ci
        size_t mn = (size_t)M * N;
        x = malloc(2 * mn * sizeof *x);
        ab_matrix_download(m[4], x, true);
        ab_matrix_download(m[5], x + mn, true);
    }
    for (int i = 0; i < 6; i++) ab_matrix_destroy(m[i]);
    return x;
}

// True if the call through via1 and via2 give the same bits.
static bool dgemm_same(int via1, int via2, bool scaled, int M, int N, int K) {
    double *x = dgemm_via(via1, scaled, M, N, K, 77), *y = dgemm_via(via2, scaled, M, N, K, 77);
    bool same = x && y && same_bits(x, y, (size_t)M * N);
    free(x);
    free(y);
    return same;
}

static bool zgemm_same(int via1, int via2, int M, int N, int K) {
    double *x = zgemm_via(via1, M, N, K, 88), *y = zgemm_via(via2, M, N, K, 88);
    bool same = x && y && same_bits(x, y, 2 * (size_t)M * N);
    free(x);
    free(y);
    return same;
}

// The same call gives the same bits after the workspace served other shapes,
// a ZGEMM, and ab_shutdown().
static void test_reuse(void) {
    const int M = 1000, K = 1001, N = 999;
    double *first = dgemm_via(VIA_STRASSEN, false, M, N, K, 5);
    double *small = dgemm_via(VIA_STRASSEN, false, 600, 700, 800, 6);
    free(zgemm_via(VIA_STRASSEN, 1100, 1100, 1100, 9));
    double *again = dgemm_via(VIA_STRASSEN, false, M, N, K, 5);
    check(first && again && same_bits(first, again, (size_t)M * N), "bits changed after reuse");
    ab_shutdown();
    check(ab_init() == AB_OK, "ab_init after ab_shutdown");
    double *after = dgemm_via(VIA_STRASSEN, false, M, N, K, 5);
    double *small2 = dgemm_via(VIA_STRASSEN, false, 600, 700, 800, 6);
    check(after && same_bits(first, after, (size_t)M * N), "bits changed after ab_shutdown");
    check(small && small2 && same_bits(small, small2, (size_t)600 * 700), "small bits changed");
    free(first); free(small); free(again); free(after); free(small2);
    finish();
}

// One child scenario: this binary as --child <scenario> with env_add.
// Its output if it passed, else NULL (the caller frees it).
static char *run_child_out(const char *scenario, const char *const *env_add) {
    char *argv[] = {self_path, "--child", (char *)scenario, NULL}, *out;
    int rc = spawn_capture(argv, env_add, &out);
    if (!check(rc == 0, "child '%s' (exit %d; -1: no exit, e.g. its %d s alarm): %.300s",
               scenario, rc, CHILD_TIMEOUT_S, out ? out : "")) {
        free(out);
        return NULL;
    }
    return out;
}

static void run_child(const char *scenario, const char *const *env_add) {
    free(run_child_out(scenario, env_add));
}

static void test_routing_off(void) {
    run_child("off", (const char *const[]){"AB_STRASSEN_MIN=1", NULL});
    run_child("off", (const char *const[]){"AB_STRASSEN=0", "AB_STRASSEN_MIN=1", NULL});
    finish();
}

// The library's own uses of its GEMMs (ZHERK, DTRSM, iterative refinement, a
// one-element ZGEMM batch) stay classical under AB_STRASSEN=1 with every size
// routed: their bits (hashed by the child) are those of a run without it.
static void test_routing_internal(void) {
    char *on = run_child_out("internal",
                             (const char *const[]){"AB_STRASSEN=1", "AB_STRASSEN_MIN=1", NULL});
    char *off = run_child_out("internal", (const char *const[]){NULL});
    check(on && off && !strcmp(on, off), "internal GEMM uses changed under AB_STRASSEN:\n%s---\n%s",
          on ? on : "", off ? off : "");
    free(on);
    free(off);
    finish();
}

static void test_routing_on(void) {
    run_child("on",
              (const char *const[]){"AB_STRASSEN=1", "AB_STRASSEN_MIN=320", "AB_MODE=gpu", NULL});
    run_child("abi", (const char *const[]){"AB_STRASSEN=1", "AB_STRASSEN_MIN=320",
                                           "AB_MODE=gpu", "AB_HOST_PACK=1", NULL});
    run_child("default", (const char *const[]){"AB_STRASSEN=1", NULL});
    finish();
}

// Whether the last BLAS-ABI call ran on the GPU: on the staged path if
// staged, else on the host path.
static bool took_path(bool staged) {
    return !strcmp(ab_get_last_dispatch_path(), "gpu") && ab_internal_last_staged(NULL) == staged;
}

// The real and imaginary parts of n interleaved complex values.
static void deinterleave(const double *z, size_t n, double *re, double *im) {
    for (size_t i = 0; i < n; i++) {
        re[i] = z[2 * i];
        im[i] = z[2 * i + 1];
    }
}

// dgemm_ with op(A) = A^T, as the BLAS wrapper runs it on the GPU, against
// the same steps through ab_dgemm_strassen: pack op(A), op(B) and C
// row-major, compute, unpack. The call must have taken the staged path if
// staged, else the host path.
static bool child_dgemm_abi(int M, int N, int K, bool staged) {
    const double alpha = 1.5, beta = -1.0;
    const int lda = K + 3, ldb = K + 1, ldc = M + 2;   // A stored K x M (transposed), B K x N
    double *a = random_matrix((size_t)lda * M, -1, 1, 301);
    double *b = random_matrix((size_t)ldb * N, -1, 1, 302);
    double *c = random_matrix((size_t)ldc * N, -1, 1, 303);
    double *c_abi = malloc((size_t)ldc * N * sizeof *c_abi);
    memcpy(c_abi, c, (size_t)ldc * N * sizeof *c);
    fortran_dgemm("T", "N", &M, &N, &K, &alpha, a, &lda, b, &ldb, &beta, c_abi, &ldc);
    bool gpu = took_path(staged);

    double *opa = malloc((size_t)M * K * sizeof *opa), *opb = malloc((size_t)K * N * sizeof *opb);
    double *rc = malloc((size_t)M * N * sizeof *rc);
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) opa[(size_t)i * K + k] = a[(size_t)i * lda + k];
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++) opb[(size_t)k * N + j] = b[(size_t)j * ldb + k];
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) rc[(size_t)i * N + j] = c[(size_t)j * ldc + i];
    ABMatrix A = upload(M, K, opa), B = upload(K, N, opb), C = upload(M, N, rc);
    bool same = ab_dgemm_strassen(alpha, A, B, beta, C) == AB_OK;
    ab_matrix_download(C, rc, true);
    for (int i = 0; i < M && same; i++)
        for (int j = 0; j < N && same; j++) {
            double x = c_abi[(size_t)j * ldc + i], y = rc[(size_t)i * N + j];
            same = !memcmp(&x, &y, sizeof x);
        }
    ab_matrix_destroy(A); ab_matrix_destroy(B); ab_matrix_destroy(C);
    free(a); free(b); free(c); free(c_abi); free(opa); free(opb); free(rc);
    return gpu && same;
}

// zgemm_ (alpha 1, beta 0, no transposes), whose GPU path computes what
// ab_zgemm computes on the split operands, against ab_zgemm_strassen on the
// same split; imaginary parts imag times the real ones' size (2^-24: 4M).
// The call must have taken the staged path if staged, else the host path.
static bool child_zgemm_abi(int M, int N, int K, double imag, bool staged) {
    const double complex one = 1.0, zero = 0.0;
    double complex *a = malloc((size_t)M * K * sizeof *a), *b = malloc((size_t)K * N * sizeof *b);
    double complex *c = calloc((size_t)M * N, sizeof *c);
    double *re = random_matrix(2 * (size_t)M * K, -1, 1, 401);
    double *rb = random_matrix(2 * (size_t)K * N, -1, 1, 402);
    for (size_t i = 0; i < (size_t)M * K; i++) a[i] = CMPLX(re[2 * i], imag * re[2 * i + 1]);
    for (size_t i = 0; i < (size_t)K * N; i++) b[i] = CMPLX(rb[2 * i], imag * rb[2 * i + 1]);
    fortran_zgemm("N", "N", &M, &N, &K, &one, a, &M, b, &K, &zero, c, &M);
    bool gpu = took_path(staged);

    // Split row-major op(A), op(B) as zgemm_blas packs them.
    double *ar = malloc((size_t)M * K * sizeof *ar), *ai = malloc((size_t)M * K * sizeof *ai);
    double *br = malloc((size_t)K * N * sizeof *br), *bi = malloc((size_t)K * N * sizeof *bi);
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            ar[(size_t)i * K + k] = creal(a[(size_t)k * M + i]);
            ai[(size_t)i * K + k] = cimag(a[(size_t)k * M + i]);
        }
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++) {
            br[(size_t)k * N + j] = creal(b[(size_t)j * K + k]);
            bi[(size_t)k * N + j] = cimag(b[(size_t)j * K + k]);
        }
    ABMatrix Ar = upload(M, K, ar), Ai = upload(M, K, ai), Br = upload(K, N, br),
             Bi = upload(K, N, bi), Cr = upload(M, N, NULL), Ci = upload(M, N, NULL);
    bool same = ab_zgemm_strassen(Ar, Ai, Br, Bi, Cr, Ci) == AB_OK;
    double *xr = download(Cr), *xi = download(Ci);
    for (int i = 0; i < M && same; i++)
        for (int j = 0; j < N && same; j++) {
            double complex z = c[(size_t)j * M + i];
            size_t k = (size_t)i * N + j;
            double x[2] = {creal(z), cimag(z)}, y[2] = {xr[k], xi[k]};
            same = !memcmp(x, y, sizeof x);
        }
    ab_matrix_destroy(Ar); ab_matrix_destroy(Ai); ab_matrix_destroy(Br);
    ab_matrix_destroy(Bi); ab_matrix_destroy(Cr); ab_matrix_destroy(Ci);
    free(a); free(b); free(c); free(re); free(rb);
    free(ar); free(ai); free(br); free(bi); free(xr); free(xi);
    return gpu && same;
}

// ab_dev_dgemm ('N', 'N', alpha -0.5, beta 2) against ab_dgemm_strassen on
// the operands it converts: column-major A, B and C read row-major are A^T,
// B^T and C^T, and the device API computes C^T = alpha B^T A^T + beta C^T.
static bool child_dgemm_dev(int M, int N, int K) {
    const double alpha = -0.5, beta = 2.0;
    const size_t mk = (size_t)M * K, kn = (size_t)K * N, mn = (size_t)M * N;
    double *a = random_matrix(mk, -1, 1, 501), *b = random_matrix(kn, -1, 1, 502);
    double *c = random_matrix(mn, -1, 1, 503), *x = malloc(mn * sizeof *x);
    ab_dev_buffer_t dA = ab_dev_malloc(mk * sizeof *a), dB = ab_dev_malloc(kn * sizeof *b);
    ab_dev_buffer_t dC = ab_dev_malloc(mn * sizeof *c);
    bool ok = dA && dB && dC && ab_dev_memcpy_h2d(dA, 0, a, mk * sizeof *a) == AB_OK &&
              ab_dev_memcpy_h2d(dB, 0, b, kn * sizeof *b) == AB_OK &&
              ab_dev_memcpy_h2d(dC, 0, c, mn * sizeof *c) == AB_OK &&
              ab_dev_dgemm(AB_NO_TRANS, AB_NO_TRANS, M, N, K, alpha, dA, M, dB, K, beta, dC, M) ==
                  AB_OK &&
              ab_dev_memcpy_d2h(x, dC, 0, mn * sizeof *x) == AB_OK;
    ABMatrix At = upload(K, M, a), Bt = upload(N, K, b), Ct = upload(N, M, c);
    ok = ok && ab_dgemm_strassen(alpha, Bt, At, beta, Ct) == AB_OK;
    double *y = download(Ct);
    ok = ok && same_bits(x, y, mn);
    ab_matrix_destroy(At); ab_matrix_destroy(Bt); ab_matrix_destroy(Ct);
    ab_dev_free(dA); ab_dev_free(dB); ab_dev_free(dC);
    free(a); free(b); free(c); free(x); free(y);
    return ok;
}

// ab_dev_zgemm ('N', 'N', alpha 1, beta 0) against ab_zgemm_strassen on the
// split operands, C^T = B^T A^T as for ab_dev_dgemm.
static bool child_zgemm_dev(int M, int N, int K) {
    const double one[2] = {1.0, 0.0}, zero[2] = {0.0, 0.0};
    const size_t mk = (size_t)M * K, kn = (size_t)K * N, mn = (size_t)M * N;
    double *a = random_matrix(2 * mk, -1, 1, 601), *b = random_matrix(2 * kn, -1, 1, 602);
    double *x = malloc(2 * mn * sizeof *x);
    ab_dev_buffer_t dA = ab_dev_malloc(2 * mk * sizeof *a), dB = ab_dev_malloc(2 * kn * sizeof *b);
    ab_dev_buffer_t dC = ab_dev_malloc(2 * mn * sizeof *x);
    bool ok = dA && dB && dC && ab_dev_memcpy_h2d(dA, 0, a, 2 * mk * sizeof *a) == AB_OK &&
              ab_dev_memcpy_h2d(dB, 0, b, 2 * kn * sizeof *b) == AB_OK &&
              ab_dev_zgemm(AB_NO_TRANS, AB_NO_TRANS, M, N, K, one, dA, M, dB, K, zero, dC, M) ==
                  AB_OK &&
              ab_dev_memcpy_d2h(x, dC, 0, 2 * mn * sizeof *x) == AB_OK;
    // Real and imaginary parts of A^T (K x M) and B^T (N x K), row-major.
    double *p[4] = {malloc(mk * sizeof(double)), malloc(mk * sizeof(double)),
                    malloc(kn * sizeof(double)), malloc(kn * sizeof(double))};
    deinterleave(a, mk, p[0], p[1]);
    deinterleave(b, kn, p[2], p[3]);
    ABMatrix Atr = upload(K, M, p[0]), Ati = upload(K, M, p[1]), Btr = upload(N, K, p[2]),
             Bti = upload(N, K, p[3]), Ctr = upload(N, M, NULL), Cti = upload(N, M, NULL);
    ok = ok && ab_zgemm_strassen(Btr, Bti, Atr, Ati, Ctr, Cti) == AB_OK;
    double *yr = download(Ctr), *yi = download(Cti);
    for (size_t i = 0; i < mn && ok; i++)
        ok = !memcmp(&x[2 * i], &yr[i], sizeof *yr) && !memcmp(&x[2 * i + 1], &yi[i], sizeof *yi);
    ab_matrix_destroy(Atr); ab_matrix_destroy(Ati); ab_matrix_destroy(Btr);
    ab_matrix_destroy(Bti); ab_matrix_destroy(Ctr); ab_matrix_destroy(Cti);
    ab_dev_free(dA); ab_dev_free(dB); ab_dev_free(dC);
    for (int i = 0; i < 4; i++) free(p[i]);
    free(a); free(b); free(x); free(yr); free(yi);
    return ok;
}

// dgemm_ ('N', 'N') whose rows of op(A) from M/2 on are 2^-21 of the others:
// the BLAS ABI gives those rows scales of their own (outside 2^±20), so the
// call takes classical products, each element within the classical
// componentwise bound (K + 2) u of its own |A||B|. Strassen-Winograd's block
// sums would add the two kinds of rows and leave the larger rows' elements
// about 2^30 times that. The call must have taken the staged path if staged.
static void child_lines_scaled_dgemm(int M, int N, int K, bool staged) {
    const size_t mk = (size_t)M * K, kn = (size_t)K * N, mn = (size_t)M * N;
    const double one = 1.0, zero = 0.0;
    double *a = random_matrix(mk, -1, 1, 701), *b = random_matrix(kn, -1, 1, 702);
    for (int k = 0; k < K; k++)
        for (int i = M / 2; i < M; i++) a[(size_t)k * M + i] *= 0x1p-21;
    double *c = calloc(mn, sizeof *c);
    fortran_dgemm("N", "N", &M, &N, &K, &one, a, &M, b, &K, &zero, c, &M);
    bool path = took_path(staged);
    // C^T = B^T A^T, from the column-major arrays read row-major.
    Problem p = problem(N, M, K, b, a);
    double cw = errors(c, p.R, p.absAB, mn, 1.0).cw;
    problem_free(p);
    free(c);
    if (check(path, "dgemm_ with scaled rows took the wrong path"))
        check(cw <= classical_cw_bound(K), "dgemm_ with scaled rows: componentwise %.3g > %.3g",
              cw, classical_cw_bound(K));
}

// The same for zgemm_ (Gauss-3M), within 4 (K + 2) u of
// (|Ar| + |Ai|)(|Br| + |Bi|).
static void child_lines_scaled_zgemm(int M, int N, int K, bool staged) {
    const size_t mk = (size_t)M * K, kn = (size_t)K * N, mn = (size_t)M * N;
    const double complex one = 1.0, zero = 0.0;
    double *a = random_matrix(2 * mk, -1, 1, 711), *b = random_matrix(2 * kn, -1, 1, 712);
    for (int k = 0; k < K; k++)
        for (int i = M / 2; i < M; i++)
            for (int part = 0; part < 2; part++) a[2 * ((size_t)k * M + i) + part] *= 0x1p-21;
    double complex *c = calloc(mn, sizeof *c);
    fortran_zgemm("N", "N", &M, &N, &K, &one, (const double complex *)a, &M,
                  (const double complex *)b, &K, &zero, c, &M);
    bool path = took_path(staged);
    // Parts of A^T (K x M) and B^T (N x K), row-major; C^T = B^T A^T.
    double *at[2] = {malloc(mk * sizeof(double)), malloc(mk * sizeof(double))};
    double *bt[2] = {malloc(kn * sizeof(double)), malloc(kn * sizeof(double))};
    deinterleave(a, mk, at[0], at[1]);
    deinterleave(b, kn, bt[0], bt[1]);
    Acc re = acc_new(mn), im = acc_new(mn);
    dot2_zgemm_acc(N, M, K, bt[0], bt[1], at[0], at[1], (double *const[]){re.S, im.S},
                   (double *const[]){re.E, im.E});
    double *ref[2] = {acc_round(re, mn), acc_round(im, mn)};
    double *abs_ab = calloc(mn, sizeof *abs_ab), *x = malloc(mn * sizeof *x), cw = 0;
    abs_gemm_acc(N, M, K, bt[0], bt[1], at[0], at[1], abs_ab);
    for (int part = 0; part < 2; part++) {
        for (size_t i = 0; i < mn; i++) x[i] = part ? cimag(c[i]) : creal(c[i]);
        cw = fmax(cw, errors(x, ref[part], abs_ab, mn, 1.0).cw);
    }
    free(a); free(b); free(c); free(abs_ab); free(x);
    for (int part = 0; part < 2; part++) { free(at[part]); free(bt[part]); free(ref[part]); }
    if (check(path, "zgemm_ with scaled rows took the wrong path"))
        check(cw <= 4.0 * classical_cw_bound(K),
              "zgemm_ with scaled rows: componentwise %.3g > %.3g", cw,
              4.0 * classical_cw_bound(K));
}

// Prints a hash of the bits of n doubles, as name.
static void print_hash(const char *name, const double *x, size_t n) {
    printf("%s %016llx\n", name, (unsigned long long)fnv1a(x, n * sizeof *x, FNV_OFFSET));
}

// The same for m's elements.
static void print_matrix_hash(const char *name, ABMatrix m) {
    double *x = download(m);
    print_hash(name, x, ab_matrix_count(m));
    free(x);
}

// The library's own uses of its GEMMs, each result's bits hashed: ZHERK,
// a one-element ZGEMM batch, DTRSM and iterative refinement (ab_dgesv_mpir).
static void child_internal(void) {
    const int N = 256, K = 300;
    double *ar = random_matrix((size_t)N * K, -1, 1, 801),
           *ai = random_matrix((size_t)N * K, -1, 1, 802);
    ABMatrix Ar = upload(N, K, ar), Ai = upload(N, K, ai), Cr = upload(N, N, NULL),
             Ci = upload(N, N, NULL);
    check(ab_zherk(Ar, Ai, Cr, Ci) == AB_OK, "ab_zherk");
    print_matrix_hash("zherk.re", Cr);
    print_matrix_hash("zherk.im", Ci);
    ab_matrix_destroy(Ar); ab_matrix_destroy(Ai); ab_matrix_destroy(Cr); ab_matrix_destroy(Ci);
    free(ar); free(ai);

    double *z = zgemm_via(VIA_BATCHED, 300, 300, 300, 811);
    if (check(z != NULL, "ab_zgemm_batched")) print_hash("zgemm_batched", z, 2 * 300 * 300);
    free(z);

    // Lower triangular A with a dominant diagonal; B is T x R.
    const int T = 300, R = 40;
    double *a = random_matrix((size_t)T * T, -1, 1, 821),
           *b = random_matrix((size_t)T * R, -1, 1, 822);
    for (int i = 0; i < T; i++)
        for (int j = 0; j < T; j++) {
            double *x = &a[(size_t)i * T + j];
            *x = j > i ? 0.0 : i == j ? 2.0 : *x / T;
        }
    ABMatrix A = upload(T, T, a), B = upload(T, R, b);
    check(ab_dtrsm(AB_LEFT, AB_LOWER, AB_NO_TRANS, AB_NON_UNIT, 1.0, A, B) == AB_OK, "ab_dtrsm");
    print_matrix_hash("dtrsm", B);
    ab_matrix_destroy(A); ab_matrix_destroy(B);

    // A diagonally dominant system, 8 right-hand sides.
    for (int i = 0; i < T; i++) a[(size_t)i * T + i] = T;
    for (size_t i = 0; i < (size_t)T * T; i++) if (a[i] == 0.0) a[i] = 0.5 / T;
    A = upload(T, T, a);
    B = upload(T, 8, b);
    check(ab_dgesv_mpir(A, B) == AB_OK, "ab_dgesv_mpir");
    print_matrix_hash("dgesv_mpir", B);
    ab_matrix_destroy(A); ab_matrix_destroy(B);
    free(a); free(b);
}

// The BLAS-ABI calls of the on (staged) and abi (host path) scenarios.
static void child_abi(bool staged) {
    const char *how = staged ? "staged" : "host-path";
    check(child_dgemm_abi(330, 340, 320, staged), "%s dgemm_ did not run Strassen", how);
    check(child_zgemm_abi(320, 330, 340, 1.0, staged), "%s zgemm_ did not run Strassen", how);
    check(child_zgemm_abi(320, 330, 340, 0x1p-24, staged), "%s zgemm_ (4M) did not run Strassen",
          how);
    child_lines_scaled_dgemm(330, 340, 320, staged);
    child_lines_scaled_zgemm(320, 330, 340, staged);
}

// Child side: --child <scenario>, in the environment run_child gave it.
//   on       AB_STRASSEN=1 AB_STRASSEN_MIN=320 AB_MODE=gpu
//   abi      the same and AB_HOST_PACK=1 (the BLAS ABI's host path)
//   internal AB_STRASSEN=1 AB_STRASSEN_MIN=1, or neither: prints the hashes
//            of child_internal
//   default  AB_STRASSEN=1 (AB_STRASSEN_MIN 4096)
//   off      AB_STRASSEN unset or 0, AB_STRASSEN_MIN=1
// Prints the first failure, if any, for the parent.
static int child_main(const char *scenario) {
    if (ab_init() != AB_OK) {
        printf("no Metal device\n");
        return 1;
    }
    if (!strcmp(scenario, "on")) {
        check(dgemm_same(VIA_API, VIA_STRASSEN, false, 320, 340, 330),
              "ab_dgemm at the minimum is not Strassen");
        check(dgemm_same(VIA_API, VIA_STRASSEN, true, 320, 340, 330),
              "ab_dgemm_scaled at the minimum is not Strassen");
        check(dgemm_same(VIA_API, VIA_BATCH, false, 319, 340, 330),
              "ab_dgemm with M below the minimum is not classical");
        check(dgemm_same(VIA_API, VIA_BATCH, true, 330, 340, 319),
              "ab_dgemm_scaled with K below the minimum is not classical");
        check(zgemm_same(VIA_API, VIA_STRASSEN, 320, 340, 330),
              "ab_zgemm at the minimum is not Strassen");
        check(zgemm_same(VIA_API, VIA_BATCH, 330, 319, 340),
              "ab_zgemm with N below the minimum is not classical");
        child_abi(true);
        check(child_dgemm_dev(330, 340, 320), "ab_dev_dgemm did not run Strassen");
        check(child_zgemm_dev(320, 330, 340), "ab_dev_zgemm did not run Strassen");
    } else if (!strcmp(scenario, "abi")) {
        child_abi(false);
    } else if (!strcmp(scenario, "internal")) {
        child_internal();
    } else if (!strcmp(scenario, "default")) {
        check(dgemm_same(VIA_API, VIA_BATCH, false, 512, 512, 512),
              "ab_dgemm below the default minimum is not classical");
    } else if (!strcmp(scenario, "off")) {
        check(dgemm_same(VIA_API, VIA_BATCH, false, 320, 340, 330), "ab_dgemm is not classical");
        check(dgemm_same(VIA_API, VIA_BATCH, true, 320, 340, 330),
              "ab_dgemm_scaled is not classical");
        check(zgemm_same(VIA_API, VIA_BATCH, 320, 340, 330), "ab_zgemm is not classical");
    } else {
        check(false, "unknown scenario");
    }
    ab_shutdown();
    if (g_why[0]) printf("%s: %s\n", scenario, g_why);
    return g_why[0] ? 1 : 0;
}

// =============================================================================
// Main
// =============================================================================

static const TestCase TESTS[] = {
    {"test_dgemm_shapes",      test_dgemm_shapes},
    {"test_dgemm_scaled_rows", test_dgemm_scaled_rows},
    {"test_dgemm_epilogue",    test_dgemm_epilogue},
    {"test_zgemm",             test_zgemm},
    {"test_zgemm_imbalanced",  test_zgemm_imbalanced},
    {"test_scaled_data",       test_scaled_data},
    {"test_arguments",         test_arguments},
    {"test_reuse",             test_reuse},
    {"test_routing_off",       test_routing_off},
    {"test_routing_on",        test_routing_on},
    {"test_routing_internal",  test_routing_internal},
};
static const int N_TESTS = (int)TEST_REGISTRY_SIZE(TESTS);

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--child")) {
        alarm(CHILD_TIMEOUT_S);
        return child_main(argv[2]);
    }
    uint32_t len = sizeof self_path;
    if (_NSGetExecutablePath(self_path, &len) != 0) {
        fprintf(stderr, "test_strassen: cannot resolve executable path\n");
        return 1;
    }
    clear_ab_env();   // AB_STRASSEN and the rest only where a test sets them
    if (ab_init() != AB_OK) {
        fprintf(stderr, "test_strassen: no Metal device\n");
        return 1;
    }

    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  apple-bottom Strassen-Winograd GEMM (opt-in)                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    for (int i = 0; i < N_TESTS; i++) {
        printf("[%d/%d] %-24s ... ", i + 1, N_TESTS, TESTS[i].name);
        fflush(stdout);
        int pre = tests_passed + tests_failed;
        TESTS[i].fn();
        if (tests_passed + tests_failed == pre) {
            fprintf(stderr, "\nFATAL: ghost test — '%s' reported no result\n", TESTS[i].name);
            abort();
        }
    }
    ab_shutdown();

    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    if (tests_failed == 0) {
        printf("✓ All %d Strassen tests PASSED\n", tests_passed);
        return 0;
    }
    printf("✗ Some tests FAILED\n");
    return 1;
}
