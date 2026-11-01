// =============================================================================
// test_strassen.c — the opt-in Strassen-Winograd GEMM (src/strassen.m)
// =============================================================================
// ab_dgemm_strassen and ab_zgemm_strassen against a compensated FP64
// reference (Dot2, test_dot2.h) over odd, tiny and rectangular shapes,
// alpha and beta, and badly scaled rows, within the bounds of
// docs/STRASSEN.md:
//   - normwise, the bound of one level of Winograd's variant (Higham, Accuracy
//     and Stability of Numerical Algorithms, 2nd ed., Thm 23.3 with n0 = K/2):
//       max|Ĉ - C| <= (4.5 K^2 + 48 K) u max|A| max|B|,  u = 2^-45;
//   - and, as a result rounded to FP32 could meet that worst case on random
//     data for K >~ 4096, the Frobenius error relative to |A||B| against
//     FRO_TOL, which any loss of the DD low words would exceed.
// The classical GEMM is checked against its componentwise bound
// |Ĉ - C| <= (K + 2) u |A||B| on the same data, the bound Strassen does not
// keep: for badly scaled rows the test prints both componentwise errors.
// Then argument checks, determinism across workspace reuse and ab_shutdown(),
// and routing, in child processes with their own environment: with
// AB_STRASSEN unset or 0, ab_dgemm, ab_dgemm_scaled and ab_zgemm are bitwise
// the classical GEMMs even with AB_STRASSEN_MIN=1; with AB_STRASSEN=1 a call
// with M, N and K >= AB_STRASSEN_MIN, native or through dgemm_/zgemm_, is
// bitwise the Strassen one and a smaller one the classical one.
// =============================================================================

#include "apple_bottom.h"
#include "test_blas_check.h"
#include "test_dot2.h"
#include "test_registry.h"
#include "test_spawn.h"
#include <Accelerate/Accelerate.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <math.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
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

// P += |A||B| (FP64: it only scales errors).
static void abs_product_acc(int M, int N, int K, const double *A, const double *B, double *P) {
    double *a = malloc((size_t)M * K * sizeof *a), *b = malloc((size_t)K * N * sizeof *b);
    for (size_t i = 0; i < (size_t)M * K; i++) a[i] = fabs(A[i]);
    for (size_t i = 0; i < (size_t)K * N; i++) b[i] = fabs(B[i]);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, a, K, b, N, 1.0, P, N);
    free(a);
    free(b);
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
    abs_product_acc(M, N, K, A, B, absAB);
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

static void test_zgemm(void) {
    static const int shapes[][3] = {{384, 256, 320}, {255, 129, 77}, {3, 1, 2}};
    for (int t = 0; t < 3; t++) {
        int M = shapes[t][0], K = shapes[t][1], N = shapes[t][2];
        size_t mk = (size_t)M * K, kn = (size_t)K * N, mn = (size_t)M * N;
        double *ar = random_matrix(mk, -1, 1, 31 + t), *ai = random_matrix(mk, -1, 1, 41 + t);
        double *br = random_matrix(kn, -1, 1, 51 + t), *bi = random_matrix(kn, -1, 1, 61 + t);

        // Cr = Ar Br - Ai Bi, Ci = Ar Bi + Ai Br, and their |.||.| terms.
        Acc re = acc_new(mn), im = acc_new(mn);
        dot2_gemm_acc(M, N, K, 1.0, ar, br, re.S, re.E);
        dot2_gemm_acc(M, N, K, -1.0, ai, bi, re.S, re.E);
        dot2_gemm_acc(M, N, K, 1.0, ar, bi, im.S, im.E);
        dot2_gemm_acc(M, N, K, 1.0, ai, br, im.S, im.E);
        double *rr = acc_round(re, mn), *ri = acc_round(im, mn);
        double *abs_r = calloc(mn, sizeof(double)), *abs_i = calloc(mn, sizeof(double));
        abs_product_acc(M, N, K, ar, br, abs_r);
        abs_product_acc(M, N, K, ai, bi, abs_r);
        abs_product_acc(M, N, K, ar, bi, abs_i);
        abs_product_acc(M, N, K, ai, br, abs_i);
        double scale = fmax(max_abs(ar, mk), max_abs(ai, mk)) *
                       fmax(max_abs(br, kn), max_abs(bi, kn));

        ABMatrix Ar = upload(M, K, ar), Ai = upload(M, K, ai), Br = upload(K, N, br),
                 Bi = upload(K, N, bi), Cr = upload(M, N, NULL), Ci = upload(M, N, NULL);
        for (int strassen = 0; strassen < 2; strassen++) {
            ABStatus st = strassen ? ab_zgemm_strassen(Ar, Ai, Br, Bi, Cr, Ci)
                                   : ab_zgemm(Ar, Ai, Br, Bi, Cr, Ci);
            if (!check(st == AB_OK, "ZGEMM %dx%dx%d: %s", M, K, N, ab_status_string(st))) break;
            double *xr = download(Cr), *xi = download(Ci);
            Err er = errors(xr, rr, abs_r, mn, scale), ei = errors(xi, ri, abs_i, mn, scale);
            const char *who = strassen ? "Strassen" : "classical";
            // Ci = (Ar + Ai)(Br + Bi) - Ar Br - Ai Bi: the first product's
            // operands are up to twice as large, so 4 + 1 + 1 times the bound.
            check(fmax(er.nw, ei.nw) <= 6.0 * winograd_bound(K),
                  "%s ZGEMM %dx%dx%d normwise %.3g > %.3g", who, M, K, N, fmax(er.nw, ei.nw),
                  6.0 * winograd_bound(K));
            check(fmax(er.fro, ei.fro) <= FRO_TOL, "%s ZGEMM %dx%dx%d Frobenius %.3g > %.3g", who,
                  M, K, N, fmax(er.fro, ei.fro), FRO_TOL);
            free(xr);
            free(xi);
        }
        ab_matrix_destroy(Ar); ab_matrix_destroy(Ai); ab_matrix_destroy(Br);
        ab_matrix_destroy(Bi); ab_matrix_destroy(Cr); ab_matrix_destroy(Ci);
        free(ar); free(ai); free(br); free(bi); free(rr); free(ri); free(abs_r); free(abs_i);
    }
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
// shapes (M < 4N; the tall-skinny one from M >= 4N).
enum { VIA_API, VIA_BATCH, VIA_STRASSEN };

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
static void run_child(const char *scenario, const char *const *env_add) {
    size_t na = 0, k;
    while (env_add[na]) na++;
    char **env = child_environ(na, &k);
    for (size_t i = 0; i < na; i++) env[k++] = (char *)env_add[i];
    char *argv[] = {self_path, "--child", (char *)scenario, NULL}, why[256];
    pid_t pid;
    FILE *in = spawn_piped(argv, env, &pid, why, sizeof why);
    free(env);
    if (!check(in != NULL, "%s", why)) return;
    char *out = slurp(in);
    int status;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    bool timed_out = WIFSIGNALED(status) && WTERMSIG(status) == SIGALRM;
    check(WIFEXITED(status) && WEXITSTATUS(status) == 0, "child '%s' (status %d%s): %.300s",
          scenario, status, timed_out ? ", timed out" : "", out ? out : "");
    free(out);
}

static void test_routing_off(void) {
    run_child("off", (const char *const[]){"AB_STRASSEN_MIN=1", NULL});
    run_child("off", (const char *const[]){"AB_STRASSEN=0", "AB_STRASSEN_MIN=1", NULL});
    finish();
}

static void test_routing_on(void) {
    run_child("on",
              (const char *const[]){"AB_STRASSEN=1", "AB_STRASSEN_MIN=320", "AB_MODE=gpu", NULL});
    run_child("default", (const char *const[]){"AB_STRASSEN=1", NULL});
    finish();
}

// dgemm_ with op(A) = A^T, as the BLAS wrapper runs it on the GPU, against
// the same steps through ab_dgemm_strassen: pack op(A), op(B) and C
// row-major, compute, unpack.
static bool child_dgemm_abi(int M, int N, int K) {
    const double alpha = 1.5, beta = -1.0;
    const int lda = K + 3, ldb = K + 1, ldc = M + 2;   // A stored K x M (transposed), B K x N
    double *a = random_matrix((size_t)lda * M, -1, 1, 301);
    double *b = random_matrix((size_t)ldb * N, -1, 1, 302);
    double *c = random_matrix((size_t)ldc * N, -1, 1, 303);
    double *c_abi = malloc((size_t)ldc * N * sizeof *c_abi);
    memcpy(c_abi, c, (size_t)ldc * N * sizeof *c);
    fortran_dgemm("T", "N", &M, &N, &K, &alpha, a, &lda, b, &ldb, &beta, c_abi, &ldc);
    bool gpu = !strcmp(ab_get_last_dispatch_path(), "gpu");

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

// zgemm_ (alpha 1, beta 0, no transposes), whose GPU path runs ab_zgemm on
// the split operands, against ab_zgemm_strassen on the same split.
static bool child_zgemm_abi(int M, int N, int K) {
    const double complex one = 1.0, zero = 0.0;
    double complex *a = malloc((size_t)M * K * sizeof *a), *b = malloc((size_t)K * N * sizeof *b);
    double complex *c = calloc((size_t)M * N, sizeof *c);
    double *re = random_matrix(2 * (size_t)M * K, -1, 1, 401);
    double *rb = random_matrix(2 * (size_t)K * N, -1, 1, 402);
    for (size_t i = 0; i < (size_t)M * K; i++) a[i] = CMPLX(re[2 * i], re[2 * i + 1]);
    for (size_t i = 0; i < (size_t)K * N; i++) b[i] = CMPLX(rb[2 * i], rb[2 * i + 1]);
    fortran_zgemm("N", "N", &M, &N, &K, &one, a, &M, b, &K, &zero, c, &M);
    bool gpu = !strcmp(ab_get_last_dispatch_path(), "gpu");

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

// Child side: --child <scenario>, in the environment run_child gave it.
//   on       AB_STRASSEN=1 AB_STRASSEN_MIN=320 AB_MODE=gpu
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
        check(child_dgemm_abi(330, 340, 320), "dgemm_ did not run Strassen on the GPU");
        check(child_zgemm_abi(320, 330, 340), "zgemm_ did not run Strassen on the GPU");
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
    {"test_arguments",         test_arguments},
    {"test_reuse",             test_reuse},
    {"test_routing_off",       test_routing_off},
    {"test_routing_on",        test_routing_on},
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
