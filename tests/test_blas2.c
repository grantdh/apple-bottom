// =============================================================================
// test_blas2.c — BLAS-2 (DGEMV, ZGEMV, ZHEMV) on the GPU
// =============================================================================
//
// Entry points: ab_dev_dgemv / ab_dev_zgemv / ab_dev_zhemv on ab_dev_malloc()
// and ab_dev_register() buffers, ab_dgemv_blas / ab_zgemv_blas /
// ab_zhemv_blas, and the Fortran symbols dgemv_ / zgemv_ / zhemv_. This
// process runs with AB_MODE=gpu; routing in the other modes, illegal
// arguments, forced GPU failures and AB_REPORT run in child processes.
//
// A case passes when
//   - y agrees with a compensated FP64 reference (test_blas2_ref.h)
//     componentwise:
//       |y_i - ref_i| <= tol(K) (|alpha| (|op(A)| |x|)_i + |beta| |y0_i|),
//       tol(K) = 16 K w u^2 + 8 * 2^-53,  u^2 = 2^-48,
//     K the dot-product length and w = 1 (real) or 2 (complex), and so does
//     Accelerate's result (a check of the reference);
//   - the expected path computed y (ab_get_last_dispatch_path(): "gpu", or
//     "cpu" with y bitwise equal to Accelerate's on the same memory, since
//     Accelerate's bits depend on A's address modulo 256 bytes);
//   - A and x are bitwise unchanged, as is every element of y's array that
//     is not one of the call's strided elements.
// Leading-dimension padding of A, its unreferenced triangle and the
// imaginary parts of its diagonal (ZHEMV), and the gaps between strided
// vector elements hold NaN, so reading any of them shows in the result.
// =============================================================================

#include "apple_bottom.h"
#include "apple_bottom_device.h"
#include "blas2.h"             // ab_internal_b2_auto_gpu, ZHEMV tiling hooks
#include "device_internal.h"   // ab_internal_dev_data
#include "test_blas2_ref.h"
#include "test_blas_check.h"
#include "test_registry.h"
#include "test_spawn.h"

#include <Accelerate/Accelerate.h>
#include <complex.h>
#include <float.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define PASS() do { printf("✓ PASS\n"); tests_passed++; } while (0)
#define FAIL(msg) do { printf("✗ FAIL (%s)\n", msg); tests_failed++; } while (0)

#define U2 0x1p-48
#define MIN_GPU_DIM 32   // AB_DISPATCH_DEFAULT_MIN_DIM

static char self_path[PATH_MAX];

// Failures of the test in progress (main resets it); the first dozen print.
static int g_fails;

static void fail_case(const char *fmt, ...) {
    if (g_fails++ >= 12) return;
    va_list ap;
    va_start(ap, fmt);
    printf("\n    FAIL ");
    vprintf(fmt, ap);
    va_end(ap);
}

// Ends a test: PASS unless fail_case() was called.
static void finish(void) {
    if (g_fails) FAIL("see cases");
    else PASS();
}

static int g_cases;

// =============================================================================
// Cases and data
// =============================================================================

static const char *const RT_NAME[] = {"dgemv", "zgemv", "zhemv"};

typedef enum { EP_DEV, EP_REG, EP_BLAS, EP_FORTRAN, EP_COUNT } Entry;
static const char *const EP_NAME[] = {"ab_dev(malloc)", "ab_dev(register)", "ab_*_blas", "fortran"};

typedef struct {
    B2Routine rt;
    char op;                 // 'N', 'T', 'C' (GEMV) or 'U', 'L' (HEMV)
    int m, n, pad, incx, incy;
    double alpha[2], beta[2];
    Entry ep;
} Case;

typedef struct {
    int w, rows, lda, xlen, ylen;
    size_t na, nx, ny;       // doubles in A, x, y
    double *A, *x, *y0;
} Data;

static bool rows_out(const Case *c) { return c->rt == B2_ZHEMV || c->op == 'N'; }

static double *memdup(const double *v, size_t n) {
    double *d = malloc((n ? n : 1) * sizeof *d);
    memcpy(d, v, n * sizeof *d);
    return d;
}

static uint64_t g_rng = 0x5eed;
static double urand(void) { return (double)(splitmix(g_rng++) >> 11) * 0x1p-52 - 1.0; }

// Element (i, j) of A as stored, its real part first.
static double *aelem(const Data *d, int i, int j) { return &d->A[((size_t)j * d->lda + i) * d->w]; }

// The case's data: random, with NaN where nothing may be read.
static void make_data(const Case *c, Data *d) {
    d->w = c->rt == B2_DGEMV ? 1 : 2;
    d->rows = c->rt == B2_ZHEMV ? c->n : c->m;
    d->lda = imax(1, d->rows) + c->pad;
    d->xlen = rows_out(c) ? c->n : c->m;
    d->ylen = rows_out(c) ? c->m : c->n;
    d->na = (size_t)d->lda * imax(1, c->n) * d->w;
    d->nx = (size_t)(1 + (imax(1, d->xlen) - 1) * abs(c->incx)) * d->w;
    d->ny = (size_t)(1 + (imax(1, d->ylen) - 1) * abs(c->incy)) * d->w;
    d->A = malloc(d->na * sizeof(double));
    d->x = malloc(d->nx * sizeof(double));
    d->y0 = malloc(d->ny * sizeof(double));
    for (int j = 0; j < imax(1, c->n); j++)
        for (int i = 0; i < d->lda; i++) {
            bool unread = i >= d->rows ||
                          (c->rt == B2_ZHEMV && (c->op == 'U' ? i > j : i < j));
            for (int k = 0; k < d->w; k++)
                aelem(d, i, j)[k] = unread || (c->rt == B2_ZHEMV && i == j && k == 1) ? NAN : urand();
        }
    for (size_t k = 0; k < d->nx; k++) d->x[k] = NAN;
    for (size_t k = 0; k < d->ny; k++) d->y0[k] = NAN;
    for (int i = 0; i < d->xlen; i++)
        for (int k = 0; k < d->w; k++) d->x[b2_vidx(d->xlen, c->incx, i) * d->w + k] = urand();
    for (int i = 0; i < d->ylen; i++)
        for (int k = 0; k < d->w; k++) d->y0[b2_vidx(d->ylen, c->incy, i) * d->w + k] = urand();
}

static void free_data(Data *d) {
    free(d->A); free(d->x); free(d->y0);
}

// =============================================================================
// Reference and checks
// =============================================================================

// yref (ylen elements, contiguous) = alpha op(A) x + beta y0, and mag =
// |alpha| |op(A)| |x| + |beta| |y0|.
static void reference(const Case *c, const Data *d, double *yref, double *mag) {
    double *t = malloc((size_t)imax(1, d->ylen) * 2 * sizeof *t);
    b2_reference(c->rt, c->op, d->A, d->lda, d->x, d->xlen, c->incx, NULL, d->ylen, t, mag);
    const double ar = c->alpha[0], ai = d->w == 2 ? c->alpha[1] : 0;
    const double br = c->beta[0], bi = d->w == 2 ? c->beta[1] : 0;
    const bool beta0 = br == 0 && bi == 0;
    for (int i = 0; i < d->ylen; i++) {
        const double *y0 = d->y0 + b2_vidx(d->ylen, c->incy, i) * d->w;
        double y0r = y0[0], y0i = d->w == 2 ? y0[1] : 0, tr = t[2 * i], ti = t[2 * i + 1];
        yref[i * d->w] = ar * tr - ai * ti + (beta0 ? 0 : br * y0r - bi * y0i);
        if (d->w == 2) yref[i * 2 + 1] = ar * ti + ai * tr + (beta0 ? 0 : br * y0i + bi * y0r);
        mag[i] = hypot(ar, ai) * mag[i] + (beta0 ? 0 : hypot(br, bi) * hypot(y0r, y0i));
    }
    free(t);
}

static double tol_of(const Data *d) {
    return 16.0 * imax(1, d->xlen) * d->w * U2 + 8 * 0x1p-53;
}

// Largest componentwise error of the strided y against yref, output skip
// left out (-1: none); NaN if any is NaN.
static double y_error(const Case *c, const Data *d, const double *y, const double *yref,
                      const double *mag, int skip) {
    double worst = 0;
    for (int i = 0; i < d->ylen; i++) {
        if (i == skip) continue;
        const double *yi = y + b2_vidx(d->ylen, c->incy, i) * d->w;
        double diff = d->w == 2 ? hypot(yi[0] - yref[2 * i], yi[1] - yref[2 * i + 1])
                                : fabs(yi[0] - yref[i]);
        double e = diff / (mag[i] + DBL_MIN);
        if (isnan(e)) return NAN;
        if (e > worst) worst = e;
    }
    return worst;
}

// =============================================================================
// Running a case
// =============================================================================

static ABTranspose ab_trans(char op) {
    return op == 'N' ? AB_NO_TRANS : op == 'T' ? AB_TRANS : AB_CONJ_TRANS;
}

// Accelerate on ya (a copy of y0), with A at a.
static void run_accel(const Case *c, const Data *d, const double *a, double *ya) {
    if (c->rt == B2_DGEMV)
        cblas_dgemv(CblasColMajor, ref_trans(c->op, false), c->m, c->n, c->alpha[0], a, d->lda,
                    d->x, c->incx, c->beta[0], ya, c->incy);
    else if (c->rt == B2_ZGEMV)
        cblas_zgemv(CblasColMajor, ref_trans(c->op, true), c->m, c->n,
                    (const double complex *)c->alpha, (const double complex *)a, d->lda,
                    (const double complex *)d->x, c->incx, (const double complex *)c->beta,
                    (double complex *)ya, c->incy);
    else
        cblas_zhemv(CblasColMajor, c->op == 'U' ? CblasUpper : CblasLower, c->n,
                    (const double complex *)c->alpha, (const double complex *)a, d->lda,
                    (const double complex *)d->x, c->incx, (const double complex *)c->beta,
                    (double complex *)ya, c->incy);
}

static ABStatus run_dev(const Case *c, const Data *d, ab_dev_buffer_t buf, size_t off, double *y) {
    if (c->rt == B2_DGEMV)
        return ab_dev_dgemv(ab_trans(c->op), c->m, c->n, c->alpha[0], buf, off, d->lda, d->x,
                            c->incx, c->beta[0], y, c->incy);
    if (c->rt == B2_ZGEMV)
        return ab_dev_zgemv(ab_trans(c->op), c->m, c->n, c->alpha, buf, off, d->lda, d->x, c->incx,
                            c->beta, y, c->incy);
    return ab_dev_zhemv(c->op == 'U' ? AB_UPPER : AB_LOWER, c->n, c->alpha, buf, off, d->lda, d->x,
                        c->incx, c->beta, y, c->incy);
}

static void run_blas(const Case *c, const Data *d, double *y) {
    double complex al = CMPLX(c->alpha[0], c->alpha[1]), be = CMPLX(c->beta[0], c->beta[1]);
    if (c->rt == B2_DGEMV)
        ab_dgemv_blas(c->op, c->m, c->n, c->alpha[0], d->A, d->lda, d->x, c->incx, c->beta[0], y,
                      c->incy);
    else if (c->rt == B2_ZGEMV)
        ab_zgemv_blas(c->op, c->m, c->n, al, (const double complex *)d->A, d->lda,
                      (const double complex *)d->x, c->incx, be, (double complex *)y, c->incy);
    else
        ab_zhemv_blas(c->op, c->n, al, (const double complex *)d->A, d->lda,
                      (const double complex *)d->x, c->incx, be, (double complex *)y, c->incy);
}

static void run_fortran(const Case *c, const Data *d, double *y) {
    double complex al = CMPLX(c->alpha[0], c->alpha[1]), be = CMPLX(c->beta[0], c->beta[1]);
    if (c->rt == B2_DGEMV)
        fortran_dgemv(&c->op, &c->m, &c->n, &c->alpha[0], d->A, &d->lda, d->x, &c->incx,
                      &c->beta[0], y, &c->incy);
    else if (c->rt == B2_ZGEMV)
        fortran_zgemv(&c->op, &c->m, &c->n, &al, (const double complex *)d->A, &d->lda,
                      (const double complex *)d->x, &c->incx, &be, (double complex *)y, &c->incy);
    else
        fortran_zhemv(&c->op, &c->n, &al, (const double complex *)d->A, &d->lda,
                      (const double complex *)d->x, &c->incx, &be, (double complex *)y, &c->incy);
}

// Runs c through its entry point on y and, unless ya is NULL, Accelerate on
// ya with A at the same address (both copies of y0). Returns the path, or
// NULL with why set if the call failed or changed A.
static const char *run_entry(const Case *c, const Data *d, double *y, double *ya, char *why,
                             size_t why_len) {
    const size_t abytes = d->na * sizeof(double), elem = d->w * sizeof(double), off = 3;
    if (c->ep == EP_BLAS || c->ep == EP_FORTRAN) {
        double *acopy = memdup(d->A, d->na);
        if (ya) run_accel(c, d, d->A, ya);
        if (c->ep == EP_BLAS) run_blas(c, d, y);
        else run_fortran(c, d, y);
        bool same = !memcmp(acopy, d->A, abytes);
        free(acopy);
        if (!same) { snprintf(why, why_len, "A changed"); return NULL; }
        return ab_get_last_dispatch_path();
    }
    // A off elements into a device buffer: allocated, or caller memory
    // registered with the data 40 bytes past a malloc block's start.
    size_t bytes = abytes + off * elem;
    char *block = NULL;
    ab_dev_buffer_t buf;
    if (c->ep == EP_DEV) {
        buf = ab_dev_malloc(bytes);
        if (buf) ab_dev_memcpy_h2d(buf, off * elem, d->A, abytes);
    } else {
        block = malloc(bytes + 40);
        memcpy(block + 40 + off * elem, d->A, abytes);
        buf = ab_dev_register(block + 40, bytes);
    }
    if (!buf) { snprintf(why, why_len, "no device buffer"); free(block); return NULL; }
    const double *a = (const double *)((char *)ab_internal_dev_data(buf) + off * elem);
    if (ya) run_accel(c, d, a, ya);
    ABStatus st = run_dev(c, d, buf, off, y);
    bool same = !memcmp(a, d->A, abytes);
    ab_dev_free(buf);
    free(block);
    if (st != AB_OK) { snprintf(why, why_len, "status %s", ab_status_string(st)); return NULL; }
    if (!same) { snprintf(why, why_len, "A changed"); return NULL; }
    return ab_get_last_dispatch_path();
}

static bool gpu_expected(const Case *c) {
    if (c->m <= 0 || c->n <= 0 || (c->alpha[0] == 0 && c->alpha[1] == 0)) return false;
    return c->ep == EP_DEV || c->ep == EP_REG || (c->m > MIN_GPU_DIM && c->n > MIN_GPU_DIM);
}

static void case_name(const Case *c, char *out, size_t len) {
    snprintf(out, len, "%s '%c' %dx%d pad %d inc %d,%d alpha (%g,%g) beta (%g,%g) via %s",
             RT_NAME[c->rt], c->op, c->m, c->n, c->pad, c->incx, c->incy, c->alpha[0],
             c->alpha[1], c->beta[0], c->beta[1], EP_NAME[c->ep]);
}

typedef enum { EXPECT_GPU, EXPECT_CPU, EXPECT_EITHER } Expect;

// How the result is held to the reference: all outputs, all but output k
// (k >= 0; that one must be Inf or NaN), or not at all (NO_REFERENCE: NaN or
// Inf inputs and degenerate calls, where Accelerate's bits are the answer).
#define NO_REFERENCE (-2)

// Runs and checks one case on d (made by make_data, possibly edited); the
// GPU result's error is added to *worst unless NULL. Records failures.
static bool check(const Case *c, const Data *d, Expect expect, int ref, double *worst) {
    char name[192], why[160] = "";
    g_cases++;
    double *y = memdup(d->y0, d->ny), *ya = memdup(d->y0, d->ny), *xcopy = memdup(d->x, d->nx);
    const char *path = run_entry(c, d, y, ya, why, sizeof why);
    bool gpu = path && !strcmp(path, "gpu");
    if (!path) {
        // why says what went wrong
    } else if (memcmp(xcopy, d->x, d->nx * sizeof(double))) {
        snprintf(why, sizeof why, "x changed");
    } else if (!outside_block_unchanged(y, d->y0, d->ny / d->w, (Layout){1, d->ylen, abs(c->incy)},
                                        d->w * sizeof(double))) {
        snprintf(why, sizeof why, "y changed between strided elements");
    } else if ((expect == EXPECT_GPU && !gpu) || (expect == EXPECT_CPU && gpu) ||
               (!gpu && strcmp(path, "cpu"))) {
        snprintf(why, sizeof why, "path %s", path);
    } else if (!gpu && memcmp(y, ya, d->ny * sizeof(double))) {
        snprintf(why, sizeof why, "cpu path differs from Accelerate");
    } else if (ref != NO_REFERENCE && d->ylen > 0) {
        double *yref = malloc((size_t)d->ylen * d->w * sizeof *yref);
        double *mag = malloc((size_t)d->ylen * sizeof *mag);
        reference(c, d, yref, mag);
        double tol = tol_of(d), e = y_error(c, d, y, yref, mag, ref);
        double ea = y_error(c, d, ya, yref, mag, ref);
        if (!(e <= tol)) snprintf(why, sizeof why, "error %.3g > tol %.3g", e, tol);
        else if (!(ea <= tol)) snprintf(why, sizeof why, "Accelerate error %.3g > tol %.3g", ea, tol);
        else if (ref >= 0 && isfinite(y[b2_vidx(d->ylen, c->incy, ref) * d->w]))
            snprintf(why, sizeof why, "output %d finite, Accelerate's is not", ref);
        if (gpu && worst && e > *worst) *worst = e;
        free(yref);
        free(mag);
    }
    if (why[0]) {
        case_name(c, name, sizeof name);
        fail_case("%s: %s", name, why);
    }
    free(y); free(ya); free(xcopy);
    return !why[0];
}

// =============================================================================
// Semantics: every op, shapes from 1 x 1 to 30001 x 5, padded and tight lda,
// positive and negative increments, alpha and beta variants, all four entry
// points
// =============================================================================

static const int GEMV_SHAPES[][2] = {
    {1, 1}, {2, 3}, {3, 2}, {31, 33}, {33, 31}, {64, 64}, {100, 1}, {1, 100}, {257, 129},
    {129, 257}, {1023, 517}, {517, 1023}, {4099, 17}, {17, 4099}, {5, 30001}, {30001, 5},
    {2000, 2000},
};
static const int HEMV_SIZES[] = {1, 2, 3, 31, 33, 64, 129, 257, 1023, 2000};
static const int INCS[][2] = {{1, 1}, {2, 3}, {-1, -2}, {-3, 1}, {1, -1}};
static const double SCALARS[][4] = {   // alpha re, im, beta re, im
    {1, 0, 0, 0}, {0.75, -0.5, -0.5, 0.25}, {0, 1, 1, 0}, {-1.25, 0.5, 1, -1},
};

static void run_matrix(B2Routine rt) {
    int nshapes = rt == B2_ZHEMV ? (int)(sizeof HEMV_SIZES / sizeof *HEMV_SIZES)
                                 : (int)(sizeof GEMV_SHAPES / sizeof *GEMV_SHAPES);
    int k = 0;
    double worst = 0;
    for (const char *op = rt == B2_ZHEMV ? "UL" : "NTC"; *op; op++)
        for (int s = 0; s < nshapes; s++)
            for (int ep = 0; ep < EP_COUNT; ep++, k++) {
                const int *inc = INCS[k % 5];
                const double *sc = SCALARS[(k / 2) % 4];
                Case c = {rt, *op, rt == B2_ZHEMV ? HEMV_SIZES[s] : GEMV_SHAPES[s][0],
                          rt == B2_ZHEMV ? HEMV_SIZES[s] : GEMV_SHAPES[s][1], (k % 3) ? 5 : 0,
                          inc[0], inc[1], {sc[0], rt == B2_DGEMV ? 0 : sc[1]},
                          {sc[2], rt == B2_DGEMV ? 0 : sc[3]}, (Entry)ep};
                Data d;
                make_data(&c, &d);
                check(&c, &d, gpu_expected(&c) ? EXPECT_GPU : EXPECT_CPU, -1, &worst);
                free_data(&d);
            }
    printf("%d cases, worst GPU error %.2e ... ", k, worst);
    finish();
}

static void test_dgemv_semantics(void) { run_matrix(B2_DGEMV); }
static void test_zgemv_semantics(void) { run_matrix(B2_ZGEMV); }
static void test_zhemv_semantics(void) { run_matrix(B2_ZHEMV); }

// ZHEMV splits the rows into panels of 32 R rows (R = 1, or 3 from n = 4608)
// and the columns into chunks of whole panels (several only for
// 4096 <= n < 4608 and from n = 12288). Each (R, panels per chunk) below,
// forced, at sizes around a panel and a chunk: under one panel, one off, a
// ragged last panel, chunk and group of four columns; both triangles.
static void test_zhemv_tiling(void) {
    const int cfg[][2] = {{1, 1}, {1, 3}, {3, 1}, {3, 2}};   // rows per lane, panels per chunk
    int k = 0;
    for (int c = 0; c < 4; c++) {
        const int H = 32 * cfg[c][0], W = H * cfg[c][1];
        const int sizes[] = {1, 2, H - 1, H, H + 1, W + 1, 2 * W + H / 2 + 3};
        ab_internal_b2_hemv_override(cfg[c][0], cfg[c][1]);
        for (size_t s = 0; s < sizeof sizes / sizeof *sizes; s++) {
            int panel, chunk;
            ab_internal_b2_hemv_layout(sizes[s], &panel, &chunk);
            if (panel != H || chunk != W)
                fail_case("n %d: panel %d chunk %d, forced %d and %d", sizes[s], panel, chunk, H, W);
            for (const char *op = "UL"; *op; op++, k++) {
                const int *inc = INCS[k % 5];
                const double *sc = SCALARS[k % 4];
                Case cs = {B2_ZHEMV, *op, sizes[s], sizes[s], k % 2 ? 3 : 0, inc[0], inc[1],
                           {sc[0], sc[1]}, {sc[2], sc[3]}, k % 3 ? EP_DEV : EP_REG};
                Data d;
                make_data(&cs, &d);
                check(&cs, &d, EXPECT_GPU, -1, NULL);
                free_data(&d);
            }
        }
    }
    ab_internal_b2_hemv_override(0, 0);
    printf("%d cases ... ", k);
    finish();
}

// =============================================================================
// Special values
// =============================================================================

// The kernel families.
static const struct { B2Routine rt; char op; } FAMILIES[] = {
    {B2_DGEMV, 'N'}, {B2_DGEMV, 'T'}, {B2_ZGEMV, 'N'}, {B2_ZGEMV, 'C'}, {B2_ZHEMV, 'U'}, {B2_ZHEMV, 'L'},
};
#define N_FAMILIES ((int)(sizeof FAMILIES / sizeof *FAMILIES))

static Case family_case(int f, Entry ep, int m, int n) {
    Case c = {FAMILIES[f].rt, FAMILIES[f].op, m, FAMILIES[f].rt == B2_ZHEMV ? m : n, 3, 1, -2,
              {0.75, FAMILIES[f].rt == B2_DGEMV ? 0 : -0.25}, {0.5, 0}, ep};
    return c;
}

// beta = 0: y must be written without being read, NaN and Inf in it included.
static void test_beta0_nonfinite_y(void) {
    const Entry eps[] = {EP_DEV, EP_BLAS};
    for (int f = 0; f < N_FAMILIES; f++)
        for (int e = 0; e < 2; e++) {
            Case c = family_case(f, eps[e], 300, 200);
            c.beta[0] = c.beta[1] = 0;
            Data d;
            make_data(&c, &d);
            for (int i = 0; i < d.ylen; i++)
                d.y0[b2_vidx(d.ylen, c.incy, i) * d.w] = i % 3 == 0 ? NAN : i % 3 == 1 ? INFINITY : -INFINITY;
            check(&c, &d, EXPECT_GPU, -1, NULL);
            free_data(&d);
        }
    finish();
}

typedef enum { SP_NAN_A, SP_INF_A, SP_NAN_X, SP_INF_X, SP_NAN_ALPHA, SP_INF_BETA, SP_INF_Y,
               SP_COUNT } Special;

// NaN and Inf propagate as Accelerate propagates them: the call runs there
// (path "cpu", bitwise), except Inf in y with beta != 0, which the GPU's
// epilogue propagates to that output (the rest within tol).
static void test_nonfinite_propagation(void) {
    const Entry eps[] = {EP_DEV, EP_FORTRAN};
    for (int f = 0; f < N_FAMILIES; f++)
        for (int sp = 0; sp < SP_COUNT; sp++)
            for (int e = 0; e < 2; e++) {
                Case c = family_case(f, eps[e], 96, 80);
                Data d;
                make_data(&c, &d);
                // A referenced element: row 7 of column 11, or row 11 of column 7 in a lower triangle.
                double *a = c.rt == B2_ZHEMV && c.op == 'L' ? aelem(&d, 11, 7) : aelem(&d, 7, 11);
                switch (sp) {
                case SP_NAN_A: a[0] = NAN; break;
                case SP_INF_A: a[d.w - 1] = INFINITY; break;
                case SP_NAN_X: d.x[b2_vidx(d.xlen, c.incx, 5) * d.w] = NAN; break;
                case SP_INF_X: d.x[b2_vidx(d.xlen, c.incx, 5) * d.w] = -INFINITY; break;
                case SP_NAN_ALPHA: c.alpha[0] = NAN; break;
                case SP_INF_BETA: c.beta[0] = INFINITY; break;
                case SP_INF_Y: d.y0[b2_vidx(d.ylen, c.incy, 4) * d.w] = INFINITY; break;
                }
                if (sp == SP_INF_Y) check(&c, &d, EXPECT_GPU, 4, NULL);
                else check(&c, &d, EXPECT_CPU, NO_REFERENCE, NULL);
                free_data(&d);
            }
    finish();
}

// =============================================================================
// Range guard
// =============================================================================

typedef enum { RG_HUGE_ELEM, RG_ALL_TINY, RG_NEAR_FLT_MAX, RG_SCALED_UP, RG_X_HUGE, RG_X_TINY,
               RG_X_WIDE, RG_ZEROS, RG_SUBNORMALS, RG_TINY_ROWS, RG_COUNT } RangeKind;
// Where each must run (RG_TINY_ROWS: the guard may keep it on the GPU or not).
static const Expect RG_WANT[] = {
    EXPECT_CPU, EXPECT_CPU, EXPECT_CPU, EXPECT_GPU, EXPECT_GPU, EXPECT_GPU,
    EXPECT_GPU, EXPECT_GPU, EXPECT_GPU, EXPECT_EITHER,
};

// Multiplies every number in v (NaN stays NaN) by s.
static void scale_all(double *v, size_t n, double s) {
    for (size_t k = 0; k < n; k++) v[k] *= s;
}

static void test_range_guard(void) {
    for (int f = 0; f < N_FAMILIES; f++)
        for (int rg = 0; rg < RG_COUNT; rg++) {
            Case c = family_case(f, EP_DEV, 200, 150);
            Data d;
            make_data(&c, &d);
            switch (rg) {
            case RG_HUGE_ELEM: aelem(&d, 9, 9)[0] = 1e200; break;
            case RG_ALL_TINY: scale_all(d.A, d.na, 1e-200); break;
            case RG_NEAR_FLT_MAX: scale_all(d.A, d.na, 1e38); break;
            case RG_SCALED_UP: scale_all(d.A, d.na, 1e30); break;
            case RG_X_HUGE: scale_all(d.x, d.nx, 1e200); break;
            case RG_X_TINY: scale_all(d.x, d.nx, 1e-200); break;
            case RG_X_WIDE:   // odd components 1e-30 of the rest
                for (int i = 1; i < d.xlen; i += 2) d.x[b2_vidx(d.xlen, c.incx, i) * d.w] *= 1e-30;
                break;
            case RG_ZEROS:    // every 7th column, every 5th row and every 4th x element
                for (int j = 0; j < c.n; j++)
                    for (int i = 0; i < d.rows; i++)
                        if (j % 7 == 0 || i % 5 == 3) memset(aelem(&d, i, j), 0, d.w * sizeof(double));
                for (int i = 0; i < d.xlen; i += 4)
                    memset(&d.x[b2_vidx(d.xlen, c.incx, i) * d.w], 0, d.w * sizeof(double));
                break;
            case RG_SUBNORMALS:   // one number in 13 an FP64 subnormal
                for (size_t k = 0; k < d.na; k += 13) d.A[k] *= 1e-310;
                break;
            case RG_TINY_ROWS:    // every third row 1e-25 of the rest
                for (int i = 0; i < d.rows; i += 3)
                    for (int j = 0; j < c.n; j++) scale_all(aelem(&d, i, j), d.w, 1e-25);
                break;
            }
            check(&c, &d, RG_WANT[rg], -1, NULL);
            free_data(&d);
        }
    finish();
}

// =============================================================================
// Degenerate calls: m or n == 0, and alpha == 0 with A and x all NaN
// =============================================================================

static void test_degenerate(void) {
    const int shapes[][2] = {{0, 5}, {5, 0}, {0, 0}, {40, 40}};
    const double betas[] = {0, 1, 2.5};
    for (int f = 0; f < N_FAMILIES; f++)
        for (int s = 0; s < 4; s++)
            for (int b = 0; b < 3; b++)
                for (int ep = 0; ep < EP_COUNT; ep++) {
                    if (FAMILIES[f].rt == B2_ZHEMV && shapes[s][0] != shapes[s][1]) continue;
                    Case c = family_case(f, (Entry)ep, shapes[s][0], shapes[s][1]);
                    if (s == 3) c.alpha[0] = c.alpha[1] = 0;
                    c.beta[0] = betas[b];
                    c.beta[1] = 0;
                    Data d;
                    make_data(&c, &d);
                    if (s == 3) {
                        for (size_t k = 0; k < d.na; k++) d.A[k] = NAN;
                        for (size_t k = 0; k < d.nx; k++) d.x[k] = NAN;
                    }
                    check(&c, &d, EXPECT_CPU, NO_REFERENCE, NULL);
                    free_data(&d);
                }
    finish();
}

// =============================================================================
// Determinism, threads, precision
// =============================================================================

// The same call twice on a registered buffer and once through the BLAS ABI
// gives the same bits.
static void test_determinism(void) {
    for (int f = 0; f < N_FAMILIES; f++) {
        Case c = family_case(f, EP_REG, 1500, 1300);
        c.pad = 0;
        c.incy = 1;
        Data d;
        make_data(&c, &d);
        double *y[3];
        char why[160];
        for (int r = 0; r < 3; r++) {
            y[r] = memdup(d.y0, d.ny);
            c.ep = r < 2 ? EP_REG : EP_BLAS;
            const char *path = run_entry(&c, &d, y[r], NULL, why, sizeof why);
            if (!path || strcmp(path, "gpu")) fail_case("%s '%c' via %s: path %s", RT_NAME[c.rt], c.op, EP_NAME[c.ep], path ? path : why);
        }
        if (memcmp(y[0], y[1], d.ny * sizeof(double)) || memcmp(y[0], y[2], d.ny * sizeof(double)))
            fail_case("%s '%c': results differ between calls", RT_NAME[c.rt], c.op);
        for (int r = 0; r < 3; r++) free(y[r]);
        free_data(&d);
    }
    finish();
}

#define N_THREADS 8
#define THREAD_CALLS 10

typedef struct {
    Case c;
    Data d;
    ab_dev_buffer_t buf;
    double *expect, *y;
    int bad;
} ThreadJob;

static void *thread_main(void *arg) {
    ThreadJob *j = arg;
    for (int r = 0; r < THREAD_CALLS; r++) {
        memcpy(j->y, j->d.y0, j->d.ny * sizeof(double));
        if (run_dev(&j->c, &j->d, j->buf, 0, j->y) != AB_OK ||
            strcmp(ab_get_last_dispatch_path(), "gpu") ||
            memcmp(j->y, j->expect, j->d.ny * sizeof(double)))
            j->bad++;
    }
    return NULL;
}

// Eight threads, each its own matrix and kernel family, against
// single-threaded bits.
static void test_threads(void) {
    ThreadJob jobs[N_THREADS];
    pthread_t th[N_THREADS];
    for (int t = 0; t < N_THREADS; t++) {
        ThreadJob *j = &jobs[t];
        j->c = family_case(t % N_FAMILIES, EP_REG, 700 + 37 * t, 900 - 23 * t);
        j->c.pad = 0;
        make_data(&j->c, &j->d);
        j->buf = ab_dev_register(j->d.A, j->d.na * sizeof(double));
        j->expect = memdup(j->d.y0, j->d.ny);
        j->y = memdup(j->d.y0, j->d.ny);
        run_dev(&j->c, &j->d, j->buf, 0, j->expect);
        j->bad = 0;
    }
    for (int t = 0; t < N_THREADS; t++) pthread_create(&th[t], NULL, thread_main, &jobs[t]);
    int bad = 0;
    for (int t = 0; t < N_THREADS; t++) {
        pthread_join(th[t], NULL);
        bad += jobs[t].bad;
        ab_dev_free(jobs[t].buf);
        free(jobs[t].expect); free(jobs[t].y);
        free_data(&jobs[t].d);
    }
    if (bad) fail_case("%d of %d calls differ from the single-threaded bits", bad, N_THREADS * THREAD_CALLS);
    finish();
}

// Long dot products (K up to 65536): within tol, and under 2e-14 relative
// to |A||x|, a tripwire well below tol for a precision regression.
static void test_long_dot_products(void) {
    const struct { B2Routine rt; char op; int m, n; } L[] = {
        {B2_DGEMV, 'T', 65536, 6}, {B2_DGEMV, 'N', 6, 65536}, {B2_ZGEMV, 'C', 65536, 4},
        {B2_ZGEMV, 'N', 4, 65536}, {B2_ZHEMV, 'U', 4096, 4096},
    };
    for (size_t k = 0; k < sizeof L / sizeof *L; k++) {
        Case c = {L[k].rt, L[k].op, L[k].m, L[k].n, 0, 1, 1, {1, 0}, {0, 0}, EP_DEV};
        Data d;
        make_data(&c, &d);
        double err = 0;
        if (check(&c, &d, EXPECT_GPU, -1, &err) && err > 2e-14)
            fail_case("%s '%c' %dx%d: error %.3g over 2e-14", RT_NAME[c.rt], c.op, c.m, c.n, err);
        printf("%s'%c' %.1e ", RT_NAME[c.rt], c.op, err);
        free_data(&d);
    }
    finish();
}

// =============================================================================
// Registered memory and argument checks
// =============================================================================

static void test_registered_memory(void) {
    const long page = sysconf(_SC_PAGESIZE);
    // Updates to the caller's memory are seen by the next call: doubling A
    // doubles y exactly.
    Case c = {B2_DGEMV, 'N', 300, 200, 0, 1, 1, {1, 0}, {0, 0}, EP_REG};
    Data d;
    make_data(&c, &d);
    char *block = malloc(d.na * sizeof(double) + 40);
    double *A = (double *)(block + 40);
    memcpy(A, d.A, d.na * sizeof(double));
    ab_dev_buffer_t buf = ab_dev_register(A, d.na * sizeof(double));
    double *y1 = calloc(d.ny, sizeof(double)), *y2 = calloc(d.ny, sizeof(double));
    if (!buf || ab_dev_buffer_size(buf) != d.na * sizeof(double)) {
        fail_case("registering malloc'd memory failed");
    } else {
        ab_dev_dgemv(AB_NO_TRANS, c.m, c.n, 1.0, buf, 0, d.lda, d.x, 1, 0.0, y1, 1);
        for (size_t k = 0; k < d.na; k++) A[k] *= 2;
        ab_dev_dgemv(AB_NO_TRANS, c.m, c.n, 1.0, buf, 0, d.lda, d.x, 1, 0.0, y2, 1);
        for (int i = 0; i < c.m; i++)
            if (y2[i] != 2 * y1[i]) { fail_case("an update to registered memory was not seen"); break; }
        // The handle reads and writes the caller's bytes at their offset.
        double probe = 0, val = 12.5;
        if (ab_dev_memcpy_d2h(&probe, buf, 8 * sizeof(double), sizeof probe) != AB_OK || probe != A[8] ||
            ab_dev_memcpy_h2d(buf, 9 * sizeof(double), &val, sizeof val) != AB_OK || A[9] != val)
            fail_case("memcpy through a registered buffer missed the caller's bytes");
        ab_dev_free(buf);
        A[0] = 1.0;   // still the caller's memory
    }
    free(y1); free(y2); free(block);
    free_data(&d);
    // Refusals: NULL, zero bytes, a span with a hole, unmapped memory, PROT_NONE.
    char *map = mmap(NULL, 3 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (ab_dev_register(NULL, 64) || ab_dev_register(map, 0)) fail_case("NULL or empty span accepted");
    munmap(map + page, page);
    const struct { char *p; size_t len; const char *what; } bad[] = {
        {map, 3 * page, "a span with an unmapped page"}, {map + page, page, "an unmapped page"},
    };
    for (int k = 0; k < 2; k++) {
        ab_dev_buffer_t b = ab_dev_register(bad[k].p, bad[k].len);
        if (b) { fail_case("%s accepted", bad[k].what); ab_dev_free(b); }
    }
    mprotect(map, page, PROT_NONE);
    ab_dev_buffer_t none = ab_dev_register(map, page);
    if (none) { fail_case("a PROT_NONE page accepted"); ab_dev_free(none); }
    munmap(map, page);
    munmap(map + 2 * page, page);
    finish();
}

static void test_invalid_dev_args(void) {
    double A[64] = {0}, x[8] = {1, 1, 1, 1, 1, 1, 1, 1}, y[8], y0[8] = {3, 3, 3, 3, 3, 3, 3, 3};
    const double one[2] = {1, 0}, zero[2] = {0, 0};
    ab_dev_buffer_t b = ab_dev_malloc(sizeof A);
    ab_dev_memcpy_h2d(b, 0, A, sizeof A);
#define TRY(label, call) do { memcpy(y, y0, sizeof y); ABStatus st_ = (call); \
        if (st_ != AB_ERROR_INVALID_ARG || memcmp(y, y0, sizeof y)) \
            fail_case("%s: %s", label, ab_status_string(st_)); } while (0)
    TRY("bad trans", ab_dev_dgemv((ABTranspose)7, 4, 4, 1.0, b, 0, 4, x, 1, 0.0, y, 1));
    TRY("m < 0", ab_dev_dgemv(AB_NO_TRANS, -1, 4, 1.0, b, 0, 4, x, 1, 0.0, y, 1));
    TRY("lda < m", ab_dev_dgemv(AB_NO_TRANS, 5, 4, 1.0, b, 0, 4, x, 1, 0.0, y, 1));
    TRY("incx = 0", ab_dev_dgemv(AB_TRANS, 4, 4, 1.0, b, 0, 4, x, 0, 0.0, y, 1));
    TRY("incy = 0", ab_dev_dgemv(AB_TRANS, 4, 4, 1.0, b, 0, 4, x, 1, 0.0, y, 0));
    TRY("NULL x", ab_dev_dgemv(AB_TRANS, 4, 4, 1.0, b, 0, 4, NULL, 1, 0.0, y, 1));
    TRY("NULL A", ab_dev_dgemv(AB_TRANS, 4, 4, 1.0, NULL, 0, 4, x, 1, 0.0, y, 1));
    TRY("A past the buffer", ab_dev_dgemv(AB_NO_TRANS, 8, 8, 1.0, b, 1, 8, x, 1, 0.0, y, 1));
    TRY("complex A past the buffer", ab_dev_zgemv(AB_NO_TRANS, 4, 9, one, b, 0, 4, x, 1, zero, y, 1));
    TRY("NULL alpha", ab_dev_zgemv(AB_NO_TRANS, 2, 2, NULL, b, 0, 2, x, 1, zero, y, 1));
    TRY("bad uplo", ab_dev_zhemv((ABUplo)5, 2, one, b, 0, 2, x, 1, zero, y, 1));
    TRY("hemv lda < n", ab_dev_zhemv(AB_UPPER, 3, one, b, 0, 2, x, 1, zero, y, 1));
#undef TRY
    ab_dev_free(b);
    finish();
}

// =============================================================================
// AB_MODE=auto rule against its measurements
// =============================================================================

// Every shape of both committed sweep runs that the auto rule sends to the
// GPU was measured >= 1.10x faster there, and the rule routes some shapes of
// exactly the classes DGEMV 'N', ZGEMV 'N' and ZHEMV. ZHEMV's runs are those
// of its single-read kernel; the GEMV directory's ZHEMV rows measured the
// two-pass kernel it replaced and are not used.
static void test_auto_rule_matches_data(void) {
    const struct { const char *file; bool hemv; } files[] = {
        {"2026-10-19-317bb91-gemv/sweep_square.csv", false},
        {"2026-10-19-317bb91-gemv/sweep_rect.csv", false},
        {"2026-10-19-317bb91-gemv/sweep_square_run1.csv", false},
        {"2026-10-19-317bb91-gemv/sweep_rect_run1.csv", false},
        {"2026-10-19-1bc503a-zhemv/sweep_square.csv", true},
        {"2026-10-19-1bc503a-zhemv/sweep_square_run1.csv", true},
    };
    enum { DN, DT, ZN, ZT, HE, CLASSES };
    const bool want_routed[CLASSES] = {true, false, true, false, true};
    int routed = 0, kept_wins = 0, rows = 0, per_class[CLASSES] = {0};
    for (size_t f = 0; f < sizeof files / sizeof *files; f++) {
        char path[PATH_MAX], line[256];
        snprintf(path, sizeof path, "%s/benchmarks/results/%s", AB_SOURCE_ROOT, files[f].file);
        FILE *in = fopen(path, "r");
        if (!in) { fail_case("cannot read %s", path); continue; }
        while (fgets(line, sizeof line, in)) {
            char routine[8], op;
            int m, n;
            double speedup;
            if (sscanf(line, "%7[a-z],%c,%d,%d,%*d,%*[^,],%*f,%*f,%lf", routine, &op, &m, &n,
                       &speedup) != 5)
                continue;   // the header
            B2Routine rt = !strcmp(routine, "dgemv") ? B2_DGEMV : !strcmp(routine, "zgemv") ? B2_ZGEMV : B2_ZHEMV;
            if ((rt == B2_ZHEMV) != files[f].hemv) continue;
            int cls = rt == B2_ZHEMV ? HE : rt == B2_DGEMV ? (op == 'N' ? DN : DT) : (op == 'N' ? ZN : ZT);
            rows++;
            if (ab_internal_b2_auto_gpu((ab_b2_routine_t)rt, op, m, n)) {
                routed++;
                per_class[cls]++;
                if (speedup < 1.10) fail_case("%s '%c' %dx%d routed at %.4fx (%s)", routine, op, m, n, speedup, files[f].file);
            } else if (speedup >= 1.10) {
                kept_wins++;
            }
        }
        fclose(in);
    }
    const char *names[CLASSES] = {"dgemv N", "dgemv T", "zgemv N", "zgemv T/C", "zhemv"};
    for (int k = 0; k < CLASSES; k++)
        if ((per_class[k] > 0) != want_routed[k])
            fail_case("%s: %d measured shapes routed", names[k], per_class[k]);
    printf("%d shapes, %d routed (all >= 1.10x), %d wins kept on Accelerate ... ", rows, routed, kept_wins);
    finish();
}

// =============================================================================
// Child processes: routing, illegal arguments, forced failures, AB_REPORT
// =============================================================================

// Runs this binary with args and the extra environment entries.
static int run_self(char *const args[], const char *const extra[], char **out) {
    char *argv[8] = {self_path};
    for (int k = 0; args[k] && k < 6; k++) argv[k + 1] = args[k];
    return spawn_capture(argv, extra, out);
}

// Child: one BLAS-ABI call of each routine at size n; prints the paths.
static int child_paths(int n) {
    const char ops[] = {'N', 'C', 'U'};
    for (int k = 0; k < 3; k++) {
        Case c = {(B2Routine)k, ops[k], n, n, 0, 1, 1, {1, 0}, {0, 0}, EP_BLAS};
        Data d;
        make_data(&c, &d);
        double *y = memdup(d.y0, d.ny);
        run_blas(&c, &d, y);
        printf("%s=%s\n", RT_NAME[k], ab_get_last_dispatch_path());
        free(y);
        free_data(&d);
    }
    return 0;
}

static void test_routing_modes(void) {
    struct { const char *env[3]; const char *n; const char *d, *z, *h; } R[] = {
        {{"AB_MODE=cpu", NULL}, "600", "cpu", "cpu", "cpu"},
        {{"AB_MODE=gpu", NULL}, "600", "gpu", "gpu", "gpu"},
        {{"AB_MODE=gpu", NULL}, "32", "cpu", "cpu", "cpu"},
        {{"AB_MODE=gpu", "AB_MIN_GPU_DIM=700", NULL}, "600", "cpu", "cpu", "cpu"},
        // AB_MODE=auto (src/blas2_abi.c, kRoutes): below every row, then
        // dgemv 'N' and zhemv on the GPU while zgemv 'C' never is.
        {{NULL}, "64", "cpu", "cpu", "cpu"},
        {{"AB_MODE=auto", NULL}, "1024", "cpu", "cpu", "cpu"},
        {{"AB_MODE=auto", NULL}, "2048", "gpu", "cpu", "gpu"},
    };
    for (size_t k = 0; k < sizeof R / sizeof *R; k++) {
        char *args[] = {"--child-paths", (char *)R[k].n, NULL}, *out, want[96];
        int st = run_self(args, R[k].env, &out);
        snprintf(want, sizeof want, "dgemv=%s\nzgemv=%s\nzhemv=%s\n", R[k].d, R[k].z, R[k].h);
        if (st != 0 || !strstr(out, want))
            fail_case("%s %s n=%s: %s", R[k].env[0] ? R[k].env[0] : "AB_MODE unset",
                      R[k].env[0] && R[k].env[1] ? R[k].env[1] : "", R[k].n, out);
        free(out);
    }
    finish();
}

// Child: a call with one illegal argument, through the Fortran symbol.
static int child_illegal(int which) {
    double A[16] = {0}, x[4] = {1, 1, 1, 1}, y[4] = {0};
    double complex za[16] = {0}, zx[4] = {1, 1, 1, 1}, zy[4] = {0}, one = 1, zero = 0;
    int m = 4, n = 4, lda = 4, ld2 = 2, inc = 1, inc0 = 0;
    double a = 1, b = 0;
    char N = 'N', bad = 'X';
    switch (which) {
    case 0: fortran_dgemv(&N, &m, &n, &a, A, &lda, x, &inc0, &b, y, &inc); break;   // incx
    case 1: fortran_dgemv(&N, &m, &n, &a, A, &ld2, x, &inc, &b, y, &inc); break;    // lda < m
    case 2: fortran_dgemv(&bad, &m, &n, &a, A, &lda, x, &inc, &b, y, &inc); break;  // trans
    case 3: fortran_zgemv(&N, &m, &n, &one, za, &lda, zx, &inc, &zero, zy, &inc0); break;  // incy
    case 4: fortran_zhemv(&bad, &n, &one, za, &lda, zx, &inc, &zero, zy, &inc); break;     // uplo
    }
    printf("returned\n");
    return 0;
}

// Accelerate's cblas rejects an illegal argument by naming it and ending
// the process with status 255; dgemv_, zgemv_ and zhemv_ hand such calls to it.
static void test_illegal_args(void) {
    const char *expect[] = {"Parameter number 9 passed to cblas_dgemv",
                            "Parameter number 7 passed to cblas_dgemv",
                            "Parameter number 2 passed to cblas_dgemv", "cblas_zgemv",
                            "cblas_zhemv"};
    for (int k = 0; k < 5; k++) {
        char which[4], *out;
        snprintf(which, sizeof which, "%d", k);
        char *args[] = {"--child-illegal", which, NULL};
        const char *env[] = {"AB_MODE=gpu", NULL};
        int st = run_self(args, env, &out);
        if (st != 255 || !strstr(out, expect[k]) || strstr(out, "returned"))
            fail_case("case %d: status %d, output: %s", k, st, out);
        free(out);
    }
    finish();
}

// Child: GPU-routed calls under AB_TEST_FAIL_STAGE run on Accelerate,
// bitwise; AB_REPORT counts the fallbacks.
static int child_fail_stage(void) {
    int bad = 0;
    const char ops[] = {'T', 'T', 'L'};
    for (int k = 0; k < 3; k++) {
        Case c = {(B2Routine)k, ops[k], 300, 300, 0, 1, 1, {1, 0}, {0.5, 0}, EP_BLAS};
        Data d;
        make_data(&c, &d);
        double *y = memdup(d.y0, d.ny), *ya = memdup(d.y0, d.ny);
        run_blas(&c, &d, y);
        run_accel(&c, &d, d.A, ya);
        if (strcmp(ab_get_last_dispatch_path(), "cpu") || memcmp(y, ya, d.ny * sizeof(double))) bad++;
        free(y); free(ya);
        free_data(&d);
    }
    printf("bad=%d\n", bad);
    return 0;
}

static void test_forced_gpu_failure(void) {
    const char *stages[] = {"AB_TEST_FAIL_STAGE=kernel", "AB_TEST_FAIL_STAGE=alloc"};
    for (int s = 0; s < 2; s++) {
        char *args[] = {"--child-fail", NULL}, *out;
        const char *env[] = {"AB_MODE=gpu", "AB_REPORT=1", stages[s], NULL};
        int st = run_self(args, env, &out);
        bool ok = st == 0 && strstr(out, "bad=0");
        for (int k = 0; k < 3 && ok; k++) {
            char want[96];
            snprintf(want, sizeof want, "%s fallbacks no_device=0 too_large=0 gpu_failed=1 other=0 range=0",
                     RT_NAME[k]);
            ok = strstr(out, want) != NULL;
        }
        if (!ok) fail_case("%s: status %d, output: %s", stages[s], st, out);
        free(out);
    }
    finish();
}

// AB_REPORT counts BLAS-2 calls by route: in gpu mode, a 300 x 300 call of
// each routine on the GPU and a 16 x 16 one on Accelerate.
static int child_report(void) {
    child_paths(300);
    child_paths(16);
    return 0;
}

static void test_report_counts(void) {
    char *args[] = {"--child-report", NULL}, *out;
    const char *env[] = {"AB_MODE=gpu", "AB_REPORT=1", NULL};
    int st = run_self(args, env, &out);
    bool ok = st == 0;
    for (int k = 0; k < 3 && ok; k++) {
        char want[96];
        snprintf(want, sizeof want, "%s calls=2 gpu_calls=1 cpu_calls=1", RT_NAME[k]);
        ok = strstr(out, want) != NULL;
    }
    if (!ok) fail_case("status %d, output: %s", st, out);
    free(out);
    finish();
}

// =============================================================================
// Registry
// =============================================================================

static const TestCase TESTS[] = {
    {"dgemv semantics ('N', 'T', 'C'; all entry points)", test_dgemv_semantics},
    {"zgemv semantics ('N', 'T', 'C'; all entry points)", test_zgemv_semantics},
    {"zhemv semantics ('U', 'L'; all entry points)", test_zhemv_semantics},
    {"zhemv tiling: panel and chunk edges", test_zhemv_tiling},
    {"beta = 0 with NaN and Inf in y", test_beta0_nonfinite_y},
    {"NaN and Inf propagate as in Accelerate", test_nonfinite_propagation},
    {"range guard", test_range_guard},
    {"m or n = 0, alpha = 0", test_degenerate},
    {"determinism", test_determinism},
    {"threads", test_threads},
    {"long dot products", test_long_dot_products},
    {"registered memory", test_registered_memory},
    {"invalid device-API arguments", test_invalid_dev_args},
    {"AB_MODE=auto rule against its measurements", test_auto_rule_matches_data},
    {"routing: AB_MODE, AB_MIN_GPU_DIM", test_routing_modes},
    {"illegal BLAS arguments (cblas xerbla)", test_illegal_args},
    {"forced GPU failures (AB_TEST_FAIL_STAGE)", test_forced_gpu_failure},
    {"AB_REPORT counts", test_report_counts},
};
#define N_TESTS ((int)TEST_REGISTRY_SIZE(TESTS))

int main(int argc, char **argv) {
    uint32_t len = sizeof self_path;
    if (_NSGetExecutablePath(self_path, &len) != 0) {
        fprintf(stderr, "executable path too long\n");
        return 2;
    }
    if (argc > 2 && !strcmp(argv[1], "--child-paths")) return child_paths(atoi(argv[2]));
    if (argc > 2 && !strcmp(argv[1], "--child-illegal")) return child_illegal(atoi(argv[2]));
    if (argc > 1 && !strcmp(argv[1], "--child-fail")) return child_fail_stage();
    if (argc > 1 && !strcmp(argv[1], "--child-report")) return child_report();

    clear_ab_env();
    setenv("AB_MODE", "gpu", 1);   // read once, at the first BLAS-ABI call
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  apple-bottom BLAS-2 Test Suite (GEMV, HEMV)                     ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    if (ab_init() != AB_OK) {
        printf("no Metal device\n");
        return 1;
    }
    for (int i = 0; i < N_TESTS; i++) {
        printf("[%d/%d] %s ... ", i + 1, N_TESTS, TESTS[i].name);
        fflush(stdout);
        int pre = tests_passed + tests_failed;
        g_fails = 0;
        TESTS[i].fn();
        if (tests_passed + tests_failed == pre) {
            fprintf(stderr, "\nFATAL: ghost test — '%s' reported no result\n", TESTS[i].name);
            abort();
        }
    }
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("Cases: %d checked\n", g_cases);
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    if (tests_failed == 0) {
        printf("✓ All %d tests PASSED\n", tests_passed);
        return 0;
    }
    printf("✗ Some tests FAILED\n");
    return 1;
}
