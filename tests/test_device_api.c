// =============================================================================
// test_device_api.c — Smoke tests for the device-buffer layer
// =============================================================================
//
// Exercises:
//   1. alloc → size → free
//   2. H2D → D2H roundtrip, byte-identical
//   3. D2D copy with source/destination offsets
//   4. memset then D2H
//   5. NULL / out-of-range error paths
//   6. Stream create / sync / destroy doesn't crash
// and the BLAS entry points: ab_dev_dgemm / ab_dev_zgemm against column-major
// cblas, ab_dev_zgemm on imbalanced real and imaginary parts against Dot2,
// both on the staging pool's reused buffers and on registered, non-page-
// aligned memory, plus the element-wise ops.
//
// Pass criteria: all assertions hold. Exits non-zero on first failure.
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>

#include "apple_bottom.h"
#include "apple_bottom_device.h"
#include "ab_internal.h"
#include "gpu_stage.h"
#include "test_registry.h"

#define N_ELEMS 1024
#define N_BYTES (N_ELEMS * sizeof(double))

static int failures = 0;
static int checks_run = 0;  /* registry counter for ghost-guard (PR-1 §2(b')) */

#define CHECK(cond, msg) do {                                        \
    checks_run++;                                                    \
    if (!(cond)) {                                                   \
        fprintf(stderr, "  ✗ FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
        failures++;                                                  \
    } else {                                                         \
        printf("  ✓ %s\n", msg);                                     \
    }                                                                \
} while (0)

static void test_lifecycle(void) {
    printf("[1] lifecycle\n");
    ab_dev_buffer_t b = ab_dev_malloc(N_BYTES);
    CHECK(b != NULL, "ab_dev_malloc succeeds");
    CHECK(ab_dev_buffer_size(b) == N_BYTES, "buffer_size matches request");
    ab_dev_free(b);
    ab_dev_free(NULL);  // must not crash
    CHECK(1, "ab_dev_free(NULL) safe");
}

static void test_roundtrip(void) {
    printf("[2] host ↔ device roundtrip\n");
    double* src = (double*)malloc(N_BYTES);
    double* dst = (double*)malloc(N_BYTES);
    assert(src && dst);
    for (int i = 0; i < N_ELEMS; i++) src[i] = (double)i * 0.5 - 3.14;

    ab_dev_buffer_t buf = ab_dev_malloc(N_BYTES);
    assert(buf);

    ABStatus s1 = ab_dev_memcpy_h2d(buf, 0, src, N_BYTES);
    CHECK(s1 == AB_OK, "h2d returns AB_OK");

    memset(dst, 0xAB, N_BYTES);
    ABStatus s2 = ab_dev_memcpy_d2h(dst, buf, 0, N_BYTES);
    CHECK(s2 == AB_OK, "d2h returns AB_OK");

    CHECK(memcmp(src, dst, N_BYTES) == 0, "byte-identical roundtrip");

    ab_dev_free(buf);
    free(src);
    free(dst);
}

static void test_d2d_with_offsets(void) {
    printf("[3] device → device with offsets\n");
    double* host = (double*)malloc(N_BYTES);
    double* back = (double*)calloc(N_ELEMS, sizeof(double));
    for (int i = 0; i < N_ELEMS; i++) host[i] = (double)(i + 1);

    ab_dev_buffer_t a = ab_dev_malloc(N_BYTES);
    ab_dev_buffer_t b = ab_dev_malloc(N_BYTES);
    assert(a && b);

    CHECK(ab_dev_memcpy_h2d(a, 0, host, N_BYTES) == AB_OK, "seed a");
    CHECK(ab_dev_memset(b, 0, 0, N_BYTES) == AB_OK, "zero b");

    // Copy second half of a → first half of b.
    size_t half = N_BYTES / 2;
    CHECK(ab_dev_memcpy_d2d(b, 0, a, half, half) == AB_OK, "d2d mid-slice");

    CHECK(ab_dev_memcpy_d2h(back, b, 0, N_BYTES) == AB_OK, "read b back");

    int ok = 1;
    for (int i = 0; i < N_ELEMS/2; i++) {
        if (back[i] != host[i + N_ELEMS/2]) { ok = 0; break; }
    }
    CHECK(ok, "first half of b matches second half of a");

    int zero_ok = 1;
    for (int i = N_ELEMS/2; i < N_ELEMS; i++) {
        if (back[i] != 0.0) { zero_ok = 0; break; }
    }
    CHECK(zero_ok, "second half of b still zero");

    ab_dev_free(a);
    ab_dev_free(b);
    free(host);
    free(back);
}

static void test_memset(void) {
    printf("[4] memset\n");
    ab_dev_buffer_t b = ab_dev_malloc(N_BYTES);
    assert(b);
    CHECK(ab_dev_memset(b, 0, 0x5A, N_BYTES) == AB_OK, "memset ok");

    unsigned char* host = (unsigned char*)malloc(N_BYTES);
    CHECK(ab_dev_memcpy_d2h(host, b, 0, N_BYTES) == AB_OK, "read back");

    int ok = 1;
    for (size_t i = 0; i < N_BYTES; i++) if (host[i] != 0x5A) { ok = 0; break; }
    CHECK(ok, "all bytes == 0x5A");

    ab_dev_free(b);
    free(host);
}

static void test_error_paths(void) {
    printf("[5] error paths\n");
    ab_dev_buffer_t b = ab_dev_malloc(64);
    assert(b);
    double tmp[1] = {0};
    CHECK(ab_dev_memcpy_h2d(NULL, 0, tmp, 8) == AB_ERROR_INVALID_ARG, "h2d NULL dst");
    CHECK(ab_dev_memcpy_h2d(b, 0, NULL, 8) == AB_ERROR_INVALID_ARG, "h2d NULL src");
    CHECK(ab_dev_memcpy_h2d(b, 60, tmp, 8) == AB_ERROR_INVALID_ARG, "h2d overflow");
    CHECK(ab_dev_memcpy_d2h(tmp, b, 60, 8) == AB_ERROR_INVALID_ARG, "d2h overflow");
    CHECK(ab_dev_malloc(0) == NULL, "malloc(0) returns NULL");
    CHECK(ab_dev_buffer_size(NULL) == 0, "size(NULL) == 0");
    ab_dev_free(b);
}

static void test_streams(void) {
    printf("[6] streams\n");
    ab_dev_stream_t s = ab_dev_stream_create();
    CHECK(s != NULL, "stream_create ok");
    CHECK(ab_dev_stream_sync(s) == AB_OK, "stream_sync ok");
    CHECK(ab_dev_stream_sync(NULL) == AB_OK, "default stream sync ok");
    ab_dev_stream_destroy(s);
    CHECK(1, "stream_destroy did not crash");
}

// =============================================================================
// BLAS validation tests
// =============================================================================
//
// ab_dev_dgemm / ab_dev_zgemm take column-major operands
// (apple_bottom_device.h), so the reference is Accelerate cblas_dgemm /
// cblas_zgemm in CblasColMajor (K-P0-2 / H-P0-4). Leading-dimension padding
// and the elements before an operand's offset are NaN, so reading them shows
// up in C. A, B and C outside its m x n block must come back bitwise
// unchanged. With beta == 0 the C block is NaN too: BLAS must not read it.

#include <math.h>
#include <complex.h>
#include <float.h>
#include <Accelerate/Accelerate.h>

#include "test_dot2.h"

#define GEMM_TOL 1e-11

typedef struct {
    ABTranspose ta, tb;
    int m, n, k;
    int pad;                        // ld = stored rows + pad
    size_t off;                     // element offset of A, B, C (0: plain entry point)
    double complex alpha, beta;     // ab_dev_dgemm uses the real parts
} GemmCase;

// A column-major rows x cols block at element `off` with leading dimension
// `ld`, in a buffer of w doubles per element (1 real, 2 complex).
typedef struct { size_t off; int ld, rows, cols, w; } Operand;

static double rand_double(void) {
    return (double)rand() / (double)RAND_MAX * 2.0 - 1.0;
}

static const char* trans_name(ABTranspose t) {
    return t == AB_NO_TRANS ? "N" : t == AB_TRANS ? "T" : "C";
}

static enum CBLAS_TRANSPOSE ref_trans(ABTranspose t, bool cplx) {
    if (t == AB_NO_TRANS) return CblasNoTrans;
    return (t == AB_CONJ_TRANS && cplx) ? CblasConjTrans : CblasTrans;
}

static size_t operand_elems(Operand o) {
    return o.off + (size_t)o.ld * o.cols;
}

static size_t operand_bytes(Operand o) { return operand_elems(o) * o.w * sizeof(double); }

static bool in_block(Operand o, size_t e) {
    if (e < o.off) return false;
    return (e - o.off) % o.ld < (size_t)o.rows && (e - o.off) / o.ld < (size_t)o.cols;
}

// NaN everywhere; uniform in [-1, 1) inside the block unless nan_block.
static double* make_operand(Operand o, bool nan_block) {
    double* X = (double*)malloc(operand_bytes(o));
    assert(X);
    for (size_t e = 0; e < operand_elems(o); e++)
        for (int q = 0; q < o.w; q++)
            X[e * o.w + q] = (!nan_block && in_block(o, e)) ? rand_double() : NAN;
    return X;
}

// max |C - Cref| / max |Cref| over the block; NaN if C has a NaN there.
static double block_rel_err(Operand o, const double* C, const double* Cref) {
    double diff = 0.0, ref = 0.0;
    for (size_t e = 0; e < operand_elems(o); e++) {
        if (!in_block(o, e)) continue;
        double d = 0.0, r = 0.0;
        for (int q = 0; q < o.w; q++) {
            double x = C[e * o.w + q] - Cref[e * o.w + q];
            d += x * x;
            r += Cref[e * o.w + q] * Cref[e * o.w + q];
        }
        if (isnan(d)) return NAN;
        diff = fmax(diff, sqrt(d));
        ref = fmax(ref, sqrt(r));
    }
    return diff / ref;
}

static bool outside_block_unchanged(Operand o, const double* X, const double* X0) {
    for (size_t e = 0; e < operand_elems(o); e++)
        if (!in_block(o, e) && memcmp(X + e * o.w, X0 + e * o.w, o.w * sizeof(double)))
            return false;
    return true;
}

// The stored A, B and C of c, w doubles per element.
static void gemm_operands(GemmCase c, int w, Operand* oA, Operand* oB, Operand* oC) {
    bool tA = (c.ta != AB_NO_TRANS), tB = (c.tb != AB_NO_TRANS);
    int a_rows = tA ? c.k : c.m, b_rows = tB ? c.n : c.k;
    *oA = (Operand){c.off, a_rows + c.pad, a_rows, tA ? c.m : c.k, w};
    *oB = (Operand){c.off, b_rows + c.pad, b_rows, tB ? c.k : c.n, w};
    *oC = (Operand){c.off, c.m + c.pad, c.m, c.n, w};
}

// A case's stored operands, from seed 42: NaN outside each block, and in
// C's for beta = 0, which BLAS must not read.
typedef struct {
    bool cplx;
    GemmCase c;
    Operand oA, oB, oC;
    double *A, *B, *C0;
} GemmData;

static void gemm_data_init(GemmData* g, bool cplx, GemmCase c) {
    *g = (GemmData){.cplx = cplx, .c = c};
    gemm_operands(c, cplx ? 2 : 1, &g->oA, &g->oB, &g->oC);
    srand(42);
    g->A  = make_operand(g->oA, false);
    g->B  = make_operand(g->oB, false);
    g->C0 = make_operand(g->oC, c.beta == 0.0);
}

static void gemm_data_free(GemmData* g) {
    free(g->A); free(g->B); free(g->C0);
}

// g's call on buffers dA, dB and dC, by ab_dev_zgemm (complex) or
// ab_dev_dgemm, their _offset forms for c.off != 0.
static ABStatus gemm_data_call(const GemmData* g, ab_dev_buffer_t dA, ab_dev_buffer_t dB,
                               ab_dev_buffer_t dC) {
    const GemmCase c = g->c;
    const int lda = g->oA.ld, ldb = g->oB.ld, ldc = g->oC.ld;
    if (g->cplx) {
        double alpha[2] = {creal(c.alpha), cimag(c.alpha)};
        double beta[2]  = {creal(c.beta), cimag(c.beta)};
        return c.off ? ab_dev_zgemm_offset(c.ta, c.tb, c.m, c.n, c.k, alpha,
                                           dA, c.off, lda, dB, c.off, ldb, beta, dC, c.off, ldc)
                     : ab_dev_zgemm(c.ta, c.tb, c.m, c.n, c.k, alpha,
                                    dA, lda, dB, ldb, beta, dC, ldc);
    }
    return c.off ? ab_dev_dgemm_offset(c.ta, c.tb, c.m, c.n, c.k, creal(c.alpha),
                                       dA, c.off, lda, dB, c.off, ldb,
                                       creal(c.beta), dC, c.off, ldc)
                 : ab_dev_dgemm(c.ta, c.tb, c.m, c.n, c.k, creal(c.alpha),
                                dA, lda, dB, ldb, creal(c.beta), dC, ldc);
}

// g's call on device copies of A (every double a NaN if nan_a), B and C0; C,
// A and B as the call leaves them go into C, A1 and B1 where not NULL.
static ABStatus gemm_data_run(const GemmData* g, bool nan_a, double* C, double* A1, double* B1) {
    size_t bA = operand_bytes(g->oA), bB = operand_bytes(g->oB), bC = operand_bytes(g->oC);
    ab_dev_buffer_t dA = ab_dev_malloc(bA), dB = ab_dev_malloc(bB), dC = ab_dev_malloc(bC);
    assert(dA && dB && dC);
    ab_dev_memcpy_h2d(dA, 0, g->A, bA);
    if (nan_a) ab_dev_memset(dA, 0, 0xFF, bA);
    ab_dev_memcpy_h2d(dB, 0, g->B, bB);
    ab_dev_memcpy_h2d(dC, 0, g->C0, bC);

    ABStatus s = gemm_data_call(g, dA, dB, dC);
    if (C)  ab_dev_memcpy_d2h(C, dC, 0, bC);
    if (A1) ab_dev_memcpy_d2h(A1, dA, 0, bA);
    if (B1) ab_dev_memcpy_d2h(B1, dB, 0, bB);
    ab_dev_free(dA); ab_dev_free(dB); ab_dev_free(dC);
    return s;
}

// One ab_dev_dgemm / ab_dev_zgemm call against cblas ColMajor (3 CHECKs).
static void check_gemm(bool cplx, GemmCase c) {
    GemmData g;
    gemm_data_init(&g, cplx, c);
    const Operand oA = g.oA, oB = g.oB, oC = g.oC;
    size_t bytesA = operand_bytes(oA), bytesB = operand_bytes(oB), bytesC = operand_bytes(oC);
    double* Cref = (double*)malloc(bytesC);
    double* C    = (double*)malloc(bytesC);
    double* A1   = (double*)malloc(bytesA);
    double* B1   = (double*)malloc(bytesB);
    assert(Cref && C && A1 && B1);
    memcpy(Cref, g.C0, bytesC);

    if (cplx)
        cblas_zgemm(CblasColMajor, ref_trans(c.ta, true), ref_trans(c.tb, true),
                    c.m, c.n, c.k, &c.alpha,
                    (double complex*)g.A + c.off, oA.ld,
                    (double complex*)g.B + c.off, oB.ld,
                    &c.beta, (double complex*)Cref + c.off, oC.ld);
    else
        cblas_dgemm(CblasColMajor, ref_trans(c.ta, false), ref_trans(c.tb, false),
                    c.m, c.n, c.k, creal(c.alpha), g.A + c.off, oA.ld,
                    g.B + c.off, oB.ld, creal(c.beta), Cref + c.off, oC.ld);
    ABStatus s = gemm_data_run(&g, false, C, A1, B1);

    char label[96], msg[192];
    snprintf(label, sizeof(label), "%s %s%s %dx%dx%d pad %d off %zu%s",
             cplx ? "zgemm" : "dgemm", trans_name(c.ta), trans_name(c.tb),
             c.m, c.n, c.k, c.pad, c.off, c.beta == 0.0 ? " beta 0" : "");
    snprintf(msg, sizeof(msg), "%s: returns AB_OK", label);
    CHECK(s == AB_OK, msg);

    double err = block_rel_err(oC, C, Cref);
    snprintf(msg, sizeof(msg), "%s: rel err %.2e vs cblas ColMajor (tol %.0e)",
             label, err, GEMM_TOL);
    CHECK(err <= GEMM_TOL, msg);  // false for NaN

    snprintf(msg, sizeof(msg), "%s: A, B and C outside its block unchanged", label);
    CHECK(!memcmp(A1, g.A, bytesA) && !memcmp(B1, g.B, bytesB) &&
          outside_block_unchanged(oC, C, g.C0), msg);

    gemm_data_free(&g);
    free(Cref); free(C); free(A1); free(B1);
}

static const ABTranspose ALL_TRANS[] = {AB_NO_TRANS, AB_TRANS, AB_CONJ_TRANS};

// Every flag pair with m, n, k distinct, ld > rows and nonzero offsets
// (ab_dev_dgemm treats AB_CONJ_TRANS as a plain transpose).
static void check_all_trans(bool cplx, double complex alpha, double complex beta) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            check_gemm(cplx, (GemmCase){ALL_TRANS[i], ALL_TRANS[j], 96, 80, 72, 3, 5,
                                        alpha, beta});
}

// Square 64^3: the row-major reading returned B*A here, and B*A again
// with the transA flag ignored.
static void test_dgemm_square(void) {
    printf("[GEMM-1] DGEMM square vs cblas ColMajor\n");
    check_gemm(false, (GemmCase){AB_NO_TRANS, AB_NO_TRANS, 64, 64, 64, 0, 0, 1.0, 0.0});
    check_gemm(false, (GemmCase){AB_TRANS,    AB_NO_TRANS, 64, 64, 64, 0, 0, 1.0, 0.0});
}

static void test_dgemm_all_trans(void) {
    printf("[GEMM-2] DGEMM all trans pairs vs cblas ColMajor\n");
    check_all_trans(false, 1.25, -0.75);
}

// Tall-skinny either way round (the kernel runs on C^T, so m >= 4n and
// n >= 4m select different kernels), then a larger padded case.
static void test_dgemm_shapes(void) {
    printf("[GEMM-3] DGEMM tall-skinny and large shapes vs cblas ColMajor\n");
    check_gemm(false, (GemmCase){AB_NO_TRANS, AB_TRANS,    300, 40, 56, 2, 0, 1.25, -0.75});
    check_gemm(false, (GemmCase){AB_TRANS,    AB_NO_TRANS, 40, 300, 56, 2, 7, 1.0, 0.0});
    check_gemm(false, (GemmCase){AB_TRANS,    AB_NO_TRANS, 1024, 768, 520, 8, 3, 1.0, 1.0});
}

static void test_zgemm_square(void) {
    printf("[GEMM-4] ZGEMM square vs cblas ColMajor\n");
    check_gemm(true, (GemmCase){AB_NO_TRANS,   AB_NO_TRANS, 64, 64, 64, 0, 0, 1.0, 0.0});
    check_gemm(true, (GemmCase){AB_CONJ_TRANS, AB_NO_TRANS, 64, 64, 64, 0, 0, 1.0, 0.0});
}

// General complex alpha/beta: the host-side epilogue path.
static void test_zgemm_all_trans(void) {
    printf("[GEMM-5] ZGEMM all trans pairs, complex alpha/beta, vs cblas ColMajor\n");
    check_all_trans(true, CMPLX(1.25, -0.5), CMPLX(-0.75, 0.25));
}

// alpha = 1, beta = 0: the direct path, with no host epilogue.
static void test_zgemm_all_trans_unit(void) {
    printf("[GEMM-6] ZGEMM all trans pairs, alpha=1 beta=0, vs cblas ColMajor\n");
    check_all_trans(true, 1.0, 0.0);
}

// The QE cegterg 'C','N' pattern (m, n << k), tall-skinny either way, and a
// C of 2^20 elements read for beta != 0, which is staged in chunks across
// cores.
static void test_zgemm_shapes(void) {
    printf("[GEMM-7] ZGEMM QE-like, tall-skinny and large-C shapes vs cblas ColMajor\n");
    check_gemm(true, (GemmCase){AB_CONJ_TRANS, AB_NO_TRANS, 150, 150, 2000, 4, 0,
                                1.0, 0.0});
    check_gemm(true, (GemmCase){AB_NO_TRANS, AB_CONJ_TRANS, 300, 40, 56, 2, 9,
                                CMPLX(0.5, 1.0), 0.0});
    check_gemm(true, (GemmCase){AB_TRANS, AB_TRANS, 40, 300, 56, 2, 0,
                                CMPLX(1.25, -0.5), 1.0});
    check_gemm(true, (GemmCase){AB_NO_TRANS, AB_TRANS, 1024, 1024, 48, 1, 3,
                                CMPLX(0.5, -0.25), CMPLX(-0.75, 0.25)});
}

// Invalid flags, short leading dimensions and operands that run past their
// buffer are rejected before anything is read or written.
static void test_gemm_arg_errors(void) {
    printf("[GEMM-8] GEMM argument checks\n");
    enum { M = 8, N = 6, K = 4 };
    // Sized for complex operands, so twice what the real calls need.
    size_t bytesC = (size_t)M * N * 2 * sizeof(double);
    ab_dev_buffer_t dA = ab_dev_malloc((size_t)M * K * 2 * sizeof(double));
    ab_dev_buffer_t dB = ab_dev_malloc((size_t)K * N * 2 * sizeof(double));
    ab_dev_buffer_t dC = ab_dev_malloc(bytesC);
    assert(dA && dB && dC);
    ab_dev_memset(dC, 0, 0x5A, bytesC);
    const ABTranspose bad = (ABTranspose)7;
    double one[2] = {1.0, 0.0}, zero[2] = {0.0, 0.0};

    CHECK(ab_dev_dgemm(bad, AB_NO_TRANS, M, N, K, 1.0, dA, M, dB, K, 0.0, dC, M)
          == AB_ERROR_INVALID_ARG, "dgemm rejects an invalid transA");
    CHECK(ab_dev_dgemm(AB_NO_TRANS, AB_NO_TRANS, M, N, K, 1.0, dA, M - 1, dB, K, 0.0, dC, M)
          == AB_ERROR_INVALID_ARG, "dgemm rejects lda < m");
    CHECK(ab_dev_dgemm(AB_TRANS, AB_NO_TRANS, M, N, K, 1.0, dA, K - 1, dB, K, 0.0, dC, M)
          == AB_ERROR_INVALID_ARG, "dgemm rejects lda < k with transA");
    CHECK(ab_dev_dgemm_offset(AB_NO_TRANS, AB_NO_TRANS, M, N, K, 1.0, dA, 0, M,
                              dB, 0, K, 0.0, dC, (size_t)M * N + 1, M)
          == AB_ERROR_INVALID_ARG, "dgemm rejects C running past its buffer");
    CHECK(ab_dev_zgemm(AB_NO_TRANS, bad, M, N, K, one, dA, M, dB, K, zero, dC, M)
          == AB_ERROR_INVALID_ARG, "zgemm rejects an invalid transB");
    CHECK(ab_dev_zgemm(AB_NO_TRANS, AB_NO_TRANS, M, N, K, one, dA, M, dB, K, zero, dC, M - 1)
          == AB_ERROR_INVALID_ARG, "zgemm rejects ldc < m");
    CHECK(ab_dev_zgemm(AB_NO_TRANS, AB_NO_TRANS, M, N, K, one, dA, M, dB, K + 1, zero, dC, M)
          == AB_ERROR_INVALID_ARG, "zgemm rejects B running past its buffer");

    unsigned char* host = (unsigned char*)malloc(bytesC);
    assert(host);
    ab_dev_memcpy_d2h(host, dC, 0, bytesC);
    int ok = 1;
    for (size_t i = 0; i < bytesC; i++) if (host[i] != 0x5A) { ok = 0; break; }
    CHECK(ok, "rejected calls leave C untouched");

    free(host);
    ab_dev_free(dA); ab_dev_free(dB); ab_dev_free(dC);
}

// On balanced data the device path adds nothing but the layout change.
// Column-major A, B, C with ld = rows are row-major A^T, B^T, C^T, so
// ab_dev_dgemm NN must be bitwise ab_dgemm_scaled(B^T, A^T) and ab_dev_zgemm
// 'C','N' (alpha = 1, beta = 0) bitwise ab_zgemm_ex('N', 'C', B^T, A^T) on
// those handles (both Gauss-3M).
static void test_gemm_matches_host_api(void) {
    printf("[GEMM-9] device GEMM bitwise equal to the host API on C^T\n");
    enum { M = 96, N = 80, K = 72 };
    const double alpha = 1.25, beta = -0.75;
    // Bytes of a real A, B, C; the complex operands are twice that.
    size_t bA = (size_t)M * K * sizeof(double), bB = (size_t)K * N * sizeof(double);
    size_t bC = (size_t)M * N * sizeof(double);
    double* A  = (double*)malloc(2 * bA);
    double* B  = (double*)malloc(2 * bB);
    double* C  = (double*)malloc(bC);
    double* Cd = (double*)malloc(2 * bC);
    double* Ch = (double*)malloc(2 * bC);
    assert(A && B && C && Cd && Ch);
    srand(7);
    for (int i = 0; i < 2 * M * K; i++) A[i] = rand_double();
    for (int i = 0; i < 2 * K * N; i++) B[i] = rand_double();
    for (int i = 0; i < M * N; i++) C[i] = rand_double();

    // DGEMM NN, general alpha/beta: A, B are the first M*K, K*N doubles.
    ab_dev_buffer_t dA = ab_dev_malloc(bA), dB = ab_dev_malloc(bB), dC = ab_dev_malloc(bC);
    assert(dA && dB && dC);
    ab_dev_memcpy_h2d(dA, 0, A, bA);
    ab_dev_memcpy_h2d(dB, 0, B, bB);
    ab_dev_memcpy_h2d(dC, 0, C, bC);
    ABStatus sd = ab_dev_dgemm(AB_NO_TRANS, AB_NO_TRANS, M, N, K, alpha,
                               dA, M, dB, K, beta, dC, M);
    ab_dev_memcpy_d2h(Cd, dC, 0, bC);
    ab_dev_free(dA); ab_dev_free(dB); ab_dev_free(dC);

    ABMatrix hA = ab_matrix_create(K, M), hB = ab_matrix_create(N, K);
    ABMatrix hC = ab_matrix_create(N, M);
    assert(hA && hB && hC);
    ab_matrix_upload(hA, A, true);
    ab_matrix_upload(hB, B, true);
    ab_matrix_upload(hC, C, true);
    ABStatus sh = ab_dgemm_scaled(alpha, hB, hA, beta, hC);
    ab_matrix_download(hC, Ch, true);
    ab_matrix_destroy(hA); ab_matrix_destroy(hB); ab_matrix_destroy(hC);
    CHECK(sd == AB_OK && sh == AB_OK && !memcmp(Cd, Ch, bC),
          "dgemm NN 96x80x72 bitwise ab_dgemm_scaled(B^T, A^T)");

    // ZGEMM 'C','N' (A stored K x M), alpha = 1, beta = 0: the QE pattern.
    double one[2] = {1.0, 0.0}, zero[2] = {0.0, 0.0};
    dA = ab_dev_malloc(2 * bA); dB = ab_dev_malloc(2 * bB); dC = ab_dev_malloc(2 * bC);
    assert(dA && dB && dC);
    ab_dev_memcpy_h2d(dA, 0, A, 2 * bA);
    ab_dev_memcpy_h2d(dB, 0, B, 2 * bB);
    sd = ab_dev_zgemm(AB_CONJ_TRANS, AB_NO_TRANS, M, N, K, one,
                      dA, K, dB, K, zero, dC, M);
    ab_dev_memcpy_d2h(Cd, dC, 0, 2 * bC);
    ab_dev_free(dA); ab_dev_free(dB); ab_dev_free(dC);

    // Real/imag planes for A, B and C; M*N is the largest of the three.
    _Static_assert(K <= M && K <= N, "planes are sized M*N");
    double* re = (double*)malloc(bC);
    double* im = (double*)malloc(bC);
    assert(re && im);
    ABMatrix hAr = ab_matrix_create(M, K), hAi = ab_matrix_create(M, K);
    ABMatrix hBr = ab_matrix_create(N, K), hBi = ab_matrix_create(N, K);
    ABMatrix hCr = ab_matrix_create(N, M), hCi = ab_matrix_create(N, M);
    assert(hAr && hAi && hBr && hBi && hCr && hCi);
    for (int i = 0; i < K * M; i++) { re[i] = A[2 * i]; im[i] = A[2 * i + 1]; }
    ab_matrix_upload(hAr, re, true);
    ab_matrix_upload(hAi, im, true);
    for (int i = 0; i < K * N; i++) { re[i] = B[2 * i]; im[i] = B[2 * i + 1]; }
    ab_matrix_upload(hBr, re, true);
    ab_matrix_upload(hBi, im, true);
    sh = ab_zgemm_ex(AB_NO_TRANS, AB_CONJ_TRANS, hBr, hBi, hAr, hAi, hCr, hCi);
    ab_matrix_download(hCr, re, true);
    ab_matrix_download(hCi, im, true);
    for (int i = 0; i < M * N; i++) { Ch[2 * i] = re[i]; Ch[2 * i + 1] = im[i]; }
    ab_matrix_destroy(hAr); ab_matrix_destroy(hAi);
    ab_matrix_destroy(hBr); ab_matrix_destroy(hBi);
    ab_matrix_destroy(hCr); ab_matrix_destroy(hCi);
    CHECK(sd == AB_OK && sh == AB_OK && !memcmp(Cd, Ch, 2 * bC),
          "zgemm CN 96x80x72 bitwise ab_zgemm_ex(N, C, B^T, A^T)");

    free(re); free(im);
    free(A); free(B); free(C); free(Cd); free(Ch);
}

// Gauss-3M's imaginary part under Re/Im imbalance (audit K-P1-2): the
// ImagData check of test_blas_abi's test_zgemm_imbalance (test_dot2.h), on
// ab_dev_zgemm_offset. IMAG_LINES needs the row- and column-wise rule: A's
// and B's largest parts are balanced.

// The complex element op(X)_ij of a column-major X at o.
static size_t op_elem(Operand o, ABTranspose t, int i, int j) {
    return o.off + (t == AB_NO_TRANS ? (size_t)j * o.ld + i : (size_t)i * o.ld + j);
}

// op(X) (rows x cols) into row-major real and imaginary parts.
static void op_split(const double* X, Operand o, ABTranspose t, int rows, int cols,
                     double* re, double* im) {
    const double sign = t == AB_CONJ_TRANS ? -1.0 : 1.0;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            size_t e = op_elem(o, t, i, j);
            re[(size_t)i * cols + j] = X[2 * e];
            im[(size_t)i * cols + j] = sign * X[2 * e + 1];
        }
}

// One ab_dev_zgemm_offset call on d with beta = 0 and a real alpha: 1 takes
// the direct path, 2 (exact in the host epilogue) the general one (2 CHECKs).
static void check_imbalance(GemmCase c, ImagData d) {
    const int M = c.m, N = c.n, K = c.k;
    const double alpha = creal(c.alpha);
    Operand oA, oB, oC;
    gemm_operands(c, 2, &oA, &oB, &oC);
    size_t bytesA = operand_bytes(oA), bytesB = operand_bytes(oB), bytesC = operand_bytes(oC);

    srand(11);
    double* A = make_operand(oA, false);
    double* B = make_operand(oB, false);
    double* C = make_operand(oC, true);         // beta = 0: C must not be read
    for (size_t e = 0; e < operand_elems(oA) * 2; e++) A[e] = half_to_one(A[e]);  // NaN stays
    for (size_t e = 0; e < operand_elems(oB) * 2; e++) B[e] = half_to_one(B[e]);
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            A[2 * op_elem(oA, c.ta, i, k) + !imag_small_re(d, i)] *= IMAG_RATIO;
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++)
            B[2 * op_elem(oB, c.tb, k, j) + !imag_small_re(d, j)] *= IMAG_RATIO;

    ab_dev_buffer_t dA = ab_dev_malloc(bytesA);
    ab_dev_buffer_t dB = ab_dev_malloc(bytesB);
    ab_dev_buffer_t dC = ab_dev_malloc(bytesC);
    assert(dA && dB && dC);
    ab_dev_memcpy_h2d(dA, 0, A, bytesA);
    ab_dev_memcpy_h2d(dB, 0, B, bytesB);
    ab_dev_memcpy_h2d(dC, 0, C, bytesC);
    double al[2] = {alpha, 0.0}, zero[2] = {0.0, 0.0};
    ABStatus s = ab_dev_zgemm_offset(c.ta, c.tb, M, N, K, al, dA, c.off, oA.ld, dB, c.off,
                                     oB.ld, zero, dC, c.off, oC.ld);
    ab_dev_memcpy_d2h(C, dC, 0, bytesC);
    ab_dev_free(dA); ab_dev_free(dB); ab_dev_free(dC);

    double *Ar = malloc((size_t)M * K * sizeof *Ar), *Ai = malloc((size_t)M * K * sizeof *Ai);
    double *Br = malloc((size_t)K * N * sizeof *Br), *Bi = malloc((size_t)K * N * sizeof *Bi);
    assert(Ar && Ai && Br && Bi);
    op_split(A, oA, c.ta, M, K, Ar, Ai);
    op_split(B, oB, c.tb, K, N, Br, Bi);
    double err[2];
    dot2_zgemm_parts_err(M, N, K, Ar, Ai, Br, Bi, alpha, C + 2 * oC.off, (size_t)oC.ld, err);

    char label[96], msg[192];
    snprintf(label, sizeof(label), "zgemm %s%s %dx%dx%d imag-%s alpha %g", trans_name(c.ta),
             trans_name(c.tb), M, N, K, imag_tag(d), alpha);
    snprintf(msg, sizeof(msg), "%s: returns AB_OK", label);
    CHECK(s == AB_OK, msg);
    snprintf(msg, sizeof(msg), "%s: Re err %.2f u², Im err %.2f u² (bound %.0f u²)", label,
             err[0] / U2, err[1] / U2, IMAG_BOUND);
    CHECK(err[0] <= IMAG_BOUND * U2 && err[1] <= IMAG_BOUND * U2, msg);  // false for NaN

    free(Ar); free(Ai); free(Br); free(Bi);
    free(A); free(B); free(C);
}

// Nearly real and nearly imaginary data, and lines of both kinds: op(A)'s
// rows are A's rows ('N') or columns, op(B)'s columns B's columns ('N') or
// rows, so the four flag pairs of IMAG_LINES cover each way of gathering a
// line's parts; then the same for operands of 2^20 elements, which are
// staged in chunks across cores.
static void test_zgemm_imbalance(void) {
    printf("[GEMM-10] ZGEMM Re/Im imbalance, per row and column, vs Dot2\n");
    static const struct { ABTranspose ta, tb; ImagData d; double alpha; } IMB[] = {
        {AB_NO_TRANS,   AB_NO_TRANS,   IMAG_SMALL, 1.0},
        {AB_CONJ_TRANS, AB_NO_TRANS,   IMAG_SMALL, 2.0},
        {AB_NO_TRANS,   AB_TRANS,      IMAG_LARGE, 1.0},
        {AB_TRANS,      AB_CONJ_TRANS, IMAG_LARGE, 2.0},
        {AB_NO_TRANS,   AB_NO_TRANS,   IMAG_LINES, 1.0},
        {AB_CONJ_TRANS, AB_NO_TRANS,   IMAG_LINES, 1.0},
        {AB_NO_TRANS,   AB_TRANS,      IMAG_LINES, 1.0},
        {AB_TRANS,      AB_CONJ_TRANS, IMAG_LINES, 1.0},
        {AB_CONJ_TRANS, AB_NO_TRANS,   IMAG_LINES, 2.0},
        {AB_NO_TRANS,   AB_TRANS,      IMAG_LINES, 2.0},
    };
    for (size_t i = 0; i < sizeof IMB / sizeof IMB[0]; i++)
        check_imbalance((GemmCase){IMB[i].ta, IMB[i].tb, 96, 80, 72, 3, 5, IMB[i].alpha, 0.0},
                        IMB[i].d);
    // A, then A and B (the QE 'C','N' pattern), then B of 2^20 elements: lines
    // that are A's rows, A's and B's columns, B's rows.
    check_imbalance((GemmCase){AB_NO_TRANS, AB_NO_TRANS, 8192, 32, 128, 3, 5, 1.0, 0.0},
                    IMAG_LINES);
    check_imbalance((GemmCase){AB_CONJ_TRANS, AB_NO_TRANS, 128, 128, 8192, 3, 5, 1.0, 0.0},
                    IMAG_LINES);
    check_imbalance((GemmCase){AB_CONJ_TRANS, AB_TRANS, 32, 8192, 128, 3, 5, 1.0, 0.0},
                    IMAG_LINES);
}

// The GEMMs take their ABMatrix handles from the staged path's pool
// (gpu_stage.h), whose buffers hold what their last call left. A call
// repeated must leave the pool as it was, and each call must give the bits
// of its first run again after the pool served NaN data (which a beta = 0
// call must not read), larger and smaller shapes, and from threads calling
// at once.
enum { POOL_CALLS = 4, POOL_THREADS = 4 };

typedef struct { GemmData g; double* first; } PoolCall;    // a call and its first C

// p's call again: its first bits?
static bool pool_call_same(const PoolCall* p) {
    size_t bytes = operand_bytes(p->g.oC);
    double* C = (double*)malloc(bytes);
    assert(C);
    bool same = gemm_data_run(&p->g, false, C, NULL, NULL) == AB_OK &&
                !memcmp(C, p->first, bytes);
    free(C);
    return same;
}

typedef struct { const PoolCall* calls; int differ; } PoolThread;

static void* pool_thread(void* arg) {
    PoolThread* t = (PoolThread*)arg;
    for (int r = 0; r < 2; r++)
        for (int i = 0; i < POOL_CALLS; i++) t->differ += !pool_call_same(&t->calls[i]);
    return NULL;
}

static void test_gemm_pool(void) {
    printf("[GEMM-11] device GEMM handles from the staging pool\n");
    // QE's patterns: zgemm 'C','N' and dgemm 'T','N' with beta = 0, and
    // 'N','N' with general alpha and beta.
    PoolCall p[POOL_CALLS];
    gemm_data_init(&p[0].g, true, (GemmCase){AB_CONJ_TRANS, AB_NO_TRANS, 150, 150, 2000, 4, 0,
                                             1.0, 0.0});
    gemm_data_init(&p[1].g, true, (GemmCase){AB_NO_TRANS, AB_NO_TRANS, 2000, 150, 150, 3, 5,
                                             CMPLX(1.25, -0.5), CMPLX(-0.75, 0.25)});
    gemm_data_init(&p[2].g, false, (GemmCase){AB_TRANS, AB_NO_TRANS, 150, 150, 4000, 2, 0,
                                              1.0, 0.0});
    gemm_data_init(&p[3].g, false, (GemmCase){AB_NO_TRANS, AB_NO_TRANS, 4000, 150, 150, 1, 3,
                                              1.25, -0.75});
    for (int i = 0; i < POOL_CALLS; i++) {
        p[i].first = (double*)malloc(operand_bytes(p[i].g.oC));
        assert(p[i].first);
    }

    int bad = gemm_data_run(&p[0].g, false, p[0].first, NULL, NULL) != AB_OK;
    size_t pool = ab_internal_stage_pool_bytes();
    CHECK(!bad && pool > 0, "zgemm CN 150x150x2000: the pool holds buffers after it");
    CHECK(pool_call_same(&p[0]) && ab_internal_stage_pool_bytes() == pool,
          "zgemm CN repeated: its first bits, the pool as it was");

    // A call repeated gets the same buffers in the same roles, so C^T's holds
    // the NaN result of the call before.
    bad = 0;
    for (int i = 1; i < POOL_CALLS; i++)
        bad += gemm_data_run(&p[i].g, false, p[i].first, NULL, NULL) != AB_OK;
    for (int i = 0; i < POOL_CALLS; i += 2)
        bad += gemm_data_run(&p[i].g, true, NULL, NULL, NULL) != AB_OK || !pool_call_same(&p[i]);
    CHECK(!bad, "zgemm CN and dgemm TN (beta 0) right after NaN data: their first bits");

    // A larger and a smaller shape, each against cblas
    check_gemm(true, (GemmCase){AB_NO_TRANS, AB_CONJ_TRANS, 1024, 512, 768, 0, 0, 1.0, 0.0});
    check_gemm(false, (GemmCase){AB_NO_TRANS, AB_TRANS, 40, 30, 50, 1, 0, 1.0, 1.0});
    bad = 0;
    for (int i = 0; i < POOL_CALLS; i++) bad += !pool_call_same(&p[i]);
    CHECK(!bad, "zgemm CN, NN and dgemm TN, NN after larger and smaller shapes: "
                "their first bits");

    PoolThread t[POOL_THREADS];
    pthread_t th[POOL_THREADS];
    for (int i = 0; i < POOL_THREADS; i++) {
        t[i] = (PoolThread){p, 0};
        pthread_create(&th[i], NULL, pool_thread, &t[i]);
    }
    bad = 0;
    for (int i = 0; i < POOL_THREADS; i++) {
        pthread_join(th[i], NULL);
        bad += t[i].differ;
    }
    CHECK(!bad, "4 threads at once, each call twice: their first bits");

    for (int i = 0; i < POOL_CALLS; i++) {
        gemm_data_free(&p[i].g);
        free(p[i].first);
    }
}

// ---- GEMM on registered memory ----------------------------------------------
//
// ab_dev_register() wraps the pages around the caller's memory and keeps the
// distance to it as the buffer's base, which every host read or write of the
// buffer must add (dev_data() in src/device_api.m). Each operand here starts
// `misalign` bytes past a page, with whole pages of sentinels on both sides, so
// that a pointer missing the base lands in a sentinel or leaves C wrong. The
// result must be bitwise that of the same call on ab_dev_malloc() buffers,
// which check_gemm holds to cblas.

#define REG_PAD 4096                // sentinel doubles on each side: 2 pages of 16 KB
#define REG_SENTINEL (-123.456)

// bytes of x at `misalign` bytes past a page, between sentinels; *raw is to free().
static double* reg_block(const double* x, size_t bytes, size_t misalign, void** raw) {
    size_t n = bytes / sizeof(double);
    int rc = posix_memalign(raw, 16384, (n + 2 * REG_PAD) * sizeof(double) + misalign);
    assert(rc == 0);
    (void)rc;
    double* p = (double*)((char*)*raw + misalign);
    for (size_t i = 0; i < n + 2 * REG_PAD; i++) p[i] = REG_SENTINEL;
    memcpy(p + REG_PAD, x, bytes);
    return p + REG_PAD;
}

static bool reg_sentinels_intact(const double* x, size_t bytes) {
    size_t n = bytes / sizeof(double);
    for (size_t i = 0; i < REG_PAD; i++)
        if (x[n + i] != REG_SENTINEL || (x - REG_PAD)[i] != REG_SENTINEL) return false;
    return true;
}

// Whether g's call on registered memory gives the bits of the call on allocated
// buffers, with A, B and every sentinel left alone.
static bool reg_call_same(const GemmData* g, size_t misalign) {
    size_t bA = operand_bytes(g->oA), bB = operand_bytes(g->oB), bC = operand_bytes(g->oC);
    double* ref = (double*)malloc(bC);
    assert(ref);
    bool ok = gemm_data_run(g, false, ref, NULL, NULL) == AB_OK;
    void *ra, *rb, *rc;
    double* A = reg_block(g->A, bA, misalign, &ra);
    double* B = reg_block(g->B, bB, misalign, &rb);
    double* C = reg_block(g->C0, bC, misalign, &rc);
    ab_dev_buffer_t dA = ab_dev_register(A, bA), dB = ab_dev_register(B, bB),
                    dC = ab_dev_register(C, bC);
    ok = ok && dA && dB && dC && gemm_data_call(g, dA, dB, dC) == AB_OK;
    ab_dev_free(dA); ab_dev_free(dB); ab_dev_free(dC);
    ok = ok && !memcmp(C, ref, bC) && !memcmp(A, g->A, bA) && !memcmp(B, g->B, bB) &&
         reg_sentinels_intact(A, bA) && reg_sentinels_intact(B, bB) && reg_sentinels_intact(C, bC);
    free(ra); free(rb); free(rc); free(ref);
    return ok;
}

static void test_gemm_registered(void) {
    printf("[GEMM-12] device GEMM on registered, non-page-aligned memory\n");
    static const size_t MISALIGN[] = {8, 24, 4088, 12344};
    static const struct { const char* name; double complex alpha, beta; } SC[] = {
        {"alpha 1, beta 0", 1.0, 0.0},
        {"real alpha and beta", 0.75, -2.0},
        {"complex alpha and beta", CMPLX(0.3, -1.1), CMPLX(0.25, 0.5)},
    };
    int calls = 0;
    for (int z = 0; z < 2; z++)
        for (int s = 0; s < 2 + z; s++) {           // a real GEMM has no complex scalars
            int bad = 0, n = 0;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    for (size_t off = 0; off <= 13; off += 13) {
                        GemmData g;
                        gemm_data_init(&g, z, (GemmCase){ALL_TRANS[i], ALL_TRANS[j], 97, 65, 129, 7,
                                                         off, SC[s].alpha, SC[s].beta});
                        bad += !reg_call_same(&g, MISALIGN[calls++ % 4]);
                        n++;
                        gemm_data_free(&g);
                    }
            char msg[128];
            snprintf(msg, sizeof msg, "%s, %s: %d calls, every trans pair, with and without offsets",
                     z ? "zgemm" : "dgemm", SC[s].name, n);
            CHECK(!bad, msg);
        }
    // A tall-skinny C, whose kernel and epilogue the shape above does not reach
    for (int z = 0; z < 2; z++) {
        GemmData g;
        gemm_data_init(&g, z, (GemmCase){AB_NO_TRANS, AB_NO_TRANS, 1024, 40, 4608, 7, 13,
                                         SC[1 + z].alpha, SC[1 + z].beta});
        CHECK(reg_call_same(&g, MISALIGN[z]), z ? "zgemm 1024x40x4608" : "dgemm 1024x40x4608");
        gemm_data_free(&g);
    }
}

// The native scale of a matrix holding values below 2^-1042, whose FP64 high
// word is 0. They count at their own exponents when the whole span, theirs
// included, is within 2^78 (exponents at most 78 apart), so that the matrix
// is centred; a wider span keeps the scale the other values give it. The
// 2x2x2 data is the ULP Fiction counterexample (A = [[0, t], [2^-997, 0]],
// for which C_00 came back 2.0e5 times the bound with AB_OK) and variants of
// it; each C entry has at most one product, checked against ULP Fiction's
// bound for native ab_dgemm, (3K + 19) u^2 |A||B| + 2^-53 |C| + 2^-1074,
// by Dot2.

enum { TS = 2 };                                    // M = N = K
#define TINY_T (177571.0 * 0x1p-1074)               // 0x1.5ad18p-1057: high word 0
#define TS_BOUND ((3 * TS + 19) * U2)
static const double TS_B[TS * TS] = {0x1p1023, 0.0, 0x1p945 * 1.7382651127463331, 0.0};

// max over C of |C - A B| / ((3K + 19) u^2 |A||B| + 2^-53 |C| + 2^-1074),
// row-major TS x TS, A B by Dot2.
static double tiny_span_ratio(const double* A, const double* B, const double* C) {
    double worst = 0.0;
    for (int i = 0; i < TS; i++)
        for (int j = 0; j < TS; j++) {
            double s = 0.0, e = 0.0, ab = 0.0;
            for (int k = 0; k < TS; k++) {
                dot2_step(&s, &e, A[i * TS + k], B[k * TS + j]);
                ab += fabs(A[i * TS + k] * B[k * TS + j]);
            }
            const double c = C[i * TS + j];
            worst = fmax(worst, fabs((c - s) - e) / (TS_BOUND * ab + 0x1p-53 * fabs(c) + 0x1p-1074));
        }
    return worst;
}

// Whether Re(C) and Im(C), interleaved column-major, are each within GEMM_TOL
// of A B (split complex, row-major TS x TS), in their own scales.
static bool tiny_span_complex_ok(const double* Ar, const double* Ai, const double* Br,
                                 const double* Bi, const double* C) {
    double err[2];
    dot2_zgemm_parts_err(TS, TS, TS, Ar, Ai, Br, Bi, 1.0, C, TS, err);
    return err[0] <= GEMM_TOL && err[1] <= GEMM_TOL;
}

// C = A B through handles (row-major A, B, C); *ea receives A's scale.
static ABStatus tiny_span_dgemm(const double* A, const double* B, double* C, int* ea) {
    ABMatrix mA = ab_matrix_create(TS, TS), mB = ab_matrix_create(TS, TS);
    ABMatrix mC = ab_matrix_create(TS, TS);
    assert(mA && mB && mC);
    ABStatus s = ab_matrix_upload(mA, A, false);
    if (s == AB_OK) s = ab_matrix_upload(mB, B, false);
    if (s == AB_OK) s = ab_dgemm(mA, mB, mC);
    if (s == AB_OK) s = ab_matrix_download(mC, C, false);
    *ea = ab_internal_matrix_scale(mA);
    ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
    return s;
}

// TS x TS X transposed into Xt, w doubles per element: row-major to
// column-major and back.
static void tiny_span_transpose(const double* X, double* Xt, int w) {
    for (int i = 0; i < TS; i++)
        for (int j = 0; j < TS; j++)
            for (int p = 0; p < w; p++) Xt[(j * TS + i) * w + p] = X[(i * TS + j) * w + p];
}

static void test_native_tiny_span(void) {
    printf("[GEMM-13] native scale of values below 2^-1042, handles and device buffers\n");
    const double t = TINY_T;
    const double cex[TS * TS] = {0.0, t, 0x1p-997, 0.0};        // span 2^59.6: centred
    const double under[TS * TS] = {0.0, t, 0x1.8p-979, 0.0};    // exponents 78 apart: centred
    const double over[TS * TS] = {0.0, t, 0x1.8p-978, 0.0};     // 79 apart: the others' scale
    double C[TS * TS], Cref[TS * TS];
    int ea = 0;

    ABStatus s = tiny_span_dgemm(cex, TS_B, C, &ea);
    CHECK(s == AB_OK && ea == -1024 &&
          tiny_span_ratio(cex, TS_B, C) <= 1.0,
          "ab_dgemm on the counterexample: A centred (scale -1024), C within the bound");
    s = tiny_span_dgemm(under, TS_B, C, &ea);
    CHECK(s == AB_OK && ea == -1018 &&
          tiny_span_ratio(under, TS_B, C) <= 1.0,
          "ab_dgemm, A's exponents 78 apart: centred (scale -1018), C within the bound");

    // 79 apart: the scale 2^-978 alone gives (-976), and the bits of A
    // stored at it with scale 0 (ab_internal_upload_scaled), times 2^-976
    s = tiny_span_dgemm(over, TS_B, C, &ea);
    {
        ABMatrix mA = ab_matrix_create(TS, TS), mB = ab_matrix_create(TS, TS);
        ABMatrix mC = ab_matrix_create(TS, TS);
        assert(mA && mB && mC);
        ABStatus r = ab_internal_upload_scaled(mA, over, NULL, NULL, -976, NULL);
        if (r == AB_OK) r = ab_matrix_upload(mB, TS_B, false);
        if (r == AB_OK) r = ab_dgemm(mA, mB, mC);
        if (r == AB_OK) r = ab_matrix_download(mC, Cref, false);
        for (int i = 0; i < TS * TS; i++) Cref[i] = ldexp(Cref[i], -976);
        ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
        CHECK(s == AB_OK && r == AB_OK && ea == -976 && !memcmp(C, Cref, sizeof C),
              "ab_dgemm, A's exponents 79 apart: the others' scale (-976) and its bits");
    }

    // Only values below 2^-1042: centred on their own exponents
    {
        const double only[TS * TS] = {5 * 0x1p-1074, 0.0, 0.0, 5 * 0x1p-1074};
        const double B[TS * TS] = {0x1p1000, 0.0, 0.0, 0x1p1000};
        s = tiny_span_dgemm(only, B, C, &ea);
        CHECK(s == AB_OK && ea == -1072 &&
              tiny_span_ratio(only, B, C) <= 1.0,
              "ab_dgemm, A only of values below 2^-1042: their own centre (scale -1072)");
    }

    // Ordinary values with one stray value below 2^-1042: scale 0 and the
    // bits of the same A with that value 0 (stored as 0 either way); B at
    // 2^10, so that a scale taking A to FP32's top would overflow
    {
        enum { S = 8 };
        double A[S * S], A0[S * S], B[S * S], Cs[S * S], C0[S * S];
        srand(13);
        for (int i = 0; i < S * S; i++) {
            A[i] = A0[i] = rand_double();
            B[i] = 0x1p10 * rand_double();
        }
        A[3] = t;
        A0[3] = 0.0;
        ABMatrix mA = ab_matrix_create(S, S), mA0 = ab_matrix_create(S, S);
        ABMatrix mB = ab_matrix_create(S, S), mC = ab_matrix_create(S, S);
        assert(mA && mA0 && mB && mC);
        s = ab_matrix_upload(mA, A, false);
        if (s == AB_OK) s = ab_matrix_upload(mA0, A0, false);
        if (s == AB_OK) s = ab_matrix_upload(mB, B, false);
        if (s == AB_OK) s = ab_dgemm(mA, mB, mC);
        if (s == AB_OK) s = ab_matrix_download(mC, Cs, false);
        if (s == AB_OK) s = ab_dgemm(mA0, mB, mC);
        if (s == AB_OK) s = ab_matrix_download(mC, C0, false);
        CHECK(s == AB_OK && ab_internal_matrix_scale(mA) == 0 && !memcmp(Cs, C0, sizeof Cs),
              "ab_dgemm, ordinary A with one value below 2^-1042: scale 0, the bits without it");
        ab_matrix_destroy(mA); ab_matrix_destroy(mA0); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
    }

    // The device API's operands go through handles. A row-major array is the
    // column-major transpose: NN on (B^T, A^T) gives C^T, row-major C, and
    // converts A^T by rows; TT on (A^T, B^T) converts them by columns.
    {
        double Cc[TS * TS];
        const size_t b = sizeof Cc;
        ab_dev_buffer_t dA = ab_dev_malloc(b), dB = ab_dev_malloc(b), dC = ab_dev_malloc(b);
        assert(dA && dB && dC);
        ab_dev_memcpy_h2d(dA, 0, cex, b);
        ab_dev_memcpy_h2d(dB, 0, TS_B, b);
        s = ab_dev_dgemm(AB_NO_TRANS, AB_NO_TRANS, TS, TS, TS, 1.0, dB, TS, dA, TS, 0.0, dC, TS);
        ab_dev_memcpy_d2h(C, dC, 0, b);
        CHECK(s == AB_OK && tiny_span_ratio(cex, TS_B, C) <= 1.0,
              "ab_dev_dgemm NN on the counterexample: C within the bound");
        s = ab_dev_dgemm(AB_TRANS, AB_TRANS, TS, TS, TS, 1.0, dA, TS, dB, TS, 0.0, dC, TS);
        ab_dev_memcpy_d2h(Cc, dC, 0, b);
        tiny_span_transpose(Cc, C, 1);
        CHECK(s == AB_OK && tiny_span_ratio(cex, TS_B, C) <= 1.0,
              "ab_dev_dgemm TT on the counterexample: C within the bound");
        ab_dev_free(dA); ab_dev_free(dB); ab_dev_free(dC);
    }

    // Complex: Ai = 3/4 Ar and Bi = 1/2 Br, each part the counterexample's
    // pattern (both A parts centred at -1024); through handles, then device
    // buffers with alpha = 1, beta = 0. Last, a regression check of the
    // epilogue's alpha and beta on these operands (alpha = 2; beta = 1 on
    // C = that result), which must give exactly twice it. It passes without
    // the fix: no C can tell the epilogue's two scales apart (see
    // ab_internal_zgemm_epilogue).
    {
        double Ai[TS * TS], Bi[TS * TS], Cr[TS * TS], Ci[TS * TS];
        for (int i = 0; i < TS * TS; i++) {
            Ai[i] = 0.75 * cex[i];
            Bi[i] = 0.5 * TS_B[i];
        }
        ABMatrix h[6];
        for (int i = 0; i < 6; i++) {
            h[i] = ab_matrix_create(TS, TS);
            assert(h[i]);
        }
        s = ab_matrix_upload(h[0], cex, false);
        if (s == AB_OK) s = ab_matrix_upload(h[1], Ai, false);
        if (s == AB_OK) s = ab_matrix_upload(h[2], TS_B, false);
        if (s == AB_OK) s = ab_matrix_upload(h[3], Bi, false);
        if (s == AB_OK) s = ab_zgemm(h[0], h[1], h[2], h[3], h[4], h[5]);
        if (s == AB_OK) s = ab_matrix_download(h[4], Cr, false);
        if (s == AB_OK) s = ab_matrix_download(h[5], Ci, false);
        double Cz[2 * TS * TS];                                 // interleaved, column-major
        for (int i = 0; i < TS; i++)
            for (int j = 0; j < TS; j++) {
                Cz[2 * (j * TS + i)] = Cr[i * TS + j];
                Cz[2 * (j * TS + i) + 1] = Ci[i * TS + j];
            }
        CHECK(s == AB_OK && ab_internal_matrix_scale(h[0]) == -1024 &&
              ab_internal_matrix_scale(h[1]) == -1024 &&
              tiny_span_complex_ok(cex, Ai, TS_B, Bi, Cz),
              "ab_zgemm on the complex counterexample: A's parts centred, C within tolerance");
        for (int i = 0; i < 6; i++) ab_matrix_destroy(h[i]);

        double A2[2 * TS * TS], B2[2 * TS * TS], Az[2 * TS * TS], Bz[2 * TS * TS];
        double C1[2 * TS * TS], C2[2 * TS * TS];
        for (int i = 0; i < TS * TS; i++) {
            A2[2 * i] = cex[i];
            A2[2 * i + 1] = Ai[i];
            B2[2 * i] = TS_B[i];
            B2[2 * i + 1] = Bi[i];
        }
        tiny_span_transpose(A2, Az, 2);
        tiny_span_transpose(B2, Bz, 2);
        const size_t b = sizeof Az;
        ab_dev_buffer_t dA = ab_dev_malloc(b), dB = ab_dev_malloc(b), dC = ab_dev_malloc(b);
        assert(dA && dB && dC);
        ab_dev_memcpy_h2d(dA, 0, Az, b);
        ab_dev_memcpy_h2d(dB, 0, Bz, b);
        const double one[2] = {1.0, 0.0}, two[2] = {2.0, 0.0}, zero[2] = {0.0, 0.0};
        s = ab_dev_zgemm(AB_NO_TRANS, AB_NO_TRANS, TS, TS, TS, one, dA, TS, dB, TS, zero, dC, TS);
        ab_dev_memcpy_d2h(C1, dC, 0, b);
        CHECK(s == AB_OK && tiny_span_complex_ok(cex, Ai, TS_B, Bi, C1),
              "ab_dev_zgemm NN on the complex counterexample: C within tolerance");
        ABStatus s2 =
            ab_dev_zgemm(AB_NO_TRANS, AB_NO_TRANS, TS, TS, TS, two, dA, TS, dB, TS, zero, dC, TS);
        ab_dev_memcpy_d2h(C2, dC, 0, b);
        bool doubled = true;
        for (int i = 0; i < 2 * TS * TS; i++) doubled = doubled && C2[i] == 2.0 * C1[i];
        ab_dev_memcpy_h2d(dC, 0, C1, b);
        ABStatus s3 =
            ab_dev_zgemm(AB_NO_TRANS, AB_NO_TRANS, TS, TS, TS, one, dA, TS, dB, TS, one, dC, TS);
        ab_dev_memcpy_d2h(C2, dC, 0, b);
        for (int i = 0; i < 2 * TS * TS; i++) doubled = doubled && C2[i] == 2.0 * C1[i];
        CHECK(s2 == AB_OK && s3 == AB_OK && doubled,
              "ab_dev_zgemm through the epilogue (alpha = 2; beta = 1 on C = A B): exactly 2 A B");
        ab_dev_free(dA); ab_dev_free(dB); ab_dev_free(dC);
    }
}

// Test element-wise operations
static void test_elementwise(void) {
    printf("[BLAS-3] element-wise ops\n");

    // conjg test
    double cplx[] = {1.0, 2.0, 3.0, -4.0};
    ab_dev_buffer_t b = ab_dev_malloc(sizeof(cplx));
    assert(b);
    ab_dev_memcpy_h2d(b, 0, cplx, sizeof(cplx));
    CHECK(ab_dev_conjg_c(b, 0, 2) == AB_OK, "conjg_c ok");
    double out[4];
    ab_dev_memcpy_d2h(out, b, 0, sizeof(out));
    CHECK(out[0] == 1.0 && out[1] == -2.0 && out[2] == 3.0 && out[3] == 4.0,
          "conjg: imag parts negated");
    ab_dev_free(b);

    // scale test
    double vals[] = {2.0, 1.0};  // 2+1i
    b = ab_dev_malloc(sizeof(vals));
    ab_dev_memcpy_h2d(b, 0, vals, sizeof(vals));
    double scale[2] = {0.0, 1.0};  // multiply by i: (2+i)*i = -1+2i
    CHECK(ab_dev_scale_z(b, 0, 1, scale) == AB_OK, "scale_z ok");
    ab_dev_memcpy_d2h(out, b, 0, 2 * sizeof(double));
    CHECK(fabs(out[0] - (-1.0)) < 1e-15 && fabs(out[1] - 2.0) < 1e-15,
          "scale: (2+i)*i = -1+2i");
    ab_dev_free(b);

    // axpy test: y = alpha*x + y
    double x[] = {1.0, 0.0};  // 1+0i
    double y[] = {0.0, 1.0};  // 0+1i
    ab_dev_buffer_t dx = ab_dev_malloc(sizeof(x));
    ab_dev_buffer_t dy = ab_dev_malloc(sizeof(y));
    ab_dev_memcpy_h2d(dx, 0, x, sizeof(x));
    ab_dev_memcpy_h2d(dy, 0, y, sizeof(y));
    double alpha_ax[2] = {2.0, 0.0};  // alpha=2: y = 2*(1+0i) + (0+1i) = 2+1i
    CHECK(ab_dev_axpy_z(1, alpha_ax, dx, 0, 1, dy, 0, 1) == AB_OK, "axpy_z ok");
    ab_dev_memcpy_d2h(out, dy, 0, 2 * sizeof(double));
    CHECK(fabs(out[0] - 2.0) < 1e-15 && fabs(out[1] - 1.0) < 1e-15,
          "axpy: 2*(1+0i)+(0+1i) = 2+1i");
    ab_dev_free(dx); ab_dev_free(dy);
}

static const TestCase TESTS[] = {
    {"test_lifecycle",                test_lifecycle},
    {"test_roundtrip",                test_roundtrip},
    {"test_d2d_with_offsets",         test_d2d_with_offsets},
    {"test_memset",                   test_memset},
    {"test_error_paths",              test_error_paths},
    {"test_streams",                  test_streams},
    {"test_dgemm_square",             test_dgemm_square},
    {"test_dgemm_all_trans",          test_dgemm_all_trans},
    {"test_dgemm_shapes",             test_dgemm_shapes},
    {"test_zgemm_square",             test_zgemm_square},
    {"test_zgemm_all_trans",          test_zgemm_all_trans},
    {"test_zgemm_all_trans_unit",     test_zgemm_all_trans_unit},
    {"test_zgemm_shapes",             test_zgemm_shapes},
    {"test_gemm_arg_errors",          test_gemm_arg_errors},
    {"test_gemm_matches_host_api",    test_gemm_matches_host_api},
    {"test_zgemm_imbalance",          test_zgemm_imbalance},
    {"test_gemm_pool",                test_gemm_pool},
    {"test_gemm_registered",          test_gemm_registered},
    {"test_native_tiny_span",         test_native_tiny_span},
    {"test_elementwise",              test_elementwise},
};
static const int N_TESTS = (int)TEST_REGISTRY_SIZE(TESTS);

int main(void) {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("apple-bottom device-API tests\n");
    printf("═══════════════════════════════════════════════════════════\n");

    if (ab_init() != AB_OK) {
        fprintf(stderr, "ab_init failed — no Metal device?\n");
        return 2;
    }

    for (int i = 0; i < N_TESTS; i++) {
        printf("[%d/%d] %s ... ", i + 1, N_TESTS, TESTS[i].name);
        fflush(stdout);
        int pre = checks_run;
        TESTS[i].fn();
        int post = checks_run;
        if (post == pre) {
            fprintf(stderr,
                "\nFATAL: ghost test — '%s' ran zero CHECK invocations\n",
                TESTS[i].name);
            abort();
        }
    }

    ab_shutdown();

    printf("═══════════════════════════════════════════════════════════\n");
    printf("Summary: %d test(s) ran, %d CHECK(s), %d failure(s)\n",
           N_TESTS, checks_run, failures);
    if (failures == 0) {
        printf("✓ All device-API tests passed\n");
        return 0;
    } else {
        printf("✗ %d device-API test(s) failed\n", failures);
        return 1;
    }
}
