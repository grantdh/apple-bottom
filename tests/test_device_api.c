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
// cblas, plus the element-wise ops.
//
// Pass criteria: all assertions hold. Exits non-zero on first failure.
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "apple_bottom.h"
#include "apple_bottom_device.h"
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
#include <Accelerate/Accelerate.h>

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

static bool in_block(Operand o, size_t e) {
    if (e < o.off) return false;
    return (e - o.off) % o.ld < (size_t)o.rows && (e - o.off) / o.ld < (size_t)o.cols;
}

// NaN everywhere; uniform in [-1, 1) inside the block unless nan_block.
static double* make_operand(Operand o, bool nan_block) {
    double* X = (double*)malloc(operand_elems(o) * o.w * sizeof(double));
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

// One ab_dev_dgemm / ab_dev_zgemm call against cblas ColMajor (3 CHECKs).
static void check_gemm(bool cplx, GemmCase c) {
    bool tA = (c.ta != AB_NO_TRANS), tB = (c.tb != AB_NO_TRANS);
    int w = cplx ? 2 : 1;
    int a_rows = tA ? c.k : c.m, a_cols = tA ? c.m : c.k;
    int b_rows = tB ? c.n : c.k, b_cols = tB ? c.k : c.n;
    Operand oA = {c.off, a_rows + c.pad, a_rows, a_cols, w};
    Operand oB = {c.off, b_rows + c.pad, b_rows, b_cols, w};
    Operand oC = {c.off, c.m + c.pad, c.m, c.n, w};
    size_t bytesA = operand_elems(oA) * w * sizeof(double);
    size_t bytesB = operand_elems(oB) * w * sizeof(double);
    size_t bytesC = operand_elems(oC) * w * sizeof(double);

    srand(42);
    double* A    = make_operand(oA, false);
    double* B    = make_operand(oB, false);
    double* C0   = make_operand(oC, c.beta == 0.0);
    double* Cref = (double*)malloc(bytesC);
    double* C    = (double*)malloc(bytesC);
    double* A1   = (double*)malloc(bytesA);
    double* B1   = (double*)malloc(bytesB);
    assert(Cref && C && A1 && B1);
    memcpy(Cref, C0, bytesC);

    ab_dev_buffer_t dA = ab_dev_malloc(bytesA);
    ab_dev_buffer_t dB = ab_dev_malloc(bytesB);
    ab_dev_buffer_t dC = ab_dev_malloc(bytesC);
    assert(dA && dB && dC);
    ab_dev_memcpy_h2d(dA, 0, A, bytesA);
    ab_dev_memcpy_h2d(dB, 0, B, bytesB);
    ab_dev_memcpy_h2d(dC, 0, C0, bytesC);

    ABStatus s;
    if (cplx) {
        cblas_zgemm(CblasColMajor, ref_trans(c.ta, true), ref_trans(c.tb, true),
                    c.m, c.n, c.k, &c.alpha,
                    (double complex*)A + c.off, oA.ld,
                    (double complex*)B + c.off, oB.ld,
                    &c.beta, (double complex*)Cref + c.off, oC.ld);
        double alpha[2] = {creal(c.alpha), cimag(c.alpha)};
        double beta[2]  = {creal(c.beta), cimag(c.beta)};
        s = c.off ? ab_dev_zgemm_offset(c.ta, c.tb, c.m, c.n, c.k, alpha,
                                        dA, c.off, oA.ld, dB, c.off, oB.ld,
                                        beta, dC, c.off, oC.ld)
                  : ab_dev_zgemm(c.ta, c.tb, c.m, c.n, c.k, alpha,
                                 dA, oA.ld, dB, oB.ld, beta, dC, oC.ld);
    } else {
        cblas_dgemm(CblasColMajor, ref_trans(c.ta, false), ref_trans(c.tb, false),
                    c.m, c.n, c.k, creal(c.alpha), A + c.off, oA.ld,
                    B + c.off, oB.ld, creal(c.beta), Cref + c.off, oC.ld);
        s = c.off ? ab_dev_dgemm_offset(c.ta, c.tb, c.m, c.n, c.k, creal(c.alpha),
                                        dA, c.off, oA.ld, dB, c.off, oB.ld,
                                        creal(c.beta), dC, c.off, oC.ld)
                  : ab_dev_dgemm(c.ta, c.tb, c.m, c.n, c.k, creal(c.alpha),
                                 dA, oA.ld, dB, oB.ld, creal(c.beta), dC, oC.ld);
    }
    ab_dev_memcpy_d2h(C, dC, 0, bytesC);
    ab_dev_memcpy_d2h(A1, dA, 0, bytesA);
    ab_dev_memcpy_d2h(B1, dB, 0, bytesB);

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
    CHECK(!memcmp(A1, A, bytesA) && !memcmp(B1, B, bytesB) &&
          outside_block_unchanged(oC, C, C0), msg);

    ab_dev_free(dA); ab_dev_free(dB); ab_dev_free(dC);
    free(A); free(B); free(C0); free(Cref); free(C); free(A1); free(B1);
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

// alpha = 1, beta = 0: the direct ab_zgemm_ex path.
static void test_zgemm_all_trans_unit(void) {
    printf("[GEMM-6] ZGEMM all trans pairs, alpha=1 beta=0, vs cblas ColMajor\n");
    check_all_trans(true, 1.0, 0.0);
}

// The QE cegterg 'C','N' pattern (m, n << k) plus tall-skinny either way.
static void test_zgemm_shapes(void) {
    printf("[GEMM-7] ZGEMM QE-like and tall-skinny shapes vs cblas ColMajor\n");
    check_gemm(true, (GemmCase){AB_CONJ_TRANS, AB_NO_TRANS, 150, 150, 2000, 4, 0,
                                1.0, 0.0});
    check_gemm(true, (GemmCase){AB_NO_TRANS, AB_CONJ_TRANS, 300, 40, 56, 2, 9,
                                CMPLX(0.5, 1.0), 0.0});
    check_gemm(true, (GemmCase){AB_TRANS, AB_TRANS, 40, 300, 56, 2, 0,
                                CMPLX(1.25, -0.5), 1.0});
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

// The device path adds nothing but the layout change. Column-major A, B, C
// with ld = rows are row-major A^T, B^T, C^T, so ab_dev_dgemm NN must be
// bitwise ab_dgemm_scaled(B^T, A^T) and ab_dev_zgemm 'C','N' (alpha = 1,
// beta = 0) bitwise ab_zgemm_ex('N', 'C', B^T, A^T) on those handles.
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
