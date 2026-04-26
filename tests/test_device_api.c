// =============================================================================
// test_device_api.c — Smoke tests for the device-buffer layer
// =============================================================================
//
// Exercises the Week-1 surface:
//   1. alloc → size → free
//   2. H2D → D2H roundtrip, byte-identical
//   3. D2D copy with source/destination offsets
//   4. memset then D2H
//   5. NULL / out-of-range error paths
//   6. Stream create / sync / destroy doesn't crash
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
// Week-2 BLAS validation tests
// =============================================================================

#include <math.h>

static double rand_double(void) {
    return (double)rand() / (double)RAND_MAX * 2.0 - 1.0;
}

// Compute Frobenius norm of difference between two arrays
static double frobenius_diff(const double* a, const double* b, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sqrt(sum);
}

// Test: ab_dev_dgemm via device-buffer path matches ab_dgemm_scaled via
// matrix-handle path.  Both paths do FP64→DD conversion, so results should
// be bit-identical (Frobenius error == 0).
static void test_dgemm_bit_identical(int N) {
    printf("[BLAS-1] DGEMM bit-identical N=%d\n", N);
    size_t nn = (size_t)N * N;

    double* A_data = (double*)malloc(nn * sizeof(double));
    double* B_data = (double*)malloc(nn * sizeof(double));
    double* C_ref  = (double*)malloc(nn * sizeof(double));
    double* C_dev  = (double*)malloc(nn * sizeof(double));
    assert(A_data && B_data && C_ref && C_dev);

    srand(42);
    for (size_t i = 0; i < nn; i++) { A_data[i] = rand_double(); B_data[i] = rand_double(); }

    // Path 1: matrix-handle API
    ABMatrix mA = ab_matrix_create(N, N);
    ABMatrix mB = ab_matrix_create(N, N);
    ABMatrix mC = ab_matrix_create(N, N);
    assert(mA && mB && mC);
    ab_matrix_upload(mA, A_data, true);
    ab_matrix_upload(mB, B_data, true);
    ab_dgemm_scaled(1.0, mA, mB, 0.0, mC);
    ab_matrix_download(mC, C_ref, true);
    ab_matrix_destroy(mA);
    ab_matrix_destroy(mB);
    ab_matrix_destroy(mC);

    // Path 2: device-buffer API
    ab_dev_buffer_t dA = ab_dev_malloc(nn * sizeof(double));
    ab_dev_buffer_t dB = ab_dev_malloc(nn * sizeof(double));
    ab_dev_buffer_t dC = ab_dev_malloc(nn * sizeof(double));
    assert(dA && dB && dC);
    ab_dev_memcpy_h2d(dA, 0, A_data, nn * sizeof(double));
    ab_dev_memcpy_h2d(dB, 0, B_data, nn * sizeof(double));
    ab_dev_memset(dC, 0, 0, nn * sizeof(double));

    ABStatus s = ab_dev_dgemm(AB_NO_TRANS, AB_NO_TRANS, N, N, N,
                              1.0, dA, N, dB, N, 0.0, dC, N);
    CHECK(s == AB_OK, "ab_dev_dgemm returns AB_OK");

    ab_dev_memcpy_d2h(C_dev, dC, 0, nn * sizeof(double));

    double err = frobenius_diff(C_ref, C_dev, nn);
    char msg[128];
    snprintf(msg, sizeof(msg), "DGEMM N=%d Frobenius diff = %.2e (expect 0)", N, err);
    CHECK(err == 0.0, msg);

    ab_dev_free(dA);
    ab_dev_free(dB);
    ab_dev_free(dC);
    free(A_data); free(B_data); free(C_ref); free(C_dev);
}

// Test: ab_dev_zgemm via device-buffer path matches ab_zgemm via matrix-handle
// path.  Complex data in the device buffer is interleaved [r,i,r,i,...].
static void test_zgemm_bit_identical(int N) {
    printf("[BLAS-2] ZGEMM bit-identical N=%d\n", N);
    size_t nn = (size_t)N * N;

    // Interleaved complex: 2 doubles per element
    double* A_cplx = (double*)malloc(nn * 2 * sizeof(double));
    double* B_cplx = (double*)malloc(nn * 2 * sizeof(double));
    double* C_dev  = (double*)malloc(nn * 2 * sizeof(double));

    // Separate real/imag for matrix-handle path
    double* Ar = (double*)malloc(nn * sizeof(double));
    double* Ai = (double*)malloc(nn * sizeof(double));
    double* Br = (double*)malloc(nn * sizeof(double));
    double* Bi = (double*)malloc(nn * sizeof(double));
    double* Cr_ref = (double*)malloc(nn * sizeof(double));
    double* Ci_ref = (double*)malloc(nn * sizeof(double));
    assert(A_cplx && B_cplx && C_dev && Ar && Ai && Br && Bi && Cr_ref && Ci_ref);

    srand(42);
    for (size_t i = 0; i < nn; i++) {
        Ar[i] = rand_double(); Ai[i] = rand_double();
        Br[i] = rand_double(); Bi[i] = rand_double();
        A_cplx[i * 2] = Ar[i]; A_cplx[i * 2 + 1] = Ai[i];
        B_cplx[i * 2] = Br[i]; B_cplx[i * 2 + 1] = Bi[i];
    }

    // Path 1: matrix-handle API (separate real/imag)
    ABMatrix mAr = ab_matrix_create(N, N);
    ABMatrix mAi = ab_matrix_create(N, N);
    ABMatrix mBr = ab_matrix_create(N, N);
    ABMatrix mBi = ab_matrix_create(N, N);
    ABMatrix mCr = ab_matrix_create(N, N);
    ABMatrix mCi = ab_matrix_create(N, N);
    assert(mAr && mAi && mBr && mBi && mCr && mCi);

    ab_matrix_upload(mAr, Ar, true);
    ab_matrix_upload(mAi, Ai, true);
    ab_matrix_upload(mBr, Br, true);
    ab_matrix_upload(mBi, Bi, true);
    ab_zgemm(mAr, mAi, mBr, mBi, mCr, mCi);
    ab_matrix_download(mCr, Cr_ref, true);
    ab_matrix_download(mCi, Ci_ref, true);
    ab_matrix_destroy(mAr); ab_matrix_destroy(mAi);
    ab_matrix_destroy(mBr); ab_matrix_destroy(mBi);
    ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);

    // Path 2: device-buffer API (interleaved complex)
    ab_dev_buffer_t dA = ab_dev_malloc(nn * 2 * sizeof(double));
    ab_dev_buffer_t dB = ab_dev_malloc(nn * 2 * sizeof(double));
    ab_dev_buffer_t dC = ab_dev_malloc(nn * 2 * sizeof(double));
    assert(dA && dB && dC);
    ab_dev_memcpy_h2d(dA, 0, A_cplx, nn * 2 * sizeof(double));
    ab_dev_memcpy_h2d(dB, 0, B_cplx, nn * 2 * sizeof(double));
    ab_dev_memset(dC, 0, 0, nn * 2 * sizeof(double));

    double alpha[2] = {1.0, 0.0};
    double beta_z[2] = {0.0, 0.0};
    ABStatus s = ab_dev_zgemm(AB_NO_TRANS, AB_NO_TRANS, N, N, N,
                              alpha, dA, N, dB, N, beta_z, dC, N);
    CHECK(s == AB_OK, "ab_dev_zgemm returns AB_OK");

    ab_dev_memcpy_d2h(C_dev, dC, 0, nn * 2 * sizeof(double));

    // Compare: deinterleave C_dev and compare with Cr_ref, Ci_ref
    double err_r = 0, err_i = 0;
    for (size_t i = 0; i < nn; i++) {
        double dr = C_dev[i * 2]     - Cr_ref[i];
        double di = C_dev[i * 2 + 1] - Ci_ref[i];
        err_r += dr * dr;
        err_i += di * di;
    }
    err_r = sqrt(err_r);
    err_i = sqrt(err_i);
    double err = sqrt(err_r * err_r + err_i * err_i);

    char msg[128];
    snprintf(msg, sizeof(msg), "ZGEMM N=%d Frobenius diff = %.2e (expect 0)", N, err);
    CHECK(err == 0.0, msg);

    ab_dev_free(dA); ab_dev_free(dB); ab_dev_free(dC);
    free(A_cplx); free(B_cplx); free(C_dev);
    free(Ar); free(Ai); free(Br); free(Bi); free(Cr_ref); free(Ci_ref);
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

/* PR-1 §2(b'') wrappers — give parameterized helpers void(void) signatures
 * so each pre-migration invocation in main() corresponds to one TestCase
 * registry entry. Original helper bodies are unchanged. Naming convention:
 * append the argument value as a suffix. */
static void test_dgemm_bit_identical_1024(void) { test_dgemm_bit_identical(1024); }
static void test_dgemm_bit_identical_768(void)  { test_dgemm_bit_identical(768); }
static void test_zgemm_bit_identical_1024(void) { test_zgemm_bit_identical(1024); }
static void test_zgemm_bit_identical_768(void)  { test_zgemm_bit_identical(768); }

static const TestCase TESTS[] = {
    {"test_lifecycle",                test_lifecycle},
    {"test_roundtrip",                test_roundtrip},
    {"test_d2d_with_offsets",         test_d2d_with_offsets},
    {"test_memset",                   test_memset},
    {"test_error_paths",              test_error_paths},
    {"test_streams",                  test_streams},
    {"test_dgemm_bit_identical_1024", test_dgemm_bit_identical_1024},
    {"test_dgemm_bit_identical_768",  test_dgemm_bit_identical_768},
    {"test_zgemm_bit_identical_1024", test_zgemm_bit_identical_1024},
    {"test_zgemm_bit_identical_768",  test_zgemm_bit_identical_768},
    {"test_elementwise",              test_elementwise},
};
static const int N_TESTS = (int)TEST_REGISTRY_SIZE(TESTS);

int main(void) {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("apple-bottom device-API tests (Week-1 + Week-2 BLAS)\n");
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
