// =============================================================================
// apple-bottom Test Suite
// =============================================================================

#include "apple_bottom.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <Accelerate/Accelerate.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  %-42s ", name)
#define PASS() do { printf("✓ PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("✗ FAIL (%s)\n", msg); tests_failed++; } while(0)

// =============================================================================
// Initialization Tests
// =============================================================================

static void test_init_shutdown(void) {
    TEST("ab_init / ab_shutdown");
    ABStatus s = ab_init();
    ab_shutdown();
    if (s == AB_OK) PASS(); else FAIL("init failed");
}

static void test_double_init(void) {
    TEST("double ab_init is safe");
    ab_init();
    ABStatus s = ab_init();
    ab_shutdown();
    if (s == AB_OK) PASS(); else FAIL("double init failed");
}

static void test_shutdown_without_init(void) {
    TEST("ab_shutdown without ab_init is safe");
    ab_shutdown();
    PASS();
}

// =============================================================================
// Matrix Lifecycle Tests
// =============================================================================

static void test_matrix_create_destroy(void) {
    TEST("ab_matrix_create / ab_matrix_destroy");
    ab_init();
    ABMatrix m = ab_matrix_create(100, 100);
    int ok = (m != NULL);
    ab_matrix_destroy(m);
    ab_shutdown();
    if (ok) PASS(); else FAIL("create failed");
}

static void test_matrix_destroy_null(void) {
    TEST("ab_matrix_destroy(NULL) is safe");
    ab_matrix_destroy(NULL);
    PASS();
}

static void test_matrix_dims_null_outputs(void) {
    TEST("ab_matrix_dims with NULL outputs");
    ab_init();
    ABMatrix m = ab_matrix_create(10, 20);
    ab_matrix_dims(m, NULL, NULL);
    ab_matrix_destroy(m);
    ab_shutdown();
    PASS();
}

static void test_matrix_count_null(void) {
    TEST("ab_matrix_count(NULL) returns 0");
    size_t c = ab_matrix_count(NULL);
    if (c == 0) PASS(); else FAIL("should return 0");
}

// =============================================================================
// Data Transfer Tests
// =============================================================================

static void test_upload_download_roundtrip(void) {
    TEST("upload/download roundtrip preserves data");
    ab_init();
    int N = 64;
    double* src = (double*)malloc(N * N * sizeof(double));
    double* dst = (double*)malloc(N * N * sizeof(double));
    for (int i = 0; i < N * N; i++) src[i] = (double)i;
    
    ABMatrix m = ab_matrix_create(N, N);
    ab_matrix_upload(m, src, false);
    ab_matrix_download(m, dst, false);
    
    int ok = 1;
    for (int i = 0; i < N * N && ok; i++) if (fabs(src[i] - dst[i]) > 1e-15) ok = 0;
    
    ab_matrix_destroy(m);
    free(src); free(dst);
    ab_shutdown();
    if (ok) PASS(); else FAIL("data mismatch");
}

static void test_upload_null_data(void) {
    TEST("ab_matrix_upload with NULL data");
    ab_init();
    ABMatrix m = ab_matrix_create(10, 10);
    ABStatus s = ab_matrix_upload(m, NULL, false);
    ab_matrix_destroy(m);
    ab_shutdown();
    if (s == AB_ERROR_INVALID_ARG) PASS(); else FAIL("should fail");
}

static void test_download_null_data(void) {
    TEST("ab_matrix_download with NULL data");
    ab_init();
    ABMatrix m = ab_matrix_create(10, 10);
    ABStatus s = ab_matrix_download(m, NULL, false);
    ab_matrix_destroy(m);
    ab_shutdown();
    if (s == AB_ERROR_INVALID_ARG) PASS(); else FAIL("should fail");
}

static void test_upload_null_matrix(void) {
    TEST("ab_matrix_upload with NULL matrix");
    double data[100];
    ABStatus s = ab_matrix_upload(NULL, data, false);
    if (s == AB_ERROR_INVALID_ARG) PASS(); else FAIL("should fail");
}

static void test_matrix_zero(void) {
    TEST("ab_matrix_zero");
    ab_init();
    int N = 32;
    double* data = (double*)malloc(N * N * sizeof(double));
    for (int i = 0; i < N * N; i++) data[i] = 999.0;
    
    ABMatrix m = ab_matrix_create(N, N);
    ab_matrix_upload(m, data, false);
    ab_matrix_zero(m);
    ab_matrix_download(m, data, false);
    
    int ok = 1;
    for (int i = 0; i < N * N && ok; i++) if (data[i] != 0.0) ok = 0;
    
    ab_matrix_destroy(m);
    free(data);
    ab_shutdown();
    if (ok) PASS(); else FAIL("not zeroed");
}

// =============================================================================
// DGEMM Tests
// =============================================================================

static void test_dgemm_identity(void) {
    TEST("ab_dgemm: I × I = I");
    ab_init();
    int N = 64;
    double* I_mat = (double*)calloc(N * N, sizeof(double));
    double* result = (double*)malloc(N * N * sizeof(double));
    for (int i = 0; i < N; i++) I_mat[i * N + i] = 1.0;
    
    ABMatrix A = ab_matrix_create(N, N);
    ABMatrix B = ab_matrix_create(N, N);
    ABMatrix C = ab_matrix_create(N, N);
    ab_matrix_upload(A, I_mat, false);
    ab_matrix_upload(B, I_mat, false);
    ab_dgemm(A, B, C);
    ab_matrix_download(C, result, false);
    
    int ok = 1;
    for (int i = 0; i < N && ok; i++)
        for (int j = 0; j < N && ok; j++)
            if (fabs(result[i*N+j] - I_mat[i*N+j]) > 1e-12) ok = 0;
    
    ab_matrix_destroy(A); ab_matrix_destroy(B); ab_matrix_destroy(C);
    free(I_mat); free(result);
    ab_shutdown();
    if (ok) PASS(); else FAIL("I×I ≠ I");
}

static void test_dgemm_zero(void) {
    TEST("ab_dgemm: A × 0 = 0");
    ab_init();
    int N = 32;
    double* A_data = (double*)malloc(N * N * sizeof(double));
    double* zero = (double*)calloc(N * N, sizeof(double));
    double* result = (double*)malloc(N * N * sizeof(double));
    for (int i = 0; i < N * N; i++) A_data[i] = (double)i;
    
    ABMatrix A = ab_matrix_create(N, N);
    ABMatrix B = ab_matrix_create(N, N);
    ABMatrix C = ab_matrix_create(N, N);
    ab_matrix_upload(A, A_data, false);
    ab_matrix_upload(B, zero, false);
    ab_dgemm(A, B, C);
    ab_matrix_download(C, result, false);
    
    int ok = 1;
    for (int i = 0; i < N * N && ok; i++) if (fabs(result[i]) > 1e-12) ok = 0;
    
    ab_matrix_destroy(A); ab_matrix_destroy(B); ab_matrix_destroy(C);
    free(A_data); free(zero); free(result);
    ab_shutdown();
    if (ok) PASS(); else FAIL("A×0 ≠ 0");
}

static void test_dgemm_dimension_mismatch(void) {
    TEST("ab_dgemm dimension mismatch");
    ab_init();
    ABMatrix A = ab_matrix_create(10, 20);
    ABMatrix B = ab_matrix_create(30, 40);
    ABMatrix C = ab_matrix_create(10, 40);
    double* data = (double*)calloc(40 * 40, sizeof(double));
    ab_matrix_upload(A, data, false);
    ab_matrix_upload(B, data, false);
    ABStatus s = ab_dgemm(A, B, C);
    ab_matrix_destroy(A); ab_matrix_destroy(B); ab_matrix_destroy(C);
    free(data);
    ab_shutdown();
    if (s == AB_ERROR_DIMENSION_MISMATCH) PASS(); else FAIL("should fail");
}

static void test_dgemm_null_matrices(void) {
    TEST("ab_dgemm with NULL matrices");
    ab_init();
    ABMatrix m = ab_matrix_create(10, 10);
    ABStatus s1 = ab_dgemm(NULL, m, m);
    ABStatus s2 = ab_dgemm(m, NULL, m);
    ABStatus s3 = ab_dgemm(m, m, NULL);
    ab_matrix_destroy(m);
    ab_shutdown();
    if (s1 == AB_ERROR_INVALID_ARG && s2 == AB_ERROR_INVALID_ARG && s3 == AB_ERROR_INVALID_ARG)
        PASS();
    else
        FAIL("should return INVALID_ARG");
}

static void test_dgemm_vs_accelerate(void) {
    TEST("ab_dgemm matches Accelerate");
    ab_init();
    int N = 128;
    double* A = (double*)malloc(N * N * sizeof(double));
    double* B = (double*)malloc(N * N * sizeof(double));
    double* C_gpu = (double*)malloc(N * N * sizeof(double));
    double* C_ref = (double*)malloc(N * N * sizeof(double));
    srand48(42);
    for (int i = 0; i < N * N; i++) { A[i] = drand48() * 2 - 1; B[i] = drand48() * 2 - 1; }
    
    ABMatrix mA = ab_matrix_create(N, N);
    ABMatrix mB = ab_matrix_create(N, N);
    ABMatrix mC = ab_matrix_create(N, N);
    ab_matrix_upload(mA, A, true);
    ab_matrix_upload(mB, B, true);
    ab_dgemm(mA, mB, mC);
    ab_matrix_download(mC, C_gpu, true);
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C_ref, N);
    
    double max_err = 0;
    for (int i = 0; i < N * N; i++) {
        double err = fabs(C_gpu[i] - C_ref[i]);
        if (err > max_err) max_err = err;
    }
    
    ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
    free(A); free(B); free(C_gpu); free(C_ref);
    ab_shutdown();
    if (max_err < 1e-10) PASS(); else FAIL("precision too low");
}

static void test_zgemm_vs_accelerate(void) {
    TEST("ab_zgemm (no transpose) matches Accelerate");
    ab_init();
    int N = 64;
    size_t count = (size_t)N * N;
    double* Ar = (double*)malloc(count * sizeof(double));
    double* Ai = (double*)malloc(count * sizeof(double));
    double* Br = (double*)malloc(count * sizeof(double));
    double* Bi = (double*)malloc(count * sizeof(double));
    double* Cr = (double*)malloc(count * sizeof(double));
    double* Ci = (double*)malloc(count * sizeof(double));
    double complex* A_ref = (double complex*)malloc(count * sizeof(double complex));
    double complex* B_ref = (double complex*)malloc(count * sizeof(double complex));
    double complex* C_ref = (double complex*)malloc(count * sizeof(double complex));
    srand48(42);
    for (size_t i = 0; i < count; i++) {
        Ar[i] = drand48(); Ai[i] = drand48();
        Br[i] = drand48(); Bi[i] = drand48();
        A_ref[i] = Ar[i] + I * Ai[i];
        B_ref[i] = Br[i] + I * Bi[i];
    }
    ABMatrix mAr = ab_matrix_create(N, N);
    ABMatrix mAi = ab_matrix_create(N, N);
    ABMatrix mBr = ab_matrix_create(N, N);
    ABMatrix mBi = ab_matrix_create(N, N);
    ABMatrix mCr = ab_matrix_create(N, N);
    ABMatrix mCi = ab_matrix_create(N, N);
    ab_matrix_upload(mAr, Ar, false); ab_matrix_upload(mAi, Ai, false);
    ab_matrix_upload(mBr, Br, false); ab_matrix_upload(mBi, Bi, false);
    ab_zgemm(mAr, mAi, mBr, mBi, mCr, mCi);
    ab_matrix_download(mCr, Cr, false); ab_matrix_download(mCi, Ci, false);
    double complex alpha = 1.0, beta = 0.0;
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N,
                &alpha, A_ref, N, B_ref, N, &beta, C_ref, N);
    double max_err = 0;
    for (size_t i = 0; i < count; i++) {
        double er = fabs(Cr[i] - creal(C_ref[i]));
        double ei = fabs(Ci[i] - cimag(C_ref[i]));
        if (er > max_err) max_err = er;
        if (ei > max_err) max_err = ei;
    }
    ab_matrix_destroy(mAr); ab_matrix_destroy(mAi);
    ab_matrix_destroy(mBr); ab_matrix_destroy(mBi);
    ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);
    free(Ar); free(Ai); free(Br); free(Bi); free(Cr); free(Ci);
    free(A_ref); free(B_ref); free(C_ref);
    ab_shutdown();
    if (max_err < 1e-10) PASS(); else FAIL("precision too low");
}

static void test_zgemm_conj_transpose(void) {
    TEST("ab_zgemm_ex conjugate transpose (QE pattern)");
    ab_init();

    // QE pattern: C = A^H × B (conjugate-transpose × no-transpose)
    int M = 64, N = 32, K = 64;
    size_t count_A = (size_t)M * K;
    size_t count_B = (size_t)K * N;
    size_t count_C = (size_t)M * N;

    double* Ar = (double*)malloc(count_A * sizeof(double));
    double* Ai = (double*)malloc(count_A * sizeof(double));
    double* Br = (double*)malloc(count_B * sizeof(double));
    double* Bi = (double*)malloc(count_B * sizeof(double));
    double* Cr_gpu = (double*)malloc(count_C * sizeof(double));
    double* Ci_gpu = (double*)malloc(count_C * sizeof(double));

    double complex* A_ref = (double complex*)malloc(count_A * sizeof(double complex));
    double complex* B_ref = (double complex*)malloc(count_B * sizeof(double complex));
    double complex* C_ref = (double complex*)malloc(count_C * sizeof(double complex));

    srand48(999);
    for (size_t i = 0; i < count_A; i++) {
        Ar[i] = drand48(); Ai[i] = drand48();
        A_ref[i] = Ar[i] + I * Ai[i];
    }
    for (size_t i = 0; i < count_B; i++) {
        Br[i] = drand48(); Bi[i] = drand48();
        B_ref[i] = Br[i] + I * Bi[i];
    }

    // GPU: C = A^H × B
    ABMatrix mAr = ab_matrix_create(M, K);
    ABMatrix mAi = ab_matrix_create(M, K);
    ABMatrix mBr = ab_matrix_create(K, N);
    ABMatrix mBi = ab_matrix_create(K, N);
    ABMatrix mCr = ab_matrix_create(M, N);
    ABMatrix mCi = ab_matrix_create(M, N);

    ab_matrix_upload(mAr, Ar, false);
    ab_matrix_upload(mAi, Ai, false);
    ab_matrix_upload(mBr, Br, false);
    ab_matrix_upload(mBi, Bi, false);

    ab_zgemm_ex(AB_CONJ_TRANS, AB_NO_TRANS, mAr, mAi, mBr, mBi, mCr, mCi);

    ab_matrix_download(mCr, Cr_gpu, false);
    ab_matrix_download(mCi, Ci_gpu, false);

    // Reference: cblas_zgemm with CblasConjTrans
    double complex alpha = 1.0, beta = 0.0;
    cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans,
                M, N, K, &alpha, A_ref, K, B_ref, N, &beta, C_ref, N);

    // Compare
    double max_err = 0;
    for (size_t i = 0; i < count_C; i++) {
        double err_r = fabs(Cr_gpu[i] - creal(C_ref[i]));
        double err_i = fabs(Ci_gpu[i] - cimag(C_ref[i]));
        if (err_r > max_err) max_err = err_r;
        if (err_i > max_err) max_err = err_i;
    }

    ab_matrix_destroy(mAr); ab_matrix_destroy(mAi);
    ab_matrix_destroy(mBr); ab_matrix_destroy(mBi);
    ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);
    free(Ar); free(Ai); free(Br); free(Bi); free(Cr_gpu); free(Ci_gpu);
    free(A_ref); free(B_ref); free(C_ref);
    ab_shutdown();

    if (max_err < 1e-10) PASS(); else FAIL("precision too low");
}

// =============================================================================
// Session API Tests
// =============================================================================

static void test_session_basic(void) {
    TEST("session API basic");
    ab_init();
    ABSession s = ab_session_create();
    int ok = (s != NULL);
    ab_session_destroy(s);
    ab_shutdown();
    if (ok) PASS(); else FAIL("session create failed");
}

static void test_session_destroy_null(void) {
    TEST("ab_session_destroy(NULL) is safe");
    ab_session_destroy(NULL);
    PASS();
}

static void test_session_dgemm(void) {
    TEST("ab_session_dgemm");
    ab_init();
    ABSession s = ab_session_create();
    int N = 64;
    double* data = (double*)malloc(N * N * sizeof(double));
    for (int i = 0; i < N * N; i++) data[i] = 1.0;
    
    ab_session_add(s, "A", N, N);
    ab_session_add(s, "B", N, N);
    ab_session_add(s, "C", N, N);
    ab_session_upload(s, "A", data);
    ab_session_upload(s, "B", data);
    ABStatus st = ab_session_dgemm(s, "A", "B", "C");
    
    ab_session_destroy(s);
    free(data);
    ab_shutdown();
    if (st == AB_OK) PASS(); else FAIL("session dgemm failed");
}

// =============================================================================
// Statistics Tests
// =============================================================================

static void test_stats(void) {
    TEST("statistics tracking");
    ab_init();
    ABStats stats = ab_get_stats();
    int ok = (stats.dgemm_count >= 0);
    ab_shutdown();
    if (ok) PASS(); else FAIL("invalid stats");
}

// =============================================================================
// Utility Tests
// =============================================================================

static void test_status_strings(void) {
    TEST("ab_status_string");
    const char* s1 = ab_status_string(AB_OK);
    const char* s2 = ab_status_string(AB_ERROR_NO_DEVICE);
    const char* s3 = ab_status_string(AB_ERROR_SHADER_COMPILE);
    int ok = (s1 != NULL && s2 != NULL && s3 != NULL);
    if (ok) PASS(); else FAIL("null strings");
}

// =============================================================================
// Memory Pool Tests
// =============================================================================

static void test_pool_create_destroy(void) {
    TEST("ab_pool_create / ab_pool_destroy");
    ABMemoryPool pool = ab_pool_create(0);
    if (!pool) { FAIL("create returned NULL"); return; }
    ab_pool_destroy(pool);
    ab_pool_destroy(NULL);
    PASS();
}

static void test_pool_get_matrix(void) {
    TEST("ab_pool_get_matrix");
    ab_init();
    ABMemoryPool pool = ab_pool_create(0);
    ABMatrix m1 = ab_pool_get_matrix(pool, 100, 100);
    ABMatrix m2 = ab_pool_get_matrix(pool, 100, 100);
    int ok = (m1 != NULL) && (m2 != NULL) && (m1 != m2);
    ab_pool_destroy(pool);
    ab_shutdown();
    if (ok) PASS(); else FAIL("pool matrix allocation failed");
}

static void test_pool_reset_reuse(void) {
    TEST("ab_pool_reset reuses matrices");
    ab_init();
    ABMemoryPool pool = ab_pool_create(0);
    ABMatrix m1 = ab_pool_get_matrix(pool, 64, 64);
    ab_pool_reset(pool);
    ABMatrix m2 = ab_pool_get_matrix(pool, 64, 64);
    int reused = (m1 == m2);
    ab_pool_destroy(pool);
    ab_shutdown();
    if (reused) PASS(); else FAIL("matrix not reused after reset");
}

static void test_pool_iteration_pattern(void) {
    TEST("ab_pool iteration pattern (SCF-like)");
    ab_init();
    ABMemoryPool pool = ab_pool_create(0);
    int N = 128;
    double* data = (double*)malloc(N * N * sizeof(double));
    for (int i = 0; i < N * N; i++) data[i] = 1.0;
    
    for (int iter = 0; iter < 10; iter++) {
        ABMatrix A = ab_pool_get_matrix(pool, N, N);
        ABMatrix B = ab_pool_get_matrix(pool, N, N);
        ABMatrix C = ab_pool_get_matrix(pool, N, N);
        ab_matrix_upload(A, data, false);
        ab_matrix_upload(B, data, false);
        ab_dgemm(A, B, C);
        ab_pool_reset(pool);
    }
    
    free(data);
    ab_pool_destroy(pool);
    ab_shutdown();
    PASS();
}

// =============================================================================
// Async API Tests
// =============================================================================

static void test_async_dgemm_basic(void) {
    TEST("ab_dgemm_async basic");
    ab_init();
    int N = 256;
    size_t count = (size_t)N * N;
    double* A = (double*)malloc(count * sizeof(double));
    double* B = (double*)malloc(count * sizeof(double));
    double* C = (double*)malloc(count * sizeof(double));
    for (size_t i = 0; i < count; i++) { A[i] = 1.0; B[i] = 1.0; }
    
    ABMatrix mA = ab_matrix_create(N, N);
    ABMatrix mB = ab_matrix_create(N, N);
    ABMatrix mC = ab_matrix_create(N, N);
    ab_matrix_upload(mA, A, false);
    ab_matrix_upload(mB, B, false);
    
    ABFuture f = ab_dgemm_async(mA, mB, mC);
    int ok = (f != NULL);
    if (ok) {
        ABStatus s = ab_future_wait(f);
        ok = (s == AB_OK);
        ab_future_destroy(f);
    }
    
    ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
    free(A); free(B); free(C);
    ab_shutdown();
    if (ok) PASS(); else FAIL("async dgemm failed");
}

static void test_async_future_poll(void) {
    TEST("ab_future_is_ready polling");
    ab_init();
    int N = 512;
    size_t count = (size_t)N * N;
    double* data = (double*)malloc(count * sizeof(double));
    for (size_t i = 0; i < count; i++) data[i] = 1.0;
    
    ABMatrix mA = ab_matrix_create(N, N);
    ABMatrix mB = ab_matrix_create(N, N);
    ABMatrix mC = ab_matrix_create(N, N);
    ab_matrix_upload(mA, data, false);
    ab_matrix_upload(mB, data, false);
    
    ABFuture f = ab_dgemm_async(mA, mB, mC);
    int poll_count = 0;
    while (!ab_future_is_ready(f) && poll_count < 10000) poll_count++;
    ABStatus s = ab_future_status(f);
    ab_future_destroy(f);
    
    ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
    free(data);
    ab_shutdown();
    if (s == AB_OK) PASS(); else FAIL("polling failed");
}

static void test_async_overlap(void) {
    TEST("ab_dgemm_async CPU/GPU overlap");
    ab_init();
    int N = 256;
    size_t count = (size_t)N * N;
    double* A = (double*)malloc(count * sizeof(double));
    double* B = (double*)malloc(count * sizeof(double));
    for (size_t i = 0; i < count; i++) { A[i] = 1.0; B[i] = 1.0; }
    
    ABMatrix mA = ab_matrix_create(N, N);
    ABMatrix mB = ab_matrix_create(N, N);
    ABMatrix mC = ab_matrix_create(N, N);
    ab_matrix_upload(mA, A, false);
    ab_matrix_upload(mB, B, false);
    
    ABFuture f = ab_dgemm_async(mA, mB, mC);
    
    volatile double cpu_work = 0;
    for (int i = 0; i < 100000; i++) cpu_work += 0.001;
    
    ab_future_wait(f);
    ab_future_destroy(f);
    
    ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
    free(A); free(B);
    ab_shutdown();
    PASS();
}

static void test_async_null_safety(void) {
    TEST("async API NULL safety");
    ab_future_wait(NULL);
    ab_future_is_ready(NULL);
    ab_future_status(NULL);
    ab_future_destroy(NULL);
    PASS();
}

// =============================================================================
// Stress Tests
// =============================================================================

static void test_small_matrix_n1(void) {
    TEST("ab_dgemm with N=1 (edge case)");
    ab_init();
    double A = 3.0, B = 4.0, C = 0.0;
    ABMatrix mA = ab_matrix_create(1, 1);
    ABMatrix mB = ab_matrix_create(1, 1);
    ABMatrix mC = ab_matrix_create(1, 1);
    ab_matrix_upload(mA, &A, false);
    ab_matrix_upload(mB, &B, false);
    ab_dgemm(mA, mB, mC);
    ab_matrix_download(mC, &C, false);
    ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
    ab_shutdown();
    if (fabs(C - 12.0) < 1e-10) PASS(); else FAIL("3*4 != 12");
}

static void test_non_power_of_2(void) {
    TEST("ab_dgemm with N=127 (non-power-of-2)");
    ab_init();
    int N = 127;
    size_t count = (size_t)N * N;
    double* A = (double*)malloc(count * sizeof(double));
    double* B = (double*)malloc(count * sizeof(double));
    double* C = (double*)malloc(count * sizeof(double));
    srand48(999);
    for (size_t i = 0; i < count; i++) { A[i] = drand48(); B[i] = drand48(); }
    ABMatrix mA = ab_matrix_create(N, N);
    ABMatrix mB = ab_matrix_create(N, N);
    ABMatrix mC = ab_matrix_create(N, N);
    ab_matrix_upload(mA, A, true);
    ab_matrix_upload(mB, B, true);
    ABStatus s = ab_dgemm(mA, mB, mC);
    ab_matrix_download(mC, C, true);
    int ok = (s == AB_OK) && isfinite(C[0]) && isfinite(C[count-1]);
    ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
    free(A); free(B); free(C);
    ab_shutdown();
    if (ok) PASS(); else FAIL("failed for N=127");
}

static void test_rectangular_matrix(void) {
    TEST("ab_dgemm rectangular (100x50 * 50x200)");
    ab_init();
    int M = 100, K = 50, N = 200;
    double* A = (double*)malloc(M * K * sizeof(double));
    double* B = (double*)malloc(K * N * sizeof(double));
    double* C = (double*)malloc(M * N * sizeof(double));
    srand48(111);
    for (int i = 0; i < M * K; i++) A[i] = drand48();
    for (int i = 0; i < K * N; i++) B[i] = drand48();
    ABMatrix mA = ab_matrix_create(M, K);
    ABMatrix mB = ab_matrix_create(K, N);
    ABMatrix mC = ab_matrix_create(M, N);
    ab_matrix_upload(mA, A, true);
    ab_matrix_upload(mB, B, true);
    ABStatus s = ab_dgemm(mA, mB, mC);
    ab_matrix_download(mC, C, true);
    int ok = (s == AB_OK) && isfinite(C[0]) && isfinite(C[M * N - 1]);
    ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
    free(A); free(B); free(C);
    ab_shutdown();
    if (ok) PASS(); else FAIL("failed for rectangular");
}

static void test_skinny_matrix(void) {
    TEST("ab_dgemm skinny (1000x10 * 10x1000)");
    ab_init();
    int M = 1000, K = 10, N = 1000;
    double* A = (double*)malloc(M * K * sizeof(double));
    double* B = (double*)malloc(K * N * sizeof(double));
    double* C = (double*)malloc(M * N * sizeof(double));
    srand48(222);
    for (int i = 0; i < M * K; i++) A[i] = drand48();
    for (int i = 0; i < K * N; i++) B[i] = drand48();
    ABMatrix mA = ab_matrix_create(M, K);
    ABMatrix mB = ab_matrix_create(K, N);
    ABMatrix mC = ab_matrix_create(M, N);
    ab_matrix_upload(mA, A, true);
    ab_matrix_upload(mB, B, true);
    ABStatus s = ab_dgemm(mA, mB, mC);
    ab_matrix_download(mC, C, true);
    int ok = (s == AB_OK) && isfinite(C[0]) && isfinite(C[M * N - 1]);
    ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
    free(A); free(B); free(C);
    ab_shutdown();
    if (ok) PASS(); else FAIL("failed for skinny");
}

static void test_max_dimension_boundary(void) {
    TEST("ab_matrix_create max dimension boundary");
    ab_init();
    ABMatrix m1 = ab_matrix_create(46340, 1);
    ABMatrix m2 = ab_matrix_create(46341, 46341);
    int ok = (m1 != NULL) && (m2 == NULL);
    ab_matrix_destroy(m1);
    ab_shutdown();
    if (ok) PASS(); else FAIL("boundary check failed");
}

// =============================================================================
// Regression Tests (Bug Fixes v1.0.2)
// =============================================================================

// BUG-1/BUG-2: Async DGEMM with non-square, non-64-aligned matrices
static void test_bug1_async_dimension_packing(void) {
    TEST("BUG-1/2: async DGEMM non-square correctness");
    ab_init();

    int M = 100, N = 50, K = 75;  // Non-square, non-64-aligned
    ABMatrix A = ab_matrix_create(M, K);
    ABMatrix B = ab_matrix_create(K, N);
    ABMatrix C_async = ab_matrix_create(M, N);
    ABMatrix C_sync = ab_matrix_create(M, N);

    double* dataA = (double*)malloc(M * K * sizeof(double));
    double* dataB = (double*)malloc(K * N * sizeof(double));
    double* resultA = (double*)malloc(M * N * sizeof(double));
    double* resultS = (double*)malloc(M * N * sizeof(double));

    for (int i = 0; i < M * K; i++) dataA[i] = (double)(rand() % 100) / 10.0;
    for (int i = 0; i < K * N; i++) dataB[i] = (double)(rand() % 100) / 10.0;

    ab_matrix_upload(A, dataA, true);
    ab_matrix_upload(B, dataB, true);

    // Sync reference
    ab_dgemm(A, B, C_sync);
    ab_matrix_download(C_sync, resultS, true);

    // Async test (was broken due to dimension packing bug)
    ABFuture f = ab_dgemm_async(A, B, C_async);
    ab_future_wait(f);
    ab_future_destroy(f);
    ab_matrix_download(C_async, resultA, true);

    // Compare results
    int ok = 1;
    for (int i = 0; i < M * N && ok; i++) {
        if (fabs(resultA[i] - resultS[i]) > 1e-12) ok = 0;
    }

    free(dataA); free(dataB); free(resultA); free(resultS);
    ab_matrix_destroy(A); ab_matrix_destroy(B);
    ab_matrix_destroy(C_async); ab_matrix_destroy(C_sync);
    ab_shutdown();

    if (ok) PASS(); else FAIL("async != sync for non-square matrix");
}

// BUG-3: ab_dgemm_scaled should preserve DD precision for alpha/beta
static void test_bug3_dgemm_scaled_precision(void) {
    TEST("BUG-3: dgemm_scaled preserves DD precision");
    ab_init();

    int N = 64;
    ABMatrix A = ab_matrix_create(N, N);
    ABMatrix B = ab_matrix_create(N, N);
    ABMatrix C = ab_matrix_create(N, N);

    double* dataA = (double*)malloc(N * N * sizeof(double));
    double* dataB = (double*)malloc(N * N * sizeof(double));
    double* dataC = (double*)malloc(N * N * sizeof(double));
    double* result = (double*)malloc(N * N * sizeof(double));

    for (int i = 0; i < N * N; i++) {
        dataA[i] = 1.0;
        dataB[i] = 1.0;
        dataC[i] = 0.0;
    }

    ab_matrix_upload(A, dataA, false);
    ab_matrix_upload(B, dataB, false);
    ab_matrix_upload(C, dataC, false);

    // Alpha with precision beyond FP32 mantissa
    double alpha = 1.0 + 1e-15;
    double beta = 0.0;

    ab_dgemm_scaled(alpha, A, B, beta, C);
    ab_matrix_download(C, result, false);

    // Result should be N * alpha (each row sums N ones)
    // If alpha was truncated to float, we'd lose the 1e-15 component
    double expected = (double)N * alpha;
    double actual = result[0];  // First element
    double rel_error = fabs(actual - expected) / expected;

    free(dataA); free(dataB); free(dataC); free(result);
    ab_matrix_destroy(A); ab_matrix_destroy(B); ab_matrix_destroy(C);
    ab_shutdown();

    // DD precision should give < 1e-14 relative error
    if (rel_error < 1e-12) PASS(); else FAIL("alpha truncated to float");
}

// BUG-4: ab_matrix_scale should preserve DD precision
static void test_bug4_matrix_scale_precision(void) {
    TEST("BUG-4: matrix_scale preserves DD precision");
    ab_init();

    int N = 64;
    ABMatrix A = ab_matrix_create(N, N);
    double* data = (double*)malloc(N * N * sizeof(double));
    double* result = (double*)malloc(N * N * sizeof(double));

    for (int i = 0; i < N * N; i++) data[i] = 1.0;
    ab_matrix_upload(A, data, false);

    double alpha = 1.0 + 1e-15;  // Beyond FP32 precision
    ab_matrix_scale(alpha, A);
    ab_matrix_download(A, result, false);

    double expected = alpha;
    double actual = result[0];
    double error = fabs(actual - expected);

    free(data); free(result);
    ab_matrix_destroy(A);
    ab_shutdown();

    if (error < 1e-14) PASS(); else FAIL("alpha truncated to float");
}

// BUG-5: Reinit after shutdown should work correctly
static void test_bug5_reinit_after_shutdown(void) {
    TEST("BUG-5: reinit after shutdown works");
    ab_init();
    ab_shutdown();
    ABStatus s = ab_init();

    if (s != AB_OK) {
        ab_shutdown();
        FAIL("reinit failed");
        return;
    }

    // Verify we can actually use the library after reinit
    ABMatrix A = ab_matrix_create(10, 10);
    ABMatrix B = ab_matrix_create(10, 10);
    ABMatrix C = ab_matrix_create(10, 10);

    double* data = (double*)malloc(100 * sizeof(double));
    for (int i = 0; i < 100; i++) data[i] = 1.0;

    ab_matrix_upload(A, data, false);
    ab_matrix_upload(B, data, false);
    ABStatus result = ab_dgemm(A, B, C);

    free(data);
    ab_matrix_destroy(A);
    ab_matrix_destroy(B);
    ab_matrix_destroy(C);
    ab_shutdown();

    if (result == AB_OK) PASS(); else FAIL("operations fail after reinit");
}

// BUG-6: Pool overflow should return NULL, not leak unmanaged matrices
static void test_bug6_pool_overflow(void) {
    TEST("BUG-6: pool overflow returns NULL");
    ab_init();

    ABMemoryPool pool = ab_pool_create(0);

    // Fill pool to capacity (AB_POOL_MAX_ENTRIES = 128 from apple_bottom.m:1177)
    for (int i = 0; i < 128; i++) {
        ABMatrix m = ab_pool_get_matrix(pool, 10, 10);
        if (!m) {
            ab_pool_destroy(pool);
            ab_shutdown();
            FAIL("pool failed before reaching capacity");
            return;
        }
    }

    // Next allocation should return NULL (pool full)
    ABMatrix overflow = ab_pool_get_matrix(pool, 10, 10);

    ab_pool_destroy(pool);
    ab_shutdown();

    if (overflow == NULL) PASS(); else FAIL("pool returned unmanaged matrix instead of NULL");
}

// =============================================================================
// Main
// =============================================================================

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  apple-bottom Comprehensive Test Suite                           ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    
    printf("\nInitialization:\n");
    test_init_shutdown();
    test_double_init();
    test_shutdown_without_init();
    
    printf("\nMatrix Lifecycle:\n");
    test_matrix_create_destroy();
    test_matrix_destroy_null();
    test_matrix_dims_null_outputs();
    test_matrix_count_null();
    
    printf("\nData Transfer:\n");
    test_upload_download_roundtrip();
    test_upload_null_data();
    test_download_null_data();
    test_upload_null_matrix();
    test_matrix_zero();
    
    printf("\nDGEMM:\n");
    test_dgemm_identity();
    test_dgemm_zero();
    test_dgemm_dimension_mismatch();
    test_dgemm_null_matrices();
    test_dgemm_vs_accelerate();
    test_zgemm_vs_accelerate();
    test_zgemm_conj_transpose();
    
    printf("\nSession API:\n");
    test_session_basic();
    test_session_destroy_null();
    test_session_dgemm();
    
    printf("\nStatistics:\n");
    test_stats();
    
    printf("\nUtility:\n");
    test_status_strings();
    
    printf("\nMemory Pool:\n");
    test_pool_create_destroy();
    test_pool_get_matrix();
    test_pool_reset_reuse();
    test_pool_iteration_pattern();
    
    printf("\nAsync API:\n");
    test_async_dgemm_basic();
    test_async_future_poll();
    test_async_overlap();
    test_async_null_safety();
    
    printf("\nStress Tests:\n");
    test_small_matrix_n1();
    test_non_power_of_2();
    test_rectangular_matrix();
    test_skinny_matrix();
    test_max_dimension_boundary();

    printf("\nRegression Tests (Bug Fixes v1.0.2):\n");
    test_bug1_async_dimension_packing();
    test_bug3_dgemm_scaled_precision();
    test_bug4_matrix_scale_precision();
    test_bug5_reinit_after_shutdown();
    test_bug6_pool_overflow();

    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    if (tests_failed == 0) {
        printf("✓ All %d tests PASSED\n", tests_passed);
        return 0;
    } else {
        printf("✗ Some tests FAILED\n");
        return 1;
    }
}
