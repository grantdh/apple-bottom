// =============================================================================
// apple-bottom Adversarial & System-Level Test Suite
// =============================================================================
// Tests pathological inputs, IEEE 754 edge cases, thread safety, and memory
// integrity that go beyond the happy-path correctness tests.
//
// Categories:
//   1. IEEE 754 Edge Cases (NaN, Inf, subnormal, beta=0 semantics)
//   2. Ill-Conditioned Matrices (extreme condition numbers)
//   3. Boundary Dimensions (1×1, prime, non-aligned, extreme aspect ratios)
//   4. ZHERK GPU Transpose (validates the GPU-native path)
//   5. Concurrent Stress (pthread hammer on separate sessions)
//   6. Memory Pool Exhaustion
// =============================================================================

#include "apple_bottom.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <complex.h>
#include <pthread.h>
#include <Accelerate/Accelerate.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  %-50s ", name)
#define PASS() do { printf("✓ PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("✗ FAIL (%s)\n", msg); tests_failed++; } while(0)

// =============================================================================
// Helpers
// =============================================================================

static double frobenius_rel_error(const double* A, const double* B, int N) {
    double norm_diff = 0.0, norm_ref = 0.0;
    for (int i = 0; i < N * N; i++) {
        double d = A[i] - B[i];
        norm_diff += d * d;
        norm_ref += B[i] * B[i];
    }
    if (norm_ref == 0.0) return norm_diff == 0.0 ? 0.0 : INFINITY;
    return sqrt(norm_diff / norm_ref);
}

// Generate a diagonal matrix with specified condition number
// sigma_i ranges geometrically from 1 to 1/kappa
static void gen_ill_conditioned(double* A, int N, double kappa) {
    memset(A, 0, (size_t)N * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        double t = (double)i / (N > 1 ? (N - 1) : 1);
        A[i * N + i] = pow(kappa, -t);  // sigma_i from 1 down to 1/kappa
    }
}

// =============================================================================
// 1. IEEE 754 Edge Cases
// =============================================================================

static void test_dgemm_nan_input(void) {
    TEST("DGEMM with NaN in A propagates NaN");
    ab_init();
    int N = 64;
    double* A = malloc(N * N * sizeof(double));
    double* B = malloc(N * N * sizeof(double));
    double* C = malloc(N * N * sizeof(double));

    for (int i = 0; i < N * N; i++) { A[i] = 1.0; B[i] = 1.0; }
    A[0] = NAN;  // Poison one element

    ABMatrix mA = ab_matrix_create(N, N);
    ABMatrix mB = ab_matrix_create(N, N);
    ABMatrix mC = ab_matrix_create(N, N);
    ab_matrix_upload(mA, A, true);
    ab_matrix_upload(mB, B, true);
    ab_dgemm(mA, mB, mC);
    ab_matrix_download(mC, C, true);

    // Row 0 of C should contain NaN (row 0 of A has NaN at col 0)
    int has_nan = isnan(C[0]);
    ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
    free(A); free(B); free(C);
    ab_shutdown();
    if (has_nan) PASS(); else FAIL("NaN not propagated to C[0,0]");
}

static void test_dgemm_inf_input(void) {
    TEST("DGEMM with Inf in A produces non-finite C");
    // DD arithmetic: twoSum(Inf, x) produces e = Inf - Inf = NaN,
    // so Inf inputs yield NaN (not Inf) in the DD representation.
    // This is inherent to Dekker/Knuth error-free transformations.
    // We verify the result is non-finite (NaN or Inf), not a valid number.
    ab_init();
    int N = 64;
    double* A = malloc(N * N * sizeof(double));
    double* B = malloc(N * N * sizeof(double));
    double* C = malloc(N * N * sizeof(double));

    for (int i = 0; i < N * N; i++) { A[i] = 1.0; B[i] = 1.0; }
    A[0] = INFINITY;

    ABMatrix mA = ab_matrix_create(N, N);
    ABMatrix mB = ab_matrix_create(N, N);
    ABMatrix mC = ab_matrix_create(N, N);
    ab_matrix_upload(mA, A, true);
    ab_matrix_upload(mB, B, true);
    ab_dgemm(mA, mB, mC);
    ab_matrix_download(mC, C, true);

    // C[0,0] should be non-finite (NaN or Inf — either is correct for DD)
    int non_finite = !isfinite(C[0]);
    ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
    free(A); free(B); free(C);
    ab_shutdown();
    if (non_finite) PASS(); else FAIL("Inf input produced finite C[0,0]");
}

static void test_dgemm_scaled_beta_zero_suppresses_nan(void) {
    TEST("beta=0 suppresses NaN in C (BLAS standard)");
    // Per BLAS standard: beta=0 means C := alpha*A*B, ignoring old C values.
    // Even if C contains NaN, result should be clean.
    ab_init();
    int N = 64;
    double* A = malloc(N * N * sizeof(double));
    double* B = malloc(N * N * sizeof(double));
    double* C = malloc(N * N * sizeof(double));

    for (int i = 0; i < N * N; i++) { A[i] = 1.0; B[i] = 1.0; C[i] = NAN; }

    ABMatrix mA = ab_matrix_create(N, N);
    ABMatrix mB = ab_matrix_create(N, N);
    ABMatrix mC = ab_matrix_create(N, N);
    ab_matrix_upload(mA, A, true);
    ab_matrix_upload(mB, B, true);
    ab_matrix_upload(mC, C, true);

    // beta=0: should ignore NaN-initialized C
    ab_dgemm_scaled(1.0, mA, mB, 0.0, mC);
    ab_matrix_download(mC, C, true);

    int all_clean = 1;
    for (int i = 0; i < N * N; i++) {
        if (isnan(C[i]) || isinf(C[i])) { all_clean = 0; break; }
    }

    // Each element of C should be N (sum of 1*1 across K=N)
    double expected = (double)N;
    int correct = 1;
    for (int i = 0; i < N; i++) {
        double err = fabs(C[i * N + i] - expected);
        if (err > 1e-10) { correct = 0; break; }
    }

    ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
    free(A); free(B); free(C);
    ab_shutdown();

    if (all_clean && correct) PASS();
    else if (!all_clean) FAIL("NaN leaked through beta=0");
    else FAIL("incorrect values with beta=0");
}

static void test_dgemm_subnormal_input(void) {
    TEST("DGEMM with subnormal inputs");
    ab_init();
    int N = 64;
    double* A = malloc(N * N * sizeof(double));
    double* B = malloc(N * N * sizeof(double));
    double* C = malloc(N * N * sizeof(double));

    // Fill with subnormal numbers (smallest non-zero doubles)
    double subnorm = DBL_MIN * 0.5;  // Below DBL_MIN = subnormal
    for (int i = 0; i < N * N; i++) { A[i] = subnorm; B[i] = subnorm; }

    ABMatrix mA = ab_matrix_create(N, N);
    ABMatrix mB = ab_matrix_create(N, N);
    ABMatrix mC = ab_matrix_create(N, N);
    ab_matrix_upload(mA, A, true);
    ab_matrix_upload(mB, B, true);
    ab_dgemm(mA, mB, mC);
    ab_matrix_download(mC, C, true);

    // Result should be zero or very small (subnorm^2 * N underflows)
    int ok = 1;
    for (int i = 0; i < N * N; i++) {
        if (isnan(C[i]) || isinf(C[i])) { ok = 0; break; }
    }

    ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
    free(A); free(B); free(C);
    ab_shutdown();
    if (ok) PASS(); else FAIL("NaN/Inf from subnormal inputs");
}

static void test_dgemm_zero_matrix(void) {
    TEST("DGEMM with all-zero A produces zero C");
    ab_init();
    int N = 128;
    double* A = calloc(N * N, sizeof(double));
    double* B = malloc(N * N * sizeof(double));
    double* C = malloc(N * N * sizeof(double));
    for (int i = 0; i < N * N; i++) B[i] = (double)(i + 1);

    ABMatrix mA = ab_matrix_create(N, N);
    ABMatrix mB = ab_matrix_create(N, N);
    ABMatrix mC = ab_matrix_create(N, N);
    ab_matrix_upload(mA, A, true);
    ab_matrix_upload(mB, B, true);
    ab_dgemm(mA, mB, mC);
    ab_matrix_download(mC, C, true);

    int ok = 1;
    for (int i = 0; i < N * N; i++) {
        if (C[i] != 0.0) { ok = 0; break; }
    }

    ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
    free(A); free(B); free(C);
    ab_shutdown();
    if (ok) PASS(); else FAIL("non-zero result from zero matrix");
}

// =============================================================================
// 2. Ill-Conditioned Matrices
// =============================================================================

static void test_dgemm_high_condition_number(void) {
    TEST("DGEMM precision with kappa=1e14 diagonal");
    ab_init();
    int N = 128;
    double* A = calloc(N * N, sizeof(double));
    double* B = calloc(N * N, sizeof(double));
    double* C_gpu = malloc(N * N * sizeof(double));
    double* C_ref = malloc(N * N * sizeof(double));

    // Diagonal A with condition number ~1e14
    double kappa = 1e14;
    for (int i = 0; i < N; i++) {
        double t = (double)i / (N - 1);
        A[i * N + i] = pow(kappa, -t);
    }
    // B = identity
    for (int i = 0; i < N; i++) B[i * N + i] = 1.0;

    ABMatrix mA = ab_matrix_create(N, N);
    ABMatrix mB = ab_matrix_create(N, N);
    ABMatrix mC = ab_matrix_create(N, N);
    ab_matrix_upload(mA, A, true);
    ab_matrix_upload(mB, B, true);
    ab_dgemm(mA, mB, mC);
    ab_matrix_download(mC, C_gpu, true);

    // Reference: C = A * I = A
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0, A, N, B, N, 0.0, C_ref, N);

    double err = frobenius_rel_error(C_gpu, C_ref, N);

    ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
    free(A); free(B); free(C_gpu); free(C_ref);
    ab_shutdown();

    // DD arithmetic should handle this well since it's just diagonal * identity
    if (err < 1e-14) PASS();
    else { char buf[64]; snprintf(buf, sizeof(buf), "err=%.2e", err); FAIL(buf); }
}

static void test_dgemm_large_dynamic_range(void) {
    TEST("DGEMM with O(1e15) and O(1e-15) elements");
    ab_init();
    int N = 64;
    double* A = malloc(N * N * sizeof(double));
    double* B = malloc(N * N * sizeof(double));
    double* C_gpu = malloc(N * N * sizeof(double));
    double* C_ref = malloc(N * N * sizeof(double));

    srand48(12345);
    for (int i = 0; i < N * N; i++) {
        A[i] = (i % 2 == 0) ? 1e15 * drand48() : 1e-15 * drand48();
        B[i] = (i % 3 == 0) ? 1e15 * drand48() : 1e-15 * drand48();
    }

    ABMatrix mA = ab_matrix_create(N, N);
    ABMatrix mB = ab_matrix_create(N, N);
    ABMatrix mC = ab_matrix_create(N, N);
    ab_matrix_upload(mA, A, true);
    ab_matrix_upload(mB, B, true);
    ab_dgemm(mA, mB, mC);
    ab_matrix_download(mC, C_gpu, true);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0, A, N, B, N, 0.0, C_ref, N);

    double err = frobenius_rel_error(C_gpu, C_ref, N);

    ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
    free(A); free(B); free(C_gpu); free(C_ref);
    ab_shutdown();

    // Wider tolerance due to extreme dynamic range
    if (err < 1e-12) PASS();
    else { char buf[64]; snprintf(buf, sizeof(buf), "err=%.2e", err); FAIL(buf); }
}

// =============================================================================
// 3. Boundary Dimensions
// =============================================================================

static void test_dgemm_1x1(void) {
    TEST("DGEMM 1×1 matrices");
    ab_init();
    double a = 3.14159, b = 2.71828, c = 0.0;
    ABMatrix mA = ab_matrix_create(1, 1);
    ABMatrix mB = ab_matrix_create(1, 1);
    ABMatrix mC = ab_matrix_create(1, 1);
    ab_matrix_upload(mA, &a, true);
    ab_matrix_upload(mB, &b, true);
    ab_dgemm(mA, mB, mC);
    ab_matrix_download(mC, &c, true);

    double expected = a * b;
    double err = fabs(c - expected) / fabs(expected);

    ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
    ab_shutdown();
    if (err < 1e-14) PASS();
    else { char buf[64]; snprintf(buf, sizeof(buf), "err=%.2e", err); FAIL(buf); }
}

static void test_dgemm_prime_dimensions(void) {
    TEST("DGEMM with prime dimensions (127×131)");
    ab_init();
    int M = 127, K = 131, N = 127;
    double* A = malloc((size_t)M * K * sizeof(double));
    double* B = malloc((size_t)K * N * sizeof(double));
    double* C_gpu = malloc((size_t)M * N * sizeof(double));
    double* C_ref = malloc((size_t)M * N * sizeof(double));

    srand48(99);
    for (int i = 0; i < M * K; i++) A[i] = drand48() * 2 - 1;
    for (int i = 0; i < K * N; i++) B[i] = drand48() * 2 - 1;

    ABMatrix mA = ab_matrix_create(M, K);
    ABMatrix mB = ab_matrix_create(K, N);
    ABMatrix mC = ab_matrix_create(M, N);
    ab_matrix_upload(mA, A, true);
    ab_matrix_upload(mB, B, true);
    ab_dgemm(mA, mB, mC);
    ab_matrix_download(mC, C_gpu, true);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0, A, K, B, N, 0.0, C_ref, N);

    double norm_diff = 0.0, norm_ref = 0.0;
    for (int i = 0; i < M * N; i++) {
        double d = C_gpu[i] - C_ref[i];
        norm_diff += d * d;
        norm_ref += C_ref[i] * C_ref[i];
    }
    double err = sqrt(norm_diff / norm_ref);

    ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
    free(A); free(B); free(C_gpu); free(C_ref);
    ab_shutdown();
    if (err < 1e-14) PASS();
    else { char buf[64]; snprintf(buf, sizeof(buf), "err=%.2e", err); FAIL(buf); }
}

static void test_zgemm_1x1(void) {
    TEST("ZGEMM 1×1 complex matrices");
    ab_init();
    double ar = 3.0, ai = 4.0, br = 1.0, bi = 2.0;
    double cr = 0.0, ci = 0.0;
    // (3+4i)(1+2i) = 3+6i+4i+8i² = 3+10i-8 = -5+10i

    ABMatrix mAr = ab_matrix_create(1, 1);
    ABMatrix mAi = ab_matrix_create(1, 1);
    ABMatrix mBr = ab_matrix_create(1, 1);
    ABMatrix mBi = ab_matrix_create(1, 1);
    ABMatrix mCr = ab_matrix_create(1, 1);
    ABMatrix mCi = ab_matrix_create(1, 1);
    ab_matrix_upload(mAr, &ar, true);
    ab_matrix_upload(mAi, &ai, true);
    ab_matrix_upload(mBr, &br, true);
    ab_matrix_upload(mBi, &bi, true);
    ab_zgemm(mAr, mAi, mBr, mBi, mCr, mCi);
    ab_matrix_download(mCr, &cr, true);
    ab_matrix_download(mCi, &ci, true);

    int ok = (fabs(cr - (-5.0)) < 1e-12) && (fabs(ci - 10.0) < 1e-12);

    ab_matrix_destroy(mAr); ab_matrix_destroy(mAi);
    ab_matrix_destroy(mBr); ab_matrix_destroy(mBi);
    ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);
    ab_shutdown();
    if (ok) PASS(); else FAIL("wrong result");
}

// =============================================================================
// 4. ZHERK GPU Transpose Validation
// =============================================================================

static void test_zherk_vs_cblas(void) {
    TEST("ZHERK GPU vs cblas_zherk reference");
    ab_init();
    int N = 256, K = 128;
    size_t countA = (size_t)N * K;
    size_t countC = (size_t)N * N;

    double* Ar = malloc(countA * sizeof(double));
    double* Ai = malloc(countA * sizeof(double));
    double* Cr_gpu = malloc(countC * sizeof(double));
    double* Ci_gpu = malloc(countC * sizeof(double));
    double complex* A_ref = malloc(countA * sizeof(double complex));
    double complex* C_ref = calloc(countC, sizeof(double complex));

    srand48(42);
    for (size_t i = 0; i < countA; i++) {
        Ar[i] = drand48() * 2 - 1;
        Ai[i] = drand48() * 2 - 1;
        A_ref[i] = Ar[i] + I * Ai[i];
    }

    // GPU ZHERK
    ABMatrix mAr = ab_matrix_create(N, K);
    ABMatrix mAi = ab_matrix_create(N, K);
    ABMatrix mCr = ab_matrix_create(N, N);
    ABMatrix mCi = ab_matrix_create(N, N);
    ab_matrix_upload(mAr, Ar, true);
    ab_matrix_upload(mAi, Ai, true);
    ab_zherk(mAr, mAi, mCr, mCi);
    ab_matrix_download(mCr, Cr_gpu, true);
    ab_matrix_download(mCi, Ci_gpu, true);

    // Reference
    cblas_zherk(CblasRowMajor, CblasUpper, CblasNoTrans,
                N, K, 1.0, A_ref, K, 0.0, C_ref, N);

    // Compare upper triangle
    double max_err = 0;
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            double err_r = fabs(Cr_gpu[i * N + j] - creal(C_ref[i * N + j]));
            double err_i = fabs(Ci_gpu[i * N + j] - cimag(C_ref[i * N + j]));
            if (err_r > max_err) max_err = err_r;
            if (err_i > max_err) max_err = err_i;
        }
    }

    ab_matrix_destroy(mAr); ab_matrix_destroy(mAi);
    ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);
    free(Ar); free(Ai); free(Cr_gpu); free(Ci_gpu);
    free(A_ref); free(C_ref);
    ab_shutdown();

    if (max_err < 1e-10) PASS();
    else { char buf[64]; snprintf(buf, sizeof(buf), "max_err=%.2e", max_err); FAIL(buf); }
}

static void test_zherk_rectangular(void) {
    TEST("ZHERK with tall-skinny A (512×32)");
    ab_init();
    int N = 512, K = 32;
    size_t countA = (size_t)N * K;
    size_t countC = (size_t)N * N;

    double* Ar = malloc(countA * sizeof(double));
    double* Ai = malloc(countA * sizeof(double));
    double* Cr_gpu = malloc(countC * sizeof(double));
    double* Ci_gpu = malloc(countC * sizeof(double));
    double complex* A_ref = malloc(countA * sizeof(double complex));
    double complex* C_ref = calloc(countC, sizeof(double complex));

    srand48(77);
    for (size_t i = 0; i < countA; i++) {
        Ar[i] = drand48() * 2 - 1;
        Ai[i] = drand48() * 2 - 1;
        A_ref[i] = Ar[i] + I * Ai[i];
    }

    ABMatrix mAr = ab_matrix_create(N, K);
    ABMatrix mAi = ab_matrix_create(N, K);
    ABMatrix mCr = ab_matrix_create(N, N);
    ABMatrix mCi = ab_matrix_create(N, N);
    ab_matrix_upload(mAr, Ar, true);
    ab_matrix_upload(mAi, Ai, true);
    ab_zherk(mAr, mAi, mCr, mCi);
    ab_matrix_download(mCr, Cr_gpu, true);
    ab_matrix_download(mCi, Ci_gpu, true);

    cblas_zherk(CblasRowMajor, CblasUpper, CblasNoTrans,
                N, K, 1.0, A_ref, K, 0.0, C_ref, N);

    double max_err = 0;
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            double err_r = fabs(Cr_gpu[i * N + j] - creal(C_ref[i * N + j]));
            double err_i = fabs(Ci_gpu[i * N + j] - cimag(C_ref[i * N + j]));
            if (err_r > max_err) max_err = err_r;
            if (err_i > max_err) max_err = err_i;
        }
    }

    ab_matrix_destroy(mAr); ab_matrix_destroy(mAi);
    ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);
    free(Ar); free(Ai); free(Cr_gpu); free(Ci_gpu);
    free(A_ref); free(C_ref);
    ab_shutdown();

    if (max_err < 1e-10) PASS();
    else { char buf[64]; snprintf(buf, sizeof(buf), "max_err=%.2e", max_err); FAIL(buf); }
}

static void test_zherk_hermitian_symmetry(void) {
    TEST("ZHERK imaginary part is anti-symmetric");
    // DSYRK only fills the upper triangle of Cr, so we can't check
    // full Cr symmetry. But the imaginary part (Ci = Ai×ArT - Ar×AiT)
    // is computed via full-matrix DGEMM, so Ci[i,j] should equal -Ci[j,i].
    // We also verify Ci diagonal is near-zero (Hermitian property).
    ab_init();
    int N = 128, K = 64;
    size_t countA = (size_t)N * K;
    size_t countC = (size_t)N * N;

    double* Ar = malloc(countA * sizeof(double));
    double* Ai = malloc(countA * sizeof(double));
    double* Ci = malloc(countC * sizeof(double));

    srand48(33);
    for (size_t i = 0; i < countA; i++) {
        Ar[i] = drand48() * 2 - 1;
        Ai[i] = drand48() * 2 - 1;
    }

    ABMatrix mAr = ab_matrix_create(N, K);
    ABMatrix mAi = ab_matrix_create(N, K);
    ABMatrix mCr = ab_matrix_create(N, N);
    ABMatrix mCi = ab_matrix_create(N, N);
    ab_matrix_upload(mAr, Ar, true);
    ab_matrix_upload(mAi, Ai, true);
    ab_zherk(mAr, mAi, mCr, mCi);
    ab_matrix_download(mCi, Ci, true);

    // Check anti-symmetry of Ci: Ci[i,j] == -Ci[j,i]
    double max_asym = 0;
    double max_diag_imag = 0;
    for (int i = 0; i < N; i++) {
        double di = fabs(Ci[i * N + i]);
        if (di > max_diag_imag) max_diag_imag = di;
        for (int j = i + 1; j < N; j++) {
            double err_i = fabs(Ci[i * N + j] + Ci[j * N + i]);
            if (err_i > max_asym) max_asym = err_i;
        }
    }

    ab_matrix_destroy(mAr); ab_matrix_destroy(mAi);
    ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);
    free(Ar); free(Ai); free(Ci);
    ab_shutdown();

    if (max_asym < 1e-10 && max_diag_imag < 1e-10) PASS();
    else {
        char buf[64];
        snprintf(buf, sizeof(buf), "asym=%.2e diag=%.2e", max_asym, max_diag_imag);
        FAIL(buf);
    }
}

// =============================================================================
// 5. Concurrent Stress Test
// =============================================================================

#define STRESS_THREADS 4
#define STRESS_ITERS   20

typedef struct {
    int thread_id;
    int passed;
    char error_msg[128];
} ThreadResult;

static void* stress_worker(void* arg) {
    ThreadResult* res = (ThreadResult*)arg;
    res->passed = 1;
    res->error_msg[0] = '\0';

    int N = 64;
    double* A = malloc(N * N * sizeof(double));
    double* B = malloc(N * N * sizeof(double));
    double* C = malloc(N * N * sizeof(double));

    // Each thread uses different seed for different data
    srand48(res->thread_id * 1000 + 42);
    for (int i = 0; i < N * N; i++) {
        A[i] = drand48();
        B[i] = drand48();
    }

    for (int iter = 0; iter < STRESS_ITERS; iter++) {
        ABMatrix mA = ab_matrix_create(N, N);
        ABMatrix mB = ab_matrix_create(N, N);
        ABMatrix mC = ab_matrix_create(N, N);

        if (!mA || !mB || !mC) {
            snprintf(res->error_msg, sizeof(res->error_msg),
                     "thread %d: alloc failed iter %d", res->thread_id, iter);
            res->passed = 0;
            ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
            break;
        }

        ab_matrix_upload(mA, A, true);
        ab_matrix_upload(mB, B, true);
        ABStatus s = ab_dgemm(mA, mB, mC);
        if (s != AB_OK) {
            snprintf(res->error_msg, sizeof(res->error_msg),
                     "thread %d: dgemm failed iter %d (status=%d)", res->thread_id, iter, s);
            res->passed = 0;
            ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
            break;
        }

        ab_matrix_download(mC, C, true);

        // Sanity check: no NaN/Inf
        for (int i = 0; i < N * N; i++) {
            if (isnan(C[i]) || isinf(C[i])) {
                snprintf(res->error_msg, sizeof(res->error_msg),
                         "thread %d: NaN/Inf at iter %d elem %d", res->thread_id, iter, i);
                res->passed = 0;
                break;
            }
        }
        if (!res->passed) {
            ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
            break;
        }

        ab_matrix_destroy(mA);
        ab_matrix_destroy(mB);
        ab_matrix_destroy(mC);
    }

    free(A); free(B); free(C);
    return NULL;
}

static void test_concurrent_dgemm_stress(void) {
    TEST("Concurrent DGEMM (4 threads × 20 iters)");
    ab_init();

    pthread_t threads[STRESS_THREADS];
    ThreadResult results[STRESS_THREADS];

    for (int i = 0; i < STRESS_THREADS; i++) {
        results[i].thread_id = i;
        pthread_create(&threads[i], NULL, stress_worker, &results[i]);
    }

    int all_passed = 1;
    char first_error[128] = "";
    for (int i = 0; i < STRESS_THREADS; i++) {
        pthread_join(threads[i], NULL);
        if (!results[i].passed) {
            all_passed = 0;
            if (first_error[0] == '\0')
                strncpy(first_error, results[i].error_msg, sizeof(first_error) - 1);
        }
    }

    ab_shutdown();
    if (all_passed) PASS(); else FAIL(first_error);
}

// =============================================================================
// 6. Memory Pool Exhaustion
// =============================================================================

static void test_pool_exhaustion_recovery(void) {
    TEST("Pool exhaustion returns NULL gracefully");
    ab_init();
    ABMemoryPool pool = ab_pool_create(0);

    int null_count = 0;
    // Try to allocate 130 matrices (pool max is 128)
    for (int i = 0; i < 130; i++) {
        ABMatrix m = ab_pool_get_matrix(pool, 64, 64);
        if (!m) null_count++;
    }

    ab_pool_destroy(pool);
    ab_shutdown();

    // Should get NULL for the last 2 (or more if capacity reached)
    if (null_count >= 2) PASS();
    else { char buf[64]; snprintf(buf, sizeof(buf), "only %d NULLs", null_count); FAIL(buf); }
}

static void test_pool_reset_and_reuse(void) {
    TEST("Pool reset enables full reuse cycle");
    ab_init();
    ABMemoryPool pool = ab_pool_create(0);

    // Fill pool
    for (int i = 0; i < 128; i++) {
        ABMatrix m = ab_pool_get_matrix(pool, 32, 32);
        if (!m) { FAIL("premature NULL"); ab_pool_destroy(pool); ab_shutdown(); return; }
    }

    // Should fail now
    ABMatrix overflow = ab_pool_get_matrix(pool, 32, 32);
    if (overflow != NULL) { FAIL("should be NULL"); ab_pool_destroy(pool); ab_shutdown(); return; }

    // Reset and try again
    ab_pool_reset(pool);
    int reused = 0;
    for (int i = 0; i < 128; i++) {
        ABMatrix m = ab_pool_get_matrix(pool, 32, 32);
        if (m) reused++;
    }

    ab_pool_destroy(pool);
    ab_shutdown();
    if (reused == 128) PASS();
    else { char buf[64]; snprintf(buf, sizeof(buf), "only reused %d/128", reused); FAIL(buf); }
}

// =============================================================================
// 7. ZGEMM Edge Cases
// =============================================================================

static void test_zgemm_purely_real(void) {
    TEST("ZGEMM with zero imaginary (reduces to DGEMM)");
    ab_init();
    int N = 128;
    double* Ar = malloc(N * N * sizeof(double));
    double* Br = malloc(N * N * sizeof(double));
    double* Ai = calloc(N * N, sizeof(double));  // All zero
    double* Bi = calloc(N * N, sizeof(double));  // All zero
    double* Cr = malloc(N * N * sizeof(double));
    double* Ci = malloc(N * N * sizeof(double));
    double* C_ref = malloc(N * N * sizeof(double));

    srand48(123);
    for (int i = 0; i < N * N; i++) {
        Ar[i] = drand48() * 2 - 1;
        Br[i] = drand48() * 2 - 1;
    }

    ABMatrix mAr = ab_matrix_create(N, N);
    ABMatrix mAi = ab_matrix_create(N, N);
    ABMatrix mBr = ab_matrix_create(N, N);
    ABMatrix mBi = ab_matrix_create(N, N);
    ABMatrix mCr = ab_matrix_create(N, N);
    ABMatrix mCi = ab_matrix_create(N, N);
    ab_matrix_upload(mAr, Ar, true);
    ab_matrix_upload(mAi, Ai, true);
    ab_matrix_upload(mBr, Br, true);
    ab_matrix_upload(mBi, Bi, true);
    ab_zgemm(mAr, mAi, mBr, mBi, mCr, mCi);
    ab_matrix_download(mCr, Cr, true);
    ab_matrix_download(mCi, Ci, true);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0, Ar, N, Br, N, 0.0, C_ref, N);

    double err_real = frobenius_rel_error(Cr, C_ref, N);

    // Imaginary part should be zero
    double max_imag = 0;
    for (int i = 0; i < N * N; i++) {
        if (fabs(Ci[i]) > max_imag) max_imag = fabs(Ci[i]);
    }

    ab_matrix_destroy(mAr); ab_matrix_destroy(mAi);
    ab_matrix_destroy(mBr); ab_matrix_destroy(mBi);
    ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);
    free(Ar); free(Br); free(Ai); free(Bi); free(Cr); free(Ci); free(C_ref);
    ab_shutdown();

    if (err_real < 1e-14 && max_imag < 1e-14) PASS();
    else {
        char buf[64]; snprintf(buf, sizeof(buf), "real_err=%.2e imag=%.2e", err_real, max_imag);
        FAIL(buf);
    }
}

static void test_zgemm_purely_imaginary(void) {
    TEST("ZGEMM with zero real (purely imaginary)");
    ab_init();
    int N = 64;
    double* Ar = calloc(N * N, sizeof(double));  // All zero
    double* Br = calloc(N * N, sizeof(double));  // All zero
    double* Ai = malloc(N * N * sizeof(double));
    double* Bi = malloc(N * N * sizeof(double));
    double* Cr = malloc(N * N * sizeof(double));
    double* Ci = malloc(N * N * sizeof(double));

    srand48(456);
    for (int i = 0; i < N * N; i++) {
        Ai[i] = drand48() * 2 - 1;
        Bi[i] = drand48() * 2 - 1;
    }

    // (0+ai)(0+bi) = -ai*bi + 0i
    // So Cr = -Ai*Bi, Ci = 0

    ABMatrix mAr = ab_matrix_create(N, N);
    ABMatrix mAi = ab_matrix_create(N, N);
    ABMatrix mBr = ab_matrix_create(N, N);
    ABMatrix mBi = ab_matrix_create(N, N);
    ABMatrix mCr = ab_matrix_create(N, N);
    ABMatrix mCi = ab_matrix_create(N, N);
    ab_matrix_upload(mAr, Ar, true);
    ab_matrix_upload(mAi, Ai, true);
    ab_matrix_upload(mBr, Br, true);
    ab_matrix_upload(mBi, Bi, true);
    ab_zgemm(mAr, mAi, mBr, mBi, mCr, mCi);
    ab_matrix_download(mCr, Cr, true);
    ab_matrix_download(mCi, Ci, true);

    // Reference: Cr = -Ai*Bi
    double* ref = malloc(N * N * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, -1.0, Ai, N, Bi, N, 0.0, ref, N);

    double err_real = frobenius_rel_error(Cr, ref, N);
    double max_imag = 0;
    for (int i = 0; i < N * N; i++) {
        if (fabs(Ci[i]) > max_imag) max_imag = fabs(Ci[i]);
    }

    ab_matrix_destroy(mAr); ab_matrix_destroy(mAi);
    ab_matrix_destroy(mBr); ab_matrix_destroy(mBi);
    ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);
    free(Ar); free(Br); free(Ai); free(Bi); free(Cr); free(Ci); free(ref);
    ab_shutdown();

    if (err_real < 1e-14 && max_imag < 1e-14) PASS();
    else {
        char buf[64]; snprintf(buf, sizeof(buf), "real_err=%.2e imag=%.2e", err_real, max_imag);
        FAIL(buf);
    }
}

// =============================================================================
// Main
// =============================================================================

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  apple-bottom Adversarial & System-Level Test Suite              ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    printf("IEEE 754 Edge Cases:\n");
    test_dgemm_nan_input();
    test_dgemm_inf_input();
    test_dgemm_scaled_beta_zero_suppresses_nan();
    test_dgemm_subnormal_input();
    test_dgemm_zero_matrix();

    printf("\nIll-Conditioned Matrices:\n");
    test_dgemm_high_condition_number();
    test_dgemm_large_dynamic_range();

    printf("\nBoundary Dimensions:\n");
    test_dgemm_1x1();
    test_dgemm_prime_dimensions();
    test_zgemm_1x1();

    printf("\nZHERK GPU Transpose:\n");
    test_zherk_vs_cblas();
    test_zherk_rectangular();
    test_zherk_hermitian_symmetry();

    printf("\nConcurrency Stress:\n");
    test_concurrent_dgemm_stress();

    printf("\nMemory Pool:\n");
    test_pool_exhaustion_recovery();
    test_pool_reset_and_reuse();

    printf("\nZGEMM Edge Cases:\n");
    test_zgemm_purely_real();
    test_zgemm_purely_imaginary();

    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    if (tests_failed == 0) {
        printf("✓ All %d adversarial tests PASSED\n", tests_passed);
        return 0;
    } else {
        printf("✗ Some tests FAILED\n");
        return 1;
    }
}
