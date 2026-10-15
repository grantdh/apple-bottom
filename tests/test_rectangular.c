// =============================================================================
// test_rectangular.c - Rectangular Matrix Tests for apple-bottom
// =============================================================================
// Tests correctness and performance of rectangular matrices, including
// QE-like tall-skinny matrices (M >> N).
//
// Compile:
//   clang -O3 -I../include -L../build -lapplebottom \
//     -framework Accelerate -framework Metal -framework Foundation \
//     -o test_rectangular test_rectangular.c
//
// Run:
//   ./test_rectangular
// =============================================================================

#include "apple_bottom.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <Accelerate/Accelerate.h>
#include <sys/time.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  %-50s ", name)
#define PASS() do { printf("✓ PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("✗ FAIL: %s\n", msg); tests_failed++; } while(0)

// =============================================================================
// Timing Utilities
// =============================================================================

static double get_time_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
}

// =============================================================================
// Validation Helpers
// =============================================================================

// Componentwise error: |got - ref| / (|op(A)| |B|) per element of C. A dot
// product's rounding error is relative to the sum of |a_k b_k|, not to the
// result; where the sum cancels, |ref| is far below its terms, so the
// elementwise |got - ref| / |ref| says nothing about accuracy.
// absA is M x K (or K x M with transA), absB is K x N, both row-major.
static double componentwise_error(const double* got_re, const double* got_im,
                                  const double* ref_re, const double* ref_im,
                                  const double* absA, const double* absB,
                                  bool transA, int M, int N, int K) {
    double* scale = (double*)malloc((size_t)M * N * sizeof(double));
    cblas_dgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0, absA, transA ? M : K, absB, N, 0.0, scale, N);
    double worst = 0.0;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        double d = fabs(got_re[i] - ref_re[i]);
        if (got_im) d = hypot(d, got_im[i] - ref_im[i]);
        if (!isfinite(d)) { worst = INFINITY; break; }       // NaN or Inf in the result
        if (scale[i] > 0.0) { if (d / scale[i] > worst) worst = d / scale[i]; }
        else if (d != 0.0) { worst = INFINITY; break; }      // nonzero where |A||B| is 0
    }
    free(scale);
    return worst;
}

static double real_error(const double* A, const double* B, const double* got,
                         const double* ref, int M, int N, int K) {
    double* absA = (double*)malloc((size_t)M * K * sizeof(double));
    double* absB = (double*)malloc((size_t)K * N * sizeof(double));
    for (size_t i = 0; i < (size_t)M * K; i++) absA[i] = fabs(A[i]);
    for (size_t i = 0; i < (size_t)K * N; i++) absB[i] = fabs(B[i]);
    double e = componentwise_error(got, NULL, ref, NULL, absA, absB, false, M, N, K);
    free(absA); free(absB);
    return e;
}

// A has a_count elements (M x K, or K x M with transA).
static double complex_error(const double complex* A, const double complex* B,
                            const double complex* got, const double complex* ref,
                            bool transA, int M, int N, int K) {
    size_t a_count = (size_t)M * K, b_count = (size_t)K * N, c_count = (size_t)M * N;
    double* absA = (double*)malloc(a_count * sizeof(double));
    double* absB = (double*)malloc(b_count * sizeof(double));
    double* g = (double*)malloc(4 * c_count * sizeof(double));
    for (size_t i = 0; i < a_count; i++) absA[i] = cabs(A[i]);
    for (size_t i = 0; i < b_count; i++) absB[i] = cabs(B[i]);
    for (size_t i = 0; i < c_count; i++) {
        g[i] = creal(got[i]); g[c_count + i] = cimag(got[i]);
        g[2 * c_count + i] = creal(ref[i]); g[3 * c_count + i] = cimag(ref[i]);
    }
    double e = componentwise_error(g, g + c_count, g + 2 * c_count, g + 3 * c_count,
                                   absA, absB, transA, M, N, K);
    free(absA); free(absB); free(g);
    return e;
}

// Accumulated DD rounding over K terms with margin, 10 K u^2 (u^2 = 2^-48),
// with a floor for short sums: the bound of test_rectangular_diag.c.
static double dd_bound(int K) {
    double b = 10.0 * K * 3.55e-15;
    return b > 3.55e-14 ? b : 3.55e-14;
}

static void check(double err, int K) {
    printf("(err %.1e, bound %.1e) ", err, dd_bound(K));
    if (err <= dd_bound(K)) PASS();
    else FAIL("componentwise error above the DD bound");
}

// =============================================================================
// DGEMM Rectangular Tests
// =============================================================================

static void test_dgemm_tall_skinny(void) {
    TEST("DGEMM tall-skinny (10000 × 100 × 100 × 100)");
    ab_init();

    int M = 10000, N = 100, K = 100;
    double* A = (double*)malloc(M * K * sizeof(double));
    double* B = (double*)malloc(K * N * sizeof(double));
    double* C_gpu = (double*)calloc(M * N, sizeof(double));
    double* C_ref = (double*)calloc(M * N, sizeof(double));

    // Initialize
    srand48(42);
    for (int i = 0; i < M * K; i++) A[i] = drand48() - 0.5;
    for (int i = 0; i < K * N; i++) B[i] = drand48() - 0.5;

    // Reference (BLAS)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0, A, K, B, N, 0.0, C_ref, N);

    // GPU
    ABMatrix mA = ab_matrix_create(M, K);
    ABMatrix mB = ab_matrix_create(K, N);
    ABMatrix mC = ab_matrix_create(M, N);
    ab_matrix_upload(mA, A, true);
    ab_matrix_upload(mB, B, true);
    ab_dgemm(mA, mB, mC);
    ab_matrix_download(mC, C_gpu, true);

    // Validate
    double err = real_error(A, B, C_gpu, C_ref, M, N, K);

    ab_matrix_destroy(mA);
    ab_matrix_destroy(mB);
    ab_matrix_destroy(mC);
    free(A); free(B); free(C_gpu); free(C_ref);
    ab_shutdown();

    check(err, K);
}

static void test_dgemm_short_wide(void) {
    TEST("DGEMM short-wide (100 × 10000 × 100 × 10000)");
    ab_init();

    int M = 100, N = 10000, K = 100;
    double* A = (double*)malloc(M * K * sizeof(double));
    double* B = (double*)malloc(K * N * sizeof(double));
    double* C_gpu = (double*)calloc(M * N, sizeof(double));
    double* C_ref = (double*)calloc(M * N, sizeof(double));

    srand48(43);
    for (int i = 0; i < M * K; i++) A[i] = drand48() - 0.5;
    for (int i = 0; i < K * N; i++) B[i] = drand48() - 0.5;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0, A, K, B, N, 0.0, C_ref, N);

    ABMatrix mA = ab_matrix_create(M, K);
    ABMatrix mB = ab_matrix_create(K, N);
    ABMatrix mC = ab_matrix_create(M, N);
    ab_matrix_upload(mA, A, true);
    ab_matrix_upload(mB, B, true);
    ab_dgemm(mA, mB, mC);
    ab_matrix_download(mC, C_gpu, true);

    double err = real_error(A, B, C_gpu, C_ref, M, N, K);

    ab_matrix_destroy(mA);
    ab_matrix_destroy(mB);
    ab_matrix_destroy(mC);
    free(A); free(B); free(C_gpu); free(C_ref);
    ab_shutdown();

    check(err, K);
}

static void test_dgemm_qe_dimensions(void) {
    TEST("DGEMM QE-like (18277 × 150 × 18277 × 150)");
    ab_init();

    // QE-like aspect (M, N as in a Davidson update); K is synthetic
    int M = 18277, N = 150, K = 18277;

    double* A = (double*)malloc(M * K * sizeof(double));
    double* B = (double*)malloc(K * N * sizeof(double));
    double* C_gpu = (double*)calloc(M * N, sizeof(double));
    double* C_ref = (double*)calloc(M * N, sizeof(double));

    srand48(44);
    for (int i = 0; i < M * K; i++) A[i] = drand48() - 0.5;
    for (int i = 0; i < K * N; i++) B[i] = drand48() - 0.5;

    // Reference
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0, A, K, B, N, 0.0, C_ref, N);

    // GPU
    ABMatrix mA = ab_matrix_create(M, K);
    ABMatrix mB = ab_matrix_create(K, N);
    ABMatrix mC = ab_matrix_create(M, N);
    ab_matrix_upload(mA, A, true);
    ab_matrix_upload(mB, B, true);
    ab_dgemm(mA, mB, mC);
    ab_matrix_download(mC, C_gpu, true);

    double err = real_error(A, B, C_gpu, C_ref, M, N, K);

    ab_matrix_destroy(mA);
    ab_matrix_destroy(mB);
    ab_matrix_destroy(mC);
    free(A); free(B); free(C_gpu); free(C_ref);
    ab_shutdown();

    check(err, K);
}

static void test_dgemm_thin_middle(void) {
    TEST("DGEMM thin middle (5000 × 5000 × 10 × 5000)");
    ab_init();

    // M=5000, N=5000, K=10 (very thin K dimension)
    int M = 5000, N = 5000, K = 10;

    double* A = (double*)malloc(M * K * sizeof(double));
    double* B = (double*)malloc(K * N * sizeof(double));
    double* C_gpu = (double*)calloc(M * N, sizeof(double));
    double* C_ref = (double*)calloc(M * N, sizeof(double));

    srand48(45);
    for (int i = 0; i < M * K; i++) A[i] = drand48() - 0.5;
    for (int i = 0; i < K * N; i++) B[i] = drand48() - 0.5;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0, A, K, B, N, 0.0, C_ref, N);

    ABMatrix mA = ab_matrix_create(M, K);
    ABMatrix mB = ab_matrix_create(K, N);
    ABMatrix mC = ab_matrix_create(M, N);
    ab_matrix_upload(mA, A, true);
    ab_matrix_upload(mB, B, true);
    ab_dgemm(mA, mB, mC);
    ab_matrix_download(mC, C_gpu, true);

    double err = real_error(A, B, C_gpu, C_ref, M, N, K);

    ab_matrix_destroy(mA);
    ab_matrix_destroy(mB);
    ab_matrix_destroy(mC);
    free(A); free(B); free(C_gpu); free(C_ref);
    ab_shutdown();

    check(err, K);
}

// =============================================================================
// ZGEMM Rectangular Tests
// =============================================================================

static void test_zgemm_qe_dimensions(void) {
    TEST("ZGEMM QE-like (18277 × 150 × 18277 × 150)");
    ab_init();

    int M = 18277, N = 150, K = 18277;

    double complex* A = (double complex*)malloc(M * K * sizeof(double complex));
    double complex* B = (double complex*)malloc(K * N * sizeof(double complex));
    double complex* C_gpu = (double complex*)calloc(M * N, sizeof(double complex));
    double complex* C_ref = (double complex*)calloc(M * N, sizeof(double complex));

    srand48(46);
    for (int i = 0; i < M * K; i++) {
        A[i] = (drand48() - 0.5) + I * (drand48() - 0.5);
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = (drand48() - 0.5) + I * (drand48() - 0.5);
    }

    // Reference
    double complex alpha = 1.0 + 0.0*I;
    double complex beta = 0.0 + 0.0*I;
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, &alpha, A, K, B, N, &beta, C_ref, N);

    // GPU - split into real/imag
    double* Ar = (double*)malloc(M * K * sizeof(double));
    double* Ai = (double*)malloc(M * K * sizeof(double));
    double* Br = (double*)malloc(K * N * sizeof(double));
    double* Bi = (double*)malloc(K * N * sizeof(double));
    double* Cr = (double*)calloc(M * N, sizeof(double));
    double* Ci = (double*)calloc(M * N, sizeof(double));

    for (int i = 0; i < M * K; i++) {
        Ar[i] = creal(A[i]);
        Ai[i] = cimag(A[i]);
    }
    for (int i = 0; i < K * N; i++) {
        Br[i] = creal(B[i]);
        Bi[i] = cimag(B[i]);
    }

    ABMatrix mAr = ab_matrix_create(M, K);
    ABMatrix mAi = ab_matrix_create(M, K);
    ABMatrix mBr = ab_matrix_create(K, N);
    ABMatrix mBi = ab_matrix_create(K, N);
    ABMatrix mCr = ab_matrix_create(M, N);
    ABMatrix mCi = ab_matrix_create(M, N);

    ab_matrix_upload(mAr, Ar, true);
    ab_matrix_upload(mAi, Ai, true);
    ab_matrix_upload(mBr, Br, true);
    ab_matrix_upload(mBi, Bi, true);

    ab_zgemm(mAr, mAi, mBr, mBi, mCr, mCi);

    ab_matrix_download(mCr, Cr, true);
    ab_matrix_download(mCi, Ci, true);

    // Reconstruct complex
    for (int i = 0; i < M * N; i++) {
        C_gpu[i] = Cr[i] + I * Ci[i];
    }

    double err = complex_error(A, B, C_gpu, C_ref, false, M, N, K);

    ab_matrix_destroy(mAr); ab_matrix_destroy(mAi);
    ab_matrix_destroy(mBr); ab_matrix_destroy(mBi);
    ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);
    free(A); free(B); free(C_gpu); free(C_ref);
    free(Ar); free(Ai); free(Br); free(Bi); free(Cr); free(Ci);
    ab_shutdown();

    check(err, K);
}

static void test_zgemm_conjugate_transpose_qe(void) {
    TEST("ZGEMM QE conjugate transpose (150 × 150 × 18277 × 150)");
    ab_init();

    // Typical QE: overlap = psi^H * psi_new
    // psi: 18277 × 150, psi_new: 18277 × 150
    // Result: 150 × 150
    int M = 150, N = 150, K = 18277;

    double complex* A = (double complex*)malloc(K * M * sizeof(double complex));
    double complex* B = (double complex*)malloc(K * N * sizeof(double complex));
    double complex* C_gpu = (double complex*)calloc(M * N, sizeof(double complex));
    double complex* C_ref = (double complex*)calloc(M * N, sizeof(double complex));

    srand48(47);
    for (int i = 0; i < K * M; i++) {
        A[i] = (drand48() - 0.5) + I * (drand48() - 0.5);
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = (drand48() - 0.5) + I * (drand48() - 0.5);
    }

    // Reference: C = A^H * B
    double complex alpha = 1.0 + 0.0*I;
    double complex beta = 0.0 + 0.0*I;
    cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasNoTrans,
                M, N, K, &alpha, A, M, B, N, &beta, C_ref, N);

    // GPU: A is stored K x M, ab_zgemm_ex applies the conjugate transpose
    size_t a_count = (size_t)K * M, b_count = (size_t)K * N, c_count = (size_t)M * N;
    double *Ar = malloc(a_count * sizeof(double)), *Ai = malloc(a_count * sizeof(double));
    double *Br = malloc(b_count * sizeof(double)), *Bi = malloc(b_count * sizeof(double));
    double *Cr = malloc(c_count * sizeof(double)), *Ci = malloc(c_count * sizeof(double));
    for (size_t i = 0; i < a_count; i++) { Ar[i] = creal(A[i]); Ai[i] = cimag(A[i]); }
    for (size_t i = 0; i < b_count; i++) { Br[i] = creal(B[i]); Bi[i] = cimag(B[i]); }
    ABMatrix mAr = ab_matrix_create(K, M), mAi = ab_matrix_create(K, M);
    ABMatrix mBr = ab_matrix_create(K, N), mBi = ab_matrix_create(K, N);
    ABMatrix mCr = ab_matrix_create(M, N), mCi = ab_matrix_create(M, N);
    ab_matrix_upload(mAr, Ar, true); ab_matrix_upload(mAi, Ai, true);
    ab_matrix_upload(mBr, Br, true); ab_matrix_upload(mBi, Bi, true);
    ABStatus st = ab_zgemm_ex(AB_CONJ_TRANS, AB_NO_TRANS, mAr, mAi, mBr, mBi, mCr, mCi);
    ab_matrix_download(mCr, Cr, true);
    ab_matrix_download(mCi, Ci, true);
    for (size_t i = 0; i < c_count; i++) C_gpu[i] = Cr[i] + I * Ci[i];
    double err = complex_error(A, B, C_gpu, C_ref, true, M, N, K);

    ab_matrix_destroy(mAr); ab_matrix_destroy(mAi);
    ab_matrix_destroy(mBr); ab_matrix_destroy(mBi);
    ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);
    free(Ar); free(Ai); free(Br); free(Bi); free(Cr); free(Ci);
    free(A); free(B); free(C_gpu); free(C_ref);
    ab_shutdown();

    if (st != AB_OK) FAIL("ab_zgemm_ex failed");
    else check(err, K);
}

// =============================================================================
// Performance Benchmarks
// =============================================================================

static void benchmark_aspect_ratios(void) {
    printf("\n");
    printf("=============================================================================\n");
    printf("Performance Benchmark: Aspect Ratios\n");
    printf("=============================================================================\n");
    printf("Testing different M:N ratios with fixed total FLOPs\n");
    printf("\n");
    printf("%-20s %-15s %-15s %-15s\n", "Dimensions", "GPU (ms)", "BLAS (ms)", "Speedup");
    printf("-----------------------------------------------------------------------------\n");

    ab_init();

    // Fixed total elements ~= 2048^3 FLOPs
    struct {
        int M, N, K;
        const char* name;
    } cases[] = {
        {2048, 2048, 2048, "Square"},
        {4096, 1024, 2048, "Tall 4:1"},
        {8192, 512, 2048, "Tall 16:1"},
        {16384, 256, 2048, "Tall 64:1"},
        {18277, 150, 18277, "QE-like"},
        {1024, 4096, 2048, "Wide 1:4"},
        {512, 8192, 2048, "Wide 1:16"},
    };

    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        int M = cases[c].M;
        int N = cases[c].N;
        int K = cases[c].K;

        double* A = (double*)malloc(M * K * sizeof(double));
        double* B = (double*)malloc(K * N * sizeof(double));
        double* C_gpu = (double*)malloc(M * N * sizeof(double));
        double* C_blas = (double*)malloc(M * N * sizeof(double));

        srand48(100 + c);
        for (int i = 0; i < M * K; i++) A[i] = drand48();
        for (int i = 0; i < K * N; i++) B[i] = drand48();

        // GPU
        ABMatrix mA = ab_matrix_create(M, K);
        ABMatrix mB = ab_matrix_create(K, N);
        ABMatrix mC = ab_matrix_create(M, N);
        ab_matrix_upload(mA, A, false);
        ab_matrix_upload(mB, B, false);

        double t0 = get_time_sec();
        ab_dgemm(mA, mB, mC);
        double t_gpu = (get_time_sec() - t0) * 1000.0;

        ab_matrix_download(mC, C_gpu, false);

        // BLAS
        t0 = get_time_sec();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0, A, K, B, N, 0.0, C_blas, N);
        double t_blas = (get_time_sec() - t0) * 1000.0;

        printf("%-20s %12.2f    %12.2f    %12.2fx\n",
               cases[c].name, t_gpu, t_blas, t_blas / t_gpu);

        ab_matrix_destroy(mA);
        ab_matrix_destroy(mB);
        ab_matrix_destroy(mC);
        free(A); free(B); free(C_gpu); free(C_blas);
    }

    ab_shutdown();
    printf("\n");
}

// =============================================================================
// Main
// =============================================================================

int main(void) {
    printf("=============================================================================\n");
    printf("apple-bottom Rectangular Matrix Test Suite\n");
    printf("=============================================================================\n");
    printf("\n");

    printf("DGEMM Rectangular Tests:\n");
    test_dgemm_tall_skinny();
    test_dgemm_short_wide();
    test_dgemm_qe_dimensions();
    test_dgemm_thin_middle();

    printf("\n");
    printf("ZGEMM Rectangular Tests:\n");
    test_zgemm_qe_dimensions();
    test_zgemm_conjugate_transpose_qe();

    printf("\n");
    printf("=============================================================================\n");
    printf("Summary: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("=============================================================================\n");

    // Run performance benchmarks
    benchmark_aspect_ratios();

    return (tests_failed == 0) ? 0 : 1;
}
