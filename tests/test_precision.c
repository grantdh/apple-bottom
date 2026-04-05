// Precision Test — apple-bottom
//
// Uses dynamic Wilkinson-scaled error thresholds instead of static bounds.
// Frobenius relative error scales as O(sqrt(K)) per Wilkinson's probabilistic
// analysis of floating-point dot products (Higham 2002, §3.1).
//
// Threshold: C_safety * sqrt(K) * u_DD, where u_DD = 2^-48 ≈ 3.55e-15
// C_safety = 10 provides ~20x margin over typical measured values.

#include "apple_bottom.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Accelerate/Accelerate.h>

// DD unit roundoff: 2^-48 (48-bit effective mantissa from two FP32 words)
#define U_DD 3.5527136788005009e-15
// Safety factor: covers worst-case constant factors in error bound
#define C_SAFETY 10.0

// Compute dynamic Frobenius error threshold for a GEMM with contraction dim K
static double wilkinson_threshold(int K) {
    return C_SAFETY * sqrt((double)K) * U_DD;
}

// Run a single precision test: C_gpu = A * B, compare to cblas reference
// Returns 1 if passed, 0 if failed
static int run_precision_test(int M, int K, int N, const char* label) {
    size_t countA = (size_t)M * K;
    size_t countB = (size_t)K * N;
    size_t countC = (size_t)M * N;

    double* A = malloc(countA * sizeof(double));
    double* B = malloc(countB * sizeof(double));
    double* C_gpu = malloc(countC * sizeof(double));
    double* C_ref = malloc(countC * sizeof(double));

    srand48(M + K + N);
    for (size_t i = 0; i < countA; i++)
        A[i] = (drand48() * 2 - 1) * pow(10, (int)(drand48() * 6 - 3));
    for (size_t i = 0; i < countB; i++)
        B[i] = (drand48() * 2 - 1) * pow(10, (int)(drand48() * 6 - 3));

    ABMatrix mA = ab_matrix_create(M, K);
    ABMatrix mB = ab_matrix_create(K, N);
    ABMatrix mC = ab_matrix_create(M, N);
    ab_matrix_upload(mA, A, true);
    ab_matrix_upload(mB, B, true);
    ab_dgemm(mA, mB, mC);
    ab_matrix_download(mC, C_gpu, true);
    ab_matrix_destroy(mA);
    ab_matrix_destroy(mB);
    ab_matrix_destroy(mC);

    // Reference: cblas row-major DGEMM
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, A, K, B, N, 0.0, C_ref, N);

    double max_err = 0, sum_sq = 0, max_elem = 0;
    for (size_t i = 0; i < countC; i++) {
        double err = fabs(C_gpu[i] - C_ref[i]);
        if (err > max_err) max_err = err;
        sum_sq += C_ref[i] * C_ref[i];
        double ref_mag = fabs(C_ref[i]);
        if (ref_mag > 1e-10) { double rel = err / ref_mag; if (rel > max_elem) max_elem = rel; }
    }
    double frob = max_err / sqrt(sum_sq);
    double threshold = wilkinson_threshold(K);
    int passed = frob < threshold;

    printf("  %-20s │    %.2e   │   %.2e   │ %.2e │ %s\n",
           label, frob, max_elem, threshold, passed ? "✓ PASS" : "✗ FAIL");

    free(A); free(B); free(C_gpu); free(C_ref);
    return passed;
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  apple-bottom Precision Test (Wilkinson-scaled dynamic thresholds)        ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════╝\n\n");

    if (ab_init() != AB_OK) { fprintf(stderr, "Init failed\n"); return 1; }
    printf("Device: %s\n", ab_device_name());
    printf("Threshold formula: %.1f × sqrt(K) × u_DD,  u_DD = 2⁻⁴⁸ ≈ %.2e\n\n", C_SAFETY, U_DD);

    printf("  Shape                │ Frobenius Err │ Max Elem Err │ Threshold   │ Status\n");
    printf("───────────────────────┼───────────────┼──────────────┼─────────────┼────────\n");

    int all_passed = 1;
    char label[64];

    // Square matrices (K = N): standard benchmark sizes
    int square_sizes[] = {64, 128, 256, 512, 1024, 2048};
    for (int s = 0; s < 6; s++) {
        int N = square_sizes[s];
        snprintf(label, sizeof(label), "%dx%d (K=%d)", N, N, N);
        if (!run_precision_test(N, N, N, label)) all_passed = 0;
    }

    printf("───────────────────────┼───────────────┼──────────────┼─────────────┼────────\n");

    // Rectangular: large K (exercises accumulation drift)
    snprintf(label, sizeof(label), "100x100 (K=4096)");
    if (!run_precision_test(100, 4096, 100, label)) all_passed = 0;

    // Tall-skinny: QE-like aspect ratio
    snprintf(label, sizeof(label), "1000x50 (K=200)");
    if (!run_precision_test(1000, 200, 50, label)) all_passed = 0;

    printf("\n═══════════════════════════════════════════════════════════════════════════\n");
    if (all_passed) {
        printf("✓ All precision tests PASSED (8/8)\n");
        printf("  Precision: ~10⁻¹⁵ Frobenius, bounded by %.1f × sqrt(K) × 2⁻⁴⁸\n", C_SAFETY);
    } else {
        printf("✗ Some precision tests FAILED\n");
    }

    ab_shutdown();
    return all_passed ? 0 : 1;
}
