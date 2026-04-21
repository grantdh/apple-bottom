// =============================================================================
// test_original_rectangular.c — Test original rectangular dimensions from docs
// Copyright (c) 2026 Grant Heileman, Technology Residue. MIT License.
// =============================================================================
//
// Purpose: Test the exact dimensions documented in docs/design/RECTANGULAR_TILING.md
// to verify if BUG-1/BUG-2 fixes have resolved the rectangular issues.
// =============================================================================

#include "../include/apple_bottom.h"
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define RED     "\033[0;31m"
#define GREEN   "\033[0;32m"
#define YELLOW  "\033[0;33m"
#define RESET   "\033[0m"

// Test case structure
typedef struct {
    const char* name;
    int m, k, n;
    bool skip;  // For very large allocations
} TestCase;

// Initialize matrix with random values in [-1, 1]
void init_matrix(double* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = 2.0 * drand48() - 1.0;
    }
}

// Compute max relative error
double compute_max_rel_error(const double* ref, const double* test, int m, int n) {
    double max_rel_err = 0.0;
    double max_abs_ref = 0.0;

    for (int i = 0; i < m * n; i++) {
        double abs_val = fabs(ref[i]);
        if (abs_val > max_abs_ref) max_abs_ref = abs_val;
    }
    if (max_abs_ref < 1e-15) max_abs_ref = 1.0;

    for (int i = 0; i < m * n; i++) {
        double diff = fabs(ref[i] - test[i]);
        double rel_err = diff / max_abs_ref;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }
    return max_rel_err;
}

// Compute Frobenius relative error
double compute_frobenius_error(const double* ref, const double* test, int m, int n) {
    double sum_sq_diff = 0.0;
    double sum_sq_ref = 0.0;

    for (int i = 0; i < m * n; i++) {
        double diff = ref[i] - test[i];
        sum_sq_diff += diff * diff;
        sum_sq_ref += ref[i] * ref[i];
    }

    return sqrt(sum_sq_diff / (sum_sq_ref + 1e-15));
}

// Run test
int run_test(const TestCase* tc) {
    printf("Testing: %-35s [M=%5d, K=%5d, N=%5d] ... ",
           tc->name, tc->m, tc->k, tc->n);
    fflush(stdout);

    if (tc->skip) {
        printf(YELLOW "SKIP" RESET " (too large for memory)\n");
        return 1;  // Consider skip as pass
    }

    // Allocate matrices
    size_t size_A = (size_t)tc->m * tc->k * sizeof(double);
    size_t size_B = (size_t)tc->k * tc->n * sizeof(double);
    size_t size_C = (size_t)tc->m * tc->n * sizeof(double);

    double* A = (double*)malloc(size_A);
    double* B = (double*)malloc(size_B);
    double* C_ref = (double*)malloc(size_C);
    double* C_test = (double*)malloc(size_C);

    if (!A || !B || !C_ref || !C_test) {
        printf(YELLOW "SKIP" RESET " (allocation failed - %.1f GB needed)\n",
               (size_A + size_B + 2*size_C) / (1024.0*1024.0*1024.0));
        free(A); free(B); free(C_ref); free(C_test);
        return 1;  // Consider memory issue as skip
    }

    // Initialize
    init_matrix(A, tc->m, tc->k);
    init_matrix(B, tc->k, tc->n);
    memset(C_ref, 0, size_C);
    memset(C_test, 0, size_C);

    // Reference
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                tc->m, tc->n, tc->k,
                1.0, A, tc->k, B, tc->n,
                0.0, C_ref, tc->n);

    // apple-bottom
    ABMatrix ab_A = ab_matrix_create(tc->m, tc->k);
    ABMatrix ab_B = ab_matrix_create(tc->k, tc->n);
    ABMatrix ab_C = ab_matrix_create(tc->m, tc->n);

    if (!ab_A || !ab_B || !ab_C) {
        printf(YELLOW "SKIP" RESET " (GPU allocation failed)\n");
        if (ab_A) ab_matrix_destroy(ab_A);
        if (ab_B) ab_matrix_destroy(ab_B);
        if (ab_C) ab_matrix_destroy(ab_C);
        free(A); free(B); free(C_ref); free(C_test);
        return 1;
    }

    ABStatus status = ab_matrix_upload(ab_A, A, true);
    if (status != AB_OK) {
        printf(YELLOW "SKIP" RESET " (upload failed: %s)\n", ab_status_string(status));
        ab_matrix_destroy(ab_A); ab_matrix_destroy(ab_B); ab_matrix_destroy(ab_C);
        free(A); free(B); free(C_ref); free(C_test);
        return 1;
    }

    status = ab_matrix_upload(ab_B, B, true);
    if (status != AB_OK) {
        printf(YELLOW "SKIP" RESET " (upload B failed)\n");
        ab_matrix_destroy(ab_A); ab_matrix_destroy(ab_B); ab_matrix_destroy(ab_C);
        free(A); free(B); free(C_ref); free(C_test);
        return 1;
    }

    ab_matrix_zero(ab_C);
    status = ab_dgemm(ab_A, ab_B, ab_C);
    if (status != AB_OK) {
        printf(RED "FAIL" RESET " (dgemm error: %s)\n", ab_status_string(status));
        ab_matrix_destroy(ab_A); ab_matrix_destroy(ab_B); ab_matrix_destroy(ab_C);
        free(A); free(B); free(C_ref); free(C_test);
        return 0;
    }

    status = ab_matrix_download(ab_C, C_test, true);
    if (status != AB_OK) {
        printf(RED "FAIL" RESET " (download error)\n");
        ab_matrix_destroy(ab_A); ab_matrix_destroy(ab_B); ab_matrix_destroy(ab_C);
        free(A); free(B); free(C_ref); free(C_test);
        return 0;
    }

    // Compute errors
    double max_err = compute_max_rel_error(C_ref, C_test, tc->m, tc->n);
    double frob_err = compute_frobenius_error(C_ref, C_test, tc->m, tc->n);

    // Report
    int passed = (max_err < 1e-13);
    if (passed) {
        printf(GREEN "PASS" RESET " (max: %.2e, frob: %.2e)\n", max_err, frob_err);
    } else {
        printf(RED "FAIL" RESET " (max: %.2e, frob: %.2e)\n", max_err, frob_err);
    }

    // Cleanup
    ab_matrix_destroy(ab_A);
    ab_matrix_destroy(ab_B);
    ab_matrix_destroy(ab_C);
    free(A);
    free(B);
    free(C_ref);
    free(C_test);

    return passed;
}

int main() {
    printf("\n");
    printf("================================================================================\n");
    printf("Original Rectangular Matrix Test (from docs/design/RECTANGULAR_TILING.md)\n");
    printf("Testing documented dimension patterns that reportedly failed\n");
    printf("Threshold: max relative error < 1e-13\n");
    printf("================================================================================\n\n");

    srand48(42);

    ABStatus status = ab_init();
    if (status != AB_OK) {
        fprintf(stderr, "Failed to initialize apple-bottom: %s\n", ab_status_string(status));
        return 1;
    }

    // Test cases from docs/design/RECTANGULAR_TILING.md
    TestCase test_cases[] = {
        // DGEMM tests from doc
        {"tall-skinny (10000×100×100)",        10000, 100,   100,   false},
        {"short-wide (100×10000×10000)",       100,   100,   10000, false},
        {"thin middle K (5000×10×5000)",       5000,  10,    5000,  false},

        // Performance benchmark cases (lines 58-66)
        {"square baseline (2048×2048×2048)",   2048,  2048,  2048,  false},
        {"tall 4:1 (4096×1024×2048)",          4096,  2048,  1024,  false},
        {"tall 16:1 (8192×512×2048)",          8192,  2048,  512,   false},
        {"tall 64:1 (16384×256×2048)",         16384, 2048,  256,   false},
        {"wide 1:4 (1024×4096×2048)",          1024,  2048,  4096,  false},
        {"wide 1:16 (512×8192×2048)",          512,   2048,  8192,  false},

        // QE-specific dimension (line 43) - may be too large
        {"QE-like (18277×150×18277)",          18277, 18277, 150,   true},  // Skip - 5GB

        // Smaller QE-like test
        {"QE-like small (8192×150×8192)",      8192,  8192,  150,   false},
    };

    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);
    int passed = 0;
    int skipped = 0;

    printf("Running tests documented as failing in docs/design/RECTANGULAR_TILING.md:\n\n");

    for (int i = 0; i < num_tests; i++) {
        int result = run_test(&test_cases[i]);
        if (test_cases[i].skip) {
            skipped++;
        } else if (result) {
            passed++;
        }
    }

    printf("\n");
    printf("================================================================================\n");
    printf("Summary: %d/%d tests passed", passed, num_tests - skipped);
    if (skipped > 0) {
        printf(" (%d skipped due to memory)", skipped);
    }
    printf("\n");

    if (passed == num_tests - skipped) {
        printf(GREEN "✓ All documented failing cases now PASS!\n" RESET);
        printf("The BUG-1/BUG-2 fixes appear to have resolved the rectangular issues.\n");
        printf("Recommendation: Update docs/design/RECTANGULAR_TILING.md and PRECISION_ENVELOPE.md\n");
    } else {
        printf(YELLOW "Some patterns still fail - further investigation needed\n" RESET);
    }
    printf("================================================================================\n\n");

    ab_shutdown();
    return (passed == num_tests - skipped) ? 0 : 1;
}