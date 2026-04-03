// =============================================================================
// test_rectangular_diag.c — Diagnostic test for rectangular matrix failures
// Copyright (c) 2026 Grant Heileman, Technology Residue. MIT License.
// =============================================================================
//
// Purpose: Identify which rectangular dimension patterns fail in ab_dgemm.
// Tests 8 specific patterns against cblas_dgemm reference implementation.
// =============================================================================

#include "../include/apple_bottom.h"
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Test threshold for PASS/FAIL
#define REL_ERROR_THRESHOLD 1e-13

// Color codes for terminal output
#define RED     "\033[0;31m"
#define GREEN   "\033[0;32m"
#define YELLOW  "\033[0;33m"
#define RESET   "\033[0m"

// Test case structure
typedef struct {
    const char* name;
    int m, k, n;
} TestCase;

// Initialize matrix with random values in [-1, 1]
void init_matrix(double* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = 2.0 * drand48() - 1.0;
    }
}

// Compute maximum relative error between two matrices
double compute_max_rel_error(const double* ref, const double* test, int m, int n) {
    double max_rel_err = 0.0;
    double max_abs_ref = 0.0;

    // Find max absolute value in reference
    for (int i = 0; i < m * n; i++) {
        double abs_val = fabs(ref[i]);
        if (abs_val > max_abs_ref) {
            max_abs_ref = abs_val;
        }
    }

    // Avoid division by zero
    if (max_abs_ref < 1e-15) {
        max_abs_ref = 1.0;
    }

    // Compute element-wise relative errors
    for (int i = 0; i < m * n; i++) {
        double diff = fabs(ref[i] - test[i]);
        double rel_err = diff / max_abs_ref;
        if (rel_err > max_rel_err) {
            max_rel_err = rel_err;
        }
    }

    return max_rel_err;
}

// Run a single test case
int run_test(const TestCase* tc) {
    printf("Testing: %-25s [M=%5d, K=%5d, N=%5d] ... ",
           tc->name, tc->m, tc->k, tc->n);
    fflush(stdout);

    // Allocate matrices
    double* A = (double*)calloc(tc->m * tc->k, sizeof(double));
    double* B = (double*)calloc(tc->k * tc->n, sizeof(double));
    double* C_ref = (double*)calloc(tc->m * tc->n, sizeof(double));
    double* C_test = (double*)calloc(tc->m * tc->n, sizeof(double));

    if (!A || !B || !C_ref || !C_test) {
        printf(RED "FAILED" RESET " (allocation error)\n");
        free(A); free(B); free(C_ref); free(C_test);
        return 0;
    }

    // Initialize with random data
    init_matrix(A, tc->m, tc->k);
    init_matrix(B, tc->k, tc->n);
    memset(C_ref, 0, tc->m * tc->n * sizeof(double));
    memset(C_test, 0, tc->m * tc->n * sizeof(double));

    // Reference: cblas_dgemm
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                tc->m, tc->n, tc->k,
                1.0, A, tc->k, B, tc->n,
                0.0, C_ref, tc->n);

    // Test: apple_bottom
    ABMatrix ab_A = ab_matrix_create(tc->m, tc->k);
    ABMatrix ab_B = ab_matrix_create(tc->k, tc->n);
    ABMatrix ab_C = ab_matrix_create(tc->m, tc->n);

    if (!ab_A || !ab_B || !ab_C) {
        printf(RED "FAILED" RESET " (AB matrix creation error)\n");
        if (ab_A) ab_matrix_destroy(ab_A);
        if (ab_B) ab_matrix_destroy(ab_B);
        if (ab_C) ab_matrix_destroy(ab_C);
        free(A); free(B); free(C_ref); free(C_test);
        return 0;
    }

    ABStatus status = AB_OK;
    status = ab_matrix_upload(ab_A, A, true);
    if (status != AB_OK) {
        printf(RED "FAILED" RESET " (upload A error: %s)\n", ab_status_string(status));
        ab_matrix_destroy(ab_A); ab_matrix_destroy(ab_B); ab_matrix_destroy(ab_C);
        free(A); free(B); free(C_ref); free(C_test);
        return 0;
    }

    status = ab_matrix_upload(ab_B, B, true);
    if (status != AB_OK) {
        printf(RED "FAILED" RESET " (upload B error: %s)\n", ab_status_string(status));
        ab_matrix_destroy(ab_A); ab_matrix_destroy(ab_B); ab_matrix_destroy(ab_C);
        free(A); free(B); free(C_ref); free(C_test);
        return 0;
    }

    status = ab_matrix_zero(ab_C);
    if (status != AB_OK) {
        printf(RED "FAILED" RESET " (zero C error: %s)\n", ab_status_string(status));
        ab_matrix_destroy(ab_A); ab_matrix_destroy(ab_B); ab_matrix_destroy(ab_C);
        free(A); free(B); free(C_ref); free(C_test);
        return 0;
    }

    status = ab_dgemm(ab_A, ab_B, ab_C);
    if (status != AB_OK) {
        printf(RED "FAILED" RESET " (dgemm error: %s)\n", ab_status_string(status));
        ab_matrix_destroy(ab_A); ab_matrix_destroy(ab_B); ab_matrix_destroy(ab_C);
        free(A); free(B); free(C_ref); free(C_test);
        return 0;
    }

    status = ab_matrix_download(ab_C, C_test, true);
    if (status != AB_OK) {
        printf(RED "FAILED" RESET " (download error: %s)\n", ab_status_string(status));
        ab_matrix_destroy(ab_A); ab_matrix_destroy(ab_B); ab_matrix_destroy(ab_C);
        free(A); free(B); free(C_ref); free(C_test);
        return 0;
    }

    // Compute error
    double max_rel_err = compute_max_rel_error(C_ref, C_test, tc->m, tc->n);

    // Report results
    int passed = (max_rel_err < REL_ERROR_THRESHOLD);
    if (passed) {
        printf(GREEN "PASS" RESET " (max rel err: %.3e)\n", max_rel_err);
    } else {
        printf(RED "FAIL" RESET " (max rel err: %.3e)\n", max_rel_err);

        // Show a few sample errors for debugging
        printf("  Sample errors (first 5 mismatches):\n");
        int count = 0;
        for (int i = 0; i < tc->m * tc->n && count < 5; i++) {
            double diff = fabs(C_ref[i] - C_test[i]);
            if (diff / fabs(C_ref[i] + 1e-15) > 1e-10) {
                int row = i / tc->n;
                int col = i % tc->n;
                printf("    [%d,%d]: ref=%.6e, test=%.6e, diff=%.3e\n",
                       row, col, C_ref[i], C_test[i], diff);
                count++;
            }
        }
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
    printf("Rectangular Matrix Diagnostic Test for apple-bottom\n");
    printf("Testing 8 dimension patterns against cblas_dgemm reference\n");
    printf("Threshold: max relative error < %.1e\n", REL_ERROR_THRESHOLD);
    printf("================================================================================\n\n");

    // Set random seed for reproducibility
    srand48(42);

    // Initialize apple-bottom
    ABStatus status = ab_init();
    if (status != AB_OK) {
        fprintf(stderr, "Failed to initialize apple-bottom: %s\n", ab_status_string(status));
        return 1;
    }

    // Define test cases
    TestCase test_cases[] = {
        {"tall A",           10000, 100,   100},
        {"wide B",           100,   100,   10000},
        {"skinny K",         10000, 10,    10000},
        {"moderate skinny K", 1000,  10,    1000},
        {"fat K",            100,   10000, 100},
        {"K < TK",           64,    1,     64},
        {"tall output",      10000, 100,   10},
        {"wide output",      10,    100,   10000}
    };

    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);
    int passed = 0;

    // Run all tests
    for (int i = 0; i < num_tests; i++) {
        if (run_test(&test_cases[i])) {
            passed++;
        }
    }

    // Summary
    printf("\n");
    printf("================================================================================\n");
    printf("Summary: %d/%d tests passed\n", passed, num_tests);

    if (passed == num_tests) {
        printf(GREEN "All rectangular patterns work correctly!\n" RESET);
    } else {
        printf(YELLOW "Found %d failing patterns — kernel debugging needed\n" RESET,
               num_tests - passed);
    }
    printf("================================================================================\n\n");

    // Cleanup
    ab_shutdown();

    return (passed == num_tests) ? 0 : 1;
}