// =============================================================================
// V-2: DGEMM Convergence Study
// =============================================================================
// Purpose: Verify that DD arithmetic error scales as O(N·ε²) as expected
// Reference: ASME V&V 10-2006 convergence verification
//
// Test approach:
// - Compute C = A × B for sizes N ∈ {64, 128, 256, 512, 1024, 2048, 4096}
// - Compare against cblas_dgemm (IEEE FP64 reference)
// - Measure Frobenius relative error: ||C_gpu - C_ref||_F / ||C_ref||_F
// - Measure max element-wise relative error: max_i |C_gpu[i] - C_ref[i]| / |C_ref[i]|
// - Fit log(error) = α·log(N) + β to determine convergence rate α
// - Expected: α ≈ 0.5 (random-sign rounding errors add up like √N; the
//   worst case is α = 1)
// - Acceptance: Frobenius error <= FROB_BOUND·√N·u² at every N (magnitude:
//   catches precision loss, e.g. a build without MTLMathModeSafe, whose slope
//   is still ≈ 0.5) and α < 1.5 (if slope > 1.5, indicates kernel bug)
// =============================================================================

#include "apple_bottom.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <Accelerate/Accelerate.h>

// Same normwise bound as tests/test_precision.c: ||ΔC||_F/||C||_F <=
// FROB_BOUND·√N·u², u² = 2⁻⁴⁸. Measured 0.22–0.24·√N·u² here.
#define U2         0x1p-48
#define FROB_BOUND 1.0

typedef struct {
    int N;
    double frob_err;
    double max_err;
} ConvergencePoint;

// Linear regression: fit log(y) = α·log(x) + β
static void fit_log_log(const double* x, const double* y, int n,
                       double* slope, double* intercept, double* r_squared) {
    // Convert to log space
    double log_x[n], log_y[n];
    for (int i = 0; i < n; i++) {
        log_x[i] = log(x[i]);
        log_y[i] = log(y[i]);
    }

    // Compute means
    double mean_x = 0, mean_y = 0;
    for (int i = 0; i < n; i++) {
        mean_x += log_x[i];
        mean_y += log_y[i];
    }
    mean_x /= n;
    mean_y /= n;

    // Compute slope and intercept
    double num = 0, den = 0, ss_tot = 0, ss_res = 0;
    for (int i = 0; i < n; i++) {
        double dx = log_x[i] - mean_x;
        double dy = log_y[i] - mean_y;
        num += dx * dy;
        den += dx * dx;
        ss_tot += dy * dy;
    }

    *slope = num / den;
    *intercept = mean_y - (*slope) * mean_x;

    // Compute R²
    for (int i = 0; i < n; i++) {
        double predicted = (*slope) * log_x[i] + (*intercept);
        double residual = log_y[i] - predicted;
        ss_res += residual * residual;
    }
    *r_squared = 1.0 - (ss_res / ss_tot);
}

static ConvergencePoint test_size(int N) {
    ConvergencePoint pt;
    pt.N = N;

    size_t count = (size_t)N * N;

    // Allocate host memory
    double* A = (double*)malloc(count * sizeof(double));
    double* B = (double*)malloc(count * sizeof(double));
    double* C_ref = (double*)malloc(count * sizeof(double));
    double* C_gpu = (double*)malloc(count * sizeof(double));

    // Initialize with random values in range [-1, 1]
    srand(42);  // Fixed seed for reproducibility
    for (size_t i = 0; i < count; i++) {
        A[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
        B[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
        C_ref[i] = 0.0;
        C_gpu[i] = 0.0;
    }

    // Reference: cblas_dgemm (IEEE FP64)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0, A, N, B, N, 0.0, C_ref, N);

    // GPU: apple-bottom (DD arithmetic)
    ABMatrix mA = ab_matrix_create(N, N);
    ABMatrix mB = ab_matrix_create(N, N);
    ABMatrix mC = ab_matrix_create(N, N);

    ab_matrix_upload(mA, A, true);
    ab_matrix_upload(mB, B, true);
    ab_dgemm(mA, mB, mC);
    ab_matrix_download(mC, C_gpu, true);

    ab_matrix_destroy(mA);
    ab_matrix_destroy(mB);
    ab_matrix_destroy(mC);

    // Compute errors
    double frob_num = 0.0, frob_den = 0.0;
    double max_rel_err = 0.0;

    for (size_t i = 0; i < count; i++) {
        double diff = C_gpu[i] - C_ref[i];
        frob_num += diff * diff;
        frob_den += C_ref[i] * C_ref[i];

        // Element-wise relative error
        if (fabs(C_ref[i]) > 1e-15) {
            double rel_err = fabs(diff) / fabs(C_ref[i]);
            if (rel_err > max_rel_err) max_rel_err = rel_err;
        }
    }

    pt.frob_err = sqrt(frob_num) / sqrt(frob_den);
    pt.max_err = max_rel_err;

    free(A); free(B); free(C_ref); free(C_gpu);
    return pt;
}

int main(int argc, char** argv) {
    // CSV output path: argv[1] (the Makefile passes $(BUILD)/...), else the
    // default build directory.
    const char* csv_path = argc > 1 ? argv[1] : "build/convergence_data.csv";

    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  V-2: DGEMM Convergence Study                                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    ABStatus status = ab_init();
    if (status != AB_OK) {
        fprintf(stderr, "Failed to initialize apple-bottom: %s\n", ab_status_string(status));
        return 1;
    }

    printf("Device: %s\n\n", ab_device_name());

    // Test sizes
    int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    ConvergencePoint* points = (ConvergencePoint*)malloc(n_sizes * sizeof(ConvergencePoint));

    printf("Testing convergence across matrix sizes:\n");
    printf("══════════════════════════════════════════════════════════════════\n");
    printf("  Size  │ Frobenius Err │ Max Elem Err │   Bound   │ Status\n");
    printf("────────┼───────────────┼──────────────┼───────────┼────────\n");

    int all_bounded = 1;
    for (int i = 0; i < n_sizes; i++) {
        points[i] = test_size(sizes[i]);
        double bound = FROB_BOUND * sqrt((double)points[i].N) * U2;
        int bounded = points[i].frob_err <= bound;
        if (!bounded) all_bounded = 0;
        printf(" %5d  │   %.2e   │   %.2e   │ %.2e  │ %s\n",
               points[i].N, points[i].frob_err, points[i].max_err, bound,
               bounded ? "✓ PASS" : "✗ FAIL");
    }

    printf("══════════════════════════════════════════════════════════════════\n\n");

    // Compute convergence slope (Frobenius error)
    double x[n_sizes], y_frob[n_sizes], y_max[n_sizes];
    for (int i = 0; i < n_sizes; i++) {
        x[i] = (double)points[i].N;
        y_frob[i] = points[i].frob_err;
        y_max[i] = points[i].max_err;
    }

    double slope_frob, intercept_frob, r2_frob;
    double slope_max, intercept_max, r2_max;

    fit_log_log(x, y_frob, n_sizes, &slope_frob, &intercept_frob, &r2_frob);
    fit_log_log(x, y_max, n_sizes, &slope_max, &intercept_max, &r2_max);

    printf("Convergence Analysis (log-log regression):\n");
    printf("══════════════════════════════════════════════════════════════════\n");
    printf("  Frobenius Error:    slope = %.3f, R² = %.4f\n", slope_frob, r2_frob);
    printf("  Max Element Error:  slope = %.3f, R² = %.4f\n", slope_max, r2_max);
    printf("\n");

    printf("Expected behavior:\n");
    printf("  - Slope ≈ 0.5: random-sign rounding errors grow like √N (worst case 1.0)\n");
    printf("  - Slope < 1.5 is acceptable for DD arithmetic\n");
    printf("  - R² > 0.9 indicates good linear fit in log-log space\n");
    printf("\n");

    // Write CSV for plotting
    FILE* csv = fopen(csv_path, "w");
    if (csv) {
        fprintf(csv, "N,Frobenius_Error,Max_Element_Error\n");
        for (int i = 0; i < n_sizes; i++) {
            fprintf(csv, "%d,%.16e,%.16e\n",
                    points[i].N, points[i].frob_err, points[i].max_err);
        }
        fclose(csv);
        printf("Data written to: %s\n\n", csv_path);
    } else {
        fprintf(stderr, "Cannot write %s: %s\n", csv_path, strerror(errno));
    }

    // Acceptance criteria
    int pass = 1;

    if (slope_frob > 1.5) {
        printf("✗ FAIL: Frobenius slope %.3f > 1.5 (potential kernel bug)\n", slope_frob);
        pass = 0;
    } else {
        printf("✓ PASS: Frobenius slope %.3f < 1.5 (acceptable)\n", slope_frob);
    }

    if (r2_frob < 0.85) {
        printf("⚠ WARN: Frobenius R² = %.4f < 0.85 (poor fit, irregular behavior)\n", r2_frob);
    } else {
        printf("✓ PASS: Frobenius R² = %.4f > 0.85 (good fit)\n", r2_frob);
    }

    if (all_bounded) {
        printf("✓ PASS: All Frobenius errors within %.0f·√N·u² (DD precision)\n", FROB_BOUND);
    } else {
        printf("✗ FAIL: Frobenius error above %.0f·√N·u² (DD precision lost)\n", FROB_BOUND);
        pass = 0;
    }

    printf("\n══════════════════════════════════════════════════════════════════\n");
    if (pass) {
        printf("✓ V-2 CONVERGENCE TEST PASSED\n");
    } else {
        printf("✗ V-2 CONVERGENCE TEST FAILED\n");
    }
    printf("══════════════════════════════════════════════════════════════════\n");

    free(points);
    ab_shutdown();

    return pass ? 0 : 1;
}
