// DTRSM & MPIR Benchmark — apple-bottom
// Benchmarks triangular solve and mixed-precision iterative refinement
// against Accelerate (AMX) baselines.
#include "apple_bottom.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <Accelerate/Accelerate.h>

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double frobenius_rel_error(const double* X, const double* Xref, int rows, int cols) {
    double norm_diff = 0, norm_ref = 0;
    for (int i = 0; i < rows * cols; i++) {
        double d = X[i] - Xref[i];
        norm_diff += d * d;
        norm_ref += Xref[i] * Xref[i];
    }
    return sqrt(norm_diff / (norm_ref > 0 ? norm_ref : 1e-300));
}

// Generate a random lower-triangular matrix with well-conditioned diagonal
static void gen_lower_triangular(double* A, int N) {
    memset(A, 0, (size_t)N * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        A[i * N + i] = 1.0 + drand48() * 5.0; // diagonal in [1, 6]
        for (int j = 0; j < i; j++) {
            A[i * N + j] = (drand48() - 0.5) * 2.0;
        }
    }
}

// Generate a random SPD matrix for MPIR testing
static void gen_spd(double* A, int N) {
    // A = L * L^T where L is lower triangular with positive diagonal
    double* L = calloc((size_t)N * N, sizeof(double));
    gen_lower_triangular(L, N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N,
                1.0, L, N, L, N, 0.0, A, N);
    free(L);
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  apple-bottom DTRSM & MPIR Benchmark                            ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    if (ab_init() != AB_OK) { fprintf(stderr, "Init failed\n"); return 1; }
    printf("Device: %s\n\n", ab_device_name());

    // =====================================================================
    // Part 1: DTRSM Benchmark
    // =====================================================================
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  DTRSM: Lower, No-Trans, Non-Unit                               ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    int dtrsm_sizes[] = {64, 128, 256, 512, 1024};
    int nrhs_vals[] = {1, 10, 64};

    printf("  N     │ NRHS │  AMX (ms)  │  GPU (ms)  │  Speedup  │ Precision\n");
    printf("────────┼──────┼────────────┼────────────┼───────────┼───────────\n");

    for (int si = 0; si < 5; si++) {
        int N = dtrsm_sizes[si];

        double* A = malloc((size_t)N * N * sizeof(double));
        srand48(42 + N);
        gen_lower_triangular(A, N);

        for (int ri = 0; ri < 3; ri++) {
            int NRHS = nrhs_vals[ri];
            int iters = (N <= 256) ? 20 : 5;
            size_t bcount = (size_t)N * NRHS;

            double* B_orig = malloc(bcount * sizeof(double));
            double* B_amx  = malloc(bcount * sizeof(double));
            double* B_gpu  = malloc(bcount * sizeof(double));

            for (size_t i = 0; i < bcount; i++)
                B_orig[i] = drand48() * 2.0 - 1.0;

            // AMX baseline: cblas_dtrsm
            memcpy(B_amx, B_orig, bcount * sizeof(double));
            cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans,
                        CblasNonUnit, N, NRHS, 1.0, A, N, B_amx, NRHS);
            // Warmup done, now time it
            double t0 = get_time_sec();
            for (int it = 0; it < iters; it++) {
                memcpy(B_amx, B_orig, bcount * sizeof(double));
                cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans,
                            CblasNonUnit, N, NRHS, 1.0, A, N, B_amx, NRHS);
            }
            double amx_ms = (get_time_sec() - t0) / iters * 1000.0;

            // Restore reference solution
            memcpy(B_amx, B_orig, bcount * sizeof(double));
            cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans,
                        CblasNonUnit, N, NRHS, 1.0, A, N, B_amx, NRHS);

            // GPU: ab_dtrsm
            ABMatrix mA = ab_matrix_create(N, N);
            ABMatrix mB = ab_matrix_create(N, NRHS);
            ab_matrix_upload(mA, A, true);

            // Warmup
            ab_matrix_upload(mB, B_orig, true);
            ab_dtrsm(AB_LEFT, AB_LOWER, AB_NO_TRANS, AB_NON_UNIT, 1.0, mA, mB);

            t0 = get_time_sec();
            for (int it = 0; it < iters; it++) {
                ab_matrix_upload(mB, B_orig, true);
                ab_dtrsm(AB_LEFT, AB_LOWER, AB_NO_TRANS, AB_NON_UNIT, 1.0, mA, mB);
            }
            double gpu_ms = (get_time_sec() - t0) / iters * 1000.0;

            ab_matrix_download(mB, B_gpu, true);

            double err = frobenius_rel_error(B_gpu, B_amx, N, NRHS);
            double speedup = amx_ms / gpu_ms;

            printf("  %4d  │ %4d │ %8.2f   │ %8.2f   │  %5.2fx   │ %.2e\n",
                   N, NRHS, amx_ms, gpu_ms, speedup, err);

            ab_matrix_destroy(mA);
            ab_matrix_destroy(mB);
            free(B_orig); free(B_amx); free(B_gpu);
        }
        free(A);
    }

    // =====================================================================
    // Part 2: MPIR Benchmark (mixed-precision iterative refinement)
    // =====================================================================
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  MPIR: Mixed-Precision Iterative Refinement (ab_dgesv_mpir)      ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    int mpir_sizes[] = {64, 128, 256, 512};
    int mpir_nrhs[] = {1, 10};

    printf("  N     │ NRHS │  LAPACK (ms) │  MPIR (ms)  │  Speedup  │ Precision\n");
    printf("────────┼──────┼──────────────┼─────────────┼───────────┼───────────\n");

    for (int si = 0; si < 4; si++) {
        int N = mpir_sizes[si];
        int iters = (N <= 128) ? 10 : 3;

        double* A_spd = malloc((size_t)N * N * sizeof(double));
        srand48(123 + N);
        gen_spd(A_spd, N);

        for (int ri = 0; ri < 2; ri++) {
            int NRHS = mpir_nrhs[ri];
            size_t bcount = (size_t)N * NRHS;

            double* B_orig = malloc(bcount * sizeof(double));
            double* B_lap  = malloc(bcount * sizeof(double));
            double* B_mpir = malloc(bcount * sizeof(double));
            double* A_tmp  = malloc((size_t)N * N * sizeof(double));

            for (size_t i = 0; i < bcount; i++)
                B_orig[i] = drand48() * 2.0 - 1.0;

            // LAPACK baseline: use LAPACKE row-major interface (LAPACK_ROW_MAJOR)
            // Accelerate provides LAPACKE via lapacke.h, but we can call dgesv_
            // with column-major layout. Store B as N×NRHS column-major (ldb=N).
            double* B_lap_cm = malloc(bcount * sizeof(double));

            // Transpose B_orig (row-major N×NRHS) → B_lap_cm (col-major N×NRHS)
            for (int r = 0; r < N; r++)
                for (int c = 0; c < NRHS; c++)
                    B_lap_cm[c * N + r] = B_orig[r * NRHS + c];

            memcpy(A_tmp, A_spd, (size_t)N * N * sizeof(double));
            {
                int n = N, nrhs = NRHS, lda = N, ldb = N, info;
                int* ipiv = malloc(N * sizeof(int));
                dgesv_(&n, &nrhs, A_tmp, &lda, ipiv, B_lap_cm, &ldb, &info);
                free(ipiv);
                if (info != 0) fprintf(stderr, "dgesv_ info=%d\n", info);
            }

            // Transpose solution back to row-major for comparison
            for (int r = 0; r < N; r++)
                for (int c = 0; c < NRHS; c++)
                    B_lap[r * NRHS + c] = B_lap_cm[c * N + r];

            // Time LAPACK
            double t0 = get_time_sec();
            for (int it = 0; it < iters; it++) {
                for (int r = 0; r < N; r++)
                    for (int c = 0; c < NRHS; c++)
                        B_lap_cm[c * N + r] = B_orig[r * NRHS + c];
                memcpy(A_tmp, A_spd, (size_t)N * N * sizeof(double));
                int n = N, nrhs = NRHS, lda = N, ldb = N, info;
                int* ipiv = malloc(N * sizeof(int));
                dgesv_(&n, &nrhs, A_tmp, &lda, ipiv, B_lap_cm, &ldb, &info);
                free(ipiv);
            }
            double lap_ms = (get_time_sec() - t0) / iters * 1000.0;

            // Recompute reference (B_lap already has solution from initial call above)

            // MPIR benchmark
            ABMatrix mA = ab_matrix_create(N, N);
            ABMatrix mB = ab_matrix_create(N, NRHS);
            ab_matrix_upload(mA, A_spd, true);

            // Warmup
            ab_matrix_upload(mB, B_orig, true);
            ab_dgesv_mpir(mA, mB);

            t0 = get_time_sec();
            for (int it = 0; it < iters; it++) {
                ab_matrix_upload(mA, A_spd, true);   // MPIR modifies A internally
                ab_matrix_upload(mB, B_orig, true);
                ab_dgesv_mpir(mA, mB);
            }
            double mpir_ms = (get_time_sec() - t0) / iters * 1000.0;

            ab_matrix_download(mB, B_mpir, true);

            double err = frobenius_rel_error(B_mpir, B_lap, N, NRHS);
            double speedup = lap_ms / mpir_ms;

            printf("  %4d  │ %4d │ %10.2f   │ %9.2f   │  %5.2fx   │ %.2e\n",
                   N, NRHS, lap_ms, mpir_ms, speedup, err);

            ab_matrix_destroy(mA);
            ab_matrix_destroy(mB);
            free(B_orig); free(B_lap); free(B_lap_cm); free(B_mpir); free(A_tmp);
        }
        free(A_spd);
    }

    printf("\n");
    ab_shutdown();
    return 0;
}
