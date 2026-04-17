// =============================================================================
// bench_device_residency.c — Microbenchmark: device-resident vs per-call upload
// =============================================================================
//
// Measures the wall-clock delta between:
//   A) 100 × ab_dev_zgemm  (matrices resident on device, no re-upload)
//   B) 100 × ab_zgemm      (upload + dispatch + download each call)
//
// The expected benefit of path A is ~5-16ms saved from eliminated FP64→DD
// conversions and buffer allocation, depending on matrix size.
//
// Usage: ./build/bench_device_residency [N]  (default N=1024)
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "apple_bottom.h"
#include "apple_bottom_device.h"

#define ITERATIONS 100

static double wallclock_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

static double rand_double(void) {
    return (double)rand() / (double)RAND_MAX * 2.0 - 1.0;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    if (N < 64) N = 64;

    printf("═══════════════════════════════════════════════════════════\n");
    printf("Device-residency benchmark: N=%d, %d iterations\n", N, ITERATIONS);
    printf("═══════════════════════════════════════════════════════════\n\n");

    if (ab_init() != AB_OK) {
        fprintf(stderr, "ab_init failed\n");
        return 1;
    }

    size_t nn = (size_t)N * N;
    srand(42);

    // Generate random complex matrices (interleaved for device path)
    double* A_cplx = (double*)malloc(nn * 2 * sizeof(double));
    double* B_cplx = (double*)malloc(nn * 2 * sizeof(double));
    double* C_cplx = (double*)malloc(nn * 2 * sizeof(double));
    double* Ar = (double*)malloc(nn * sizeof(double));
    double* Ai = (double*)malloc(nn * sizeof(double));
    double* Br = (double*)malloc(nn * sizeof(double));
    double* Bi = (double*)malloc(nn * sizeof(double));
    double* Cr = (double*)malloc(nn * sizeof(double));
    double* Ci = (double*)malloc(nn * sizeof(double));

    for (size_t i = 0; i < nn; i++) {
        Ar[i] = rand_double(); Ai[i] = rand_double();
        Br[i] = rand_double(); Bi[i] = rand_double();
        A_cplx[i * 2] = Ar[i]; A_cplx[i * 2 + 1] = Ai[i];
        B_cplx[i * 2] = Br[i]; B_cplx[i * 2 + 1] = Bi[i];
    }

    // -----------------------------------------------------------------------
    // Path A: device-buffer API (data staged once, ZGEMM dispatched 100x)
    // -----------------------------------------------------------------------
    ab_dev_buffer_t dA = ab_dev_malloc(nn * 2 * sizeof(double));
    ab_dev_buffer_t dB = ab_dev_malloc(nn * 2 * sizeof(double));
    ab_dev_buffer_t dC = ab_dev_malloc(nn * 2 * sizeof(double));
    ab_dev_memcpy_h2d(dA, 0, A_cplx, nn * 2 * sizeof(double));
    ab_dev_memcpy_h2d(dB, 0, B_cplx, nn * 2 * sizeof(double));

    double alpha[2] = {1.0, 0.0};
    double beta[2]  = {0.0, 0.0};

    // Warm up
    ab_dev_zgemm(AB_NO_TRANS, AB_NO_TRANS, N, N, N,
                 alpha, dA, N, dB, N, beta, dC, N);

    double t0 = wallclock_ms();
    for (int i = 0; i < ITERATIONS; i++) {
        ab_dev_zgemm(AB_NO_TRANS, AB_NO_TRANS, N, N, N,
                     alpha, dA, N, dB, N, beta, dC, N);
    }
    double dev_ms = wallclock_ms() - t0;

    ab_dev_free(dA);
    ab_dev_free(dB);
    ab_dev_free(dC);

    // -----------------------------------------------------------------------
    // Path B: matrix-handle API (upload + dispatch + download each call)
    // -----------------------------------------------------------------------

    // Warm up
    {
        ABMatrix mAr = ab_matrix_create(N, N);
        ABMatrix mAi = ab_matrix_create(N, N);
        ABMatrix mBr = ab_matrix_create(N, N);
        ABMatrix mBi = ab_matrix_create(N, N);
        ABMatrix mCr = ab_matrix_create(N, N);
        ABMatrix mCi = ab_matrix_create(N, N);
        ab_matrix_upload(mAr, Ar, true); ab_matrix_upload(mAi, Ai, true);
        ab_matrix_upload(mBr, Br, true); ab_matrix_upload(mBi, Bi, true);
        ab_zgemm(mAr, mAi, mBr, mBi, mCr, mCi);
        ab_matrix_destroy(mAr); ab_matrix_destroy(mAi);
        ab_matrix_destroy(mBr); ab_matrix_destroy(mBi);
        ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);
    }

    t0 = wallclock_ms();
    for (int i = 0; i < ITERATIONS; i++) {
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
        ab_matrix_destroy(mAr); ab_matrix_destroy(mAi);
        ab_matrix_destroy(mBr); ab_matrix_destroy(mBi);
        ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);
    }
    double handle_ms = wallclock_ms() - t0;

    // -----------------------------------------------------------------------
    // Report
    // -----------------------------------------------------------------------
    printf("Device-buffer path:  %8.1f ms (%5.1f ms/call)\n",
           dev_ms, dev_ms / ITERATIONS);
    printf("Matrix-handle path:  %8.1f ms (%5.1f ms/call)\n",
           handle_ms, handle_ms / ITERATIONS);
    printf("Delta:               %+8.1f ms (%+.1f ms/call)\n",
           dev_ms - handle_ms, (dev_ms - handle_ms) / ITERATIONS);
    printf("\nNote: Week-2 device-buffer path still does FP64<->DD conversion\n");
    printf("per call. Month-2 DD-native storage will eliminate this overhead.\n");

    free(A_cplx); free(B_cplx); free(C_cplx);
    free(Ar); free(Ai); free(Br); free(Bi); free(Cr); free(Ci);
    ab_shutdown();
    return 0;
}
