// Basic DGEMM Example — apple-bottom
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

int main(int argc, char** argv) {
    int N = 2048;
    if (argc > 1) N = atoi(argv[1]);
    if (N < 64 || N > 8192) { fprintf(stderr, "Usage: %s [size 64-8192]\n", argv[0]); return 1; }
    
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("apple-bottom Basic DGEMM Example\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    
    ABStatus status = ab_init();
    if (status != AB_OK) { fprintf(stderr, "Init failed: %s\n", ab_status_string(status)); return 1; }
    printf("Device: %s\n\n", ab_device_name());
    
    size_t count = (size_t)N * N;
    double* A = malloc(count * sizeof(double));
    double* B = malloc(count * sizeof(double));
    double* C_gpu = malloc(count * sizeof(double));
    double* C_ref = malloc(count * sizeof(double));
    
    printf("Initializing %d × %d matrices...\n", N, N);
    srand48(42);
    for (size_t i = 0; i < count; i++) { A[i] = drand48() * 2 - 1; B[i] = drand48() * 2 - 1; }
    
    ABMatrix mA = ab_matrix_create(N, N);
    ABMatrix mB = ab_matrix_create(N, N);
    ABMatrix mC = ab_matrix_create(N, N);
    
    printf("Uploading to GPU...\n");
    double t0 = get_time_sec();
    ab_matrix_upload(mA, A, true);
    ab_matrix_upload(mB, B, true);
    double upload_ms = (get_time_sec() - t0) * 1000;
    printf("  Upload time: %.1f ms\n\n", upload_ms);
    
    printf("Computing C = A × B on GPU...\n");
    ab_dgemm(mA, mB, mC); // warmup
    
    int iterations = 5;
    t0 = get_time_sec();
    for (int i = 0; i < iterations; i++) ab_dgemm(mA, mB, mC);
    double kernel_ms = (get_time_sec() - t0) * 1000 / iterations;
    double flops = 2.0 * N * N * N;
    printf("  Kernel time: %.1f ms\n  Performance: %.0f GFLOP/s\n\n", kernel_ms, flops / (kernel_ms * 1e6));
    
    printf("Downloading from GPU...\n");
    t0 = get_time_sec();
    ab_matrix_download(mC, C_gpu, true);
    printf("  Download time: %.1f ms\n\n", (get_time_sec() - t0) * 1000);
    
    printf("Computing reference with Accelerate (AMX)...\n");
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C_ref, N);
    t0 = get_time_sec();
    for (int i = 0; i < iterations; i++)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C_ref, N);
    double amx_ms = (get_time_sec() - t0) * 1000 / iterations;
    printf("  AMX time: %.1f ms (%.0f GFLOP/s)\n\n", amx_ms, flops / (amx_ms * 1e6));
    
    double max_err = 0, sum_sq = 0;
    for (size_t i = 0; i < count; i++) {
        double err = fabs(C_gpu[i] - C_ref[i]);
        if (err > max_err) max_err = err;
        sum_sq += C_ref[i] * C_ref[i];
    }
    double frob_err = max_err / sqrt(sum_sq);
    printf("Precision (Frobenius): %.2e %s\n\n", frob_err, frob_err < 1e-14 ? "✓" : "✗");
    
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("SUMMARY (%d × %d)\n", N, N);
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    double speedup = amx_ms / kernel_ms;
    printf("  apple-bottom: %.1f ms (%.0f GFLOP/s)\n", kernel_ms, flops / (kernel_ms * 1e6));
    printf("  AMX:          %.1f ms (%.0f GFLOP/s)\n", amx_ms, flops / (amx_ms * 1e6));
    printf("  Speedup:      %.2fx %s\n\n", speedup, speedup > 1.0 ? "✓ GPU wins" : "(AMX wins)");
    
    ab_matrix_destroy(mA); ab_matrix_destroy(mB); ab_matrix_destroy(mC);
    ab_shutdown();
    free(A); free(B); free(C_gpu); free(C_ref);
    return 0;
}
