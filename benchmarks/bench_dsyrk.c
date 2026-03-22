// DSYRK Benchmark — apple-bottom
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

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  apple-bottom DSYRK Benchmark                                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    if (ab_init() != AB_OK) { fprintf(stderr, "Init failed\n"); return 1; }
    printf("Device: %s\n\n", ab_device_name());
    
    int sizes[] = {256, 512, 1024, 2048, 3072};
    int K = 256;  // Fixed K dimension
    
    printf("  N       │  AMX GFLOP/s │  GPU GFLOP/s │  Speedup  │ Status\n");
    printf("──────────┼──────────────┼──────────────┼───────────┼────────\n");
    
    for (int s = 0; s < 5; s++) {
        int N = sizes[s];
        size_t countA = (size_t)N * K;
        size_t countC = (size_t)N * N;
        // DSYRK: N²K flops (symmetric, half the work)
        double flops = 1.0 * N * N * K;
        int iters = (N <= 1024) ? 5 : 3;
        
        double* A = malloc(countA * sizeof(double));
        double* C_gpu = malloc(countC * sizeof(double));
        double* C_ref = malloc(countC * sizeof(double));
        
        srand48(42);
        for (size_t i = 0; i < countA; i++) A[i] = drand48() * 2 - 1;
        memset(C_ref, 0, countC * sizeof(double));
        
        // AMX benchmark: C = A × Aᵀ
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, N, K, 1.0, A, K, 0.0, C_ref, N);
        double t0 = get_time_sec();
        for (int i = 0; i < iters; i++)
            cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, N, K, 1.0, A, K, 0.0, C_ref, N);
        double amx_gflops = flops * iters / ((get_time_sec() - t0) * 1e9);
        
        // GPU benchmark
        ABMatrix mA = ab_matrix_create(N, K);
        ABMatrix mC = ab_matrix_create(N, N);
        ab_matrix_upload(mA, A, true);
        ab_dsyrk(mA, mC); // warmup
        
        t0 = get_time_sec();
        for (int i = 0; i < iters; i++)
            ab_dsyrk(mA, mC);
        double gpu_gflops = flops * iters / ((get_time_sec() - t0) * 1e9);
        
        ab_matrix_download(mC, C_gpu, true);
        
        // Precision check (upper triangle only)
        double max_err = 0;
        for (int i = 0; i < N; i++) {
            for (int j = i; j < N; j++) {
                double err = fabs(C_gpu[i * N + j] - C_ref[i * N + j]);
                if (err > max_err) max_err = err;
            }
        }
        
        double speedup = gpu_gflops / amx_gflops;
        printf("  %4d    │    %6.0f    │    %6.0f    │  %5.2fx %s │  %s\n",
               N, amx_gflops, gpu_gflops, speedup, speedup >= 1.0 ? "✓" : " ",
               max_err < 1e-10 ? "✓ PASS" : "✗ FAIL");
        
        ab_matrix_destroy(mA);
        ab_matrix_destroy(mC);
        free(A); free(C_gpu); free(C_ref);
    }
    
    printf("\n");
    printf("Note: K fixed at %d\n", K);
    
    ab_shutdown();
    return 0;
}
