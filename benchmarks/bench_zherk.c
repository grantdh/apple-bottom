// ZHERK Benchmark — apple-bottom
#include "apple_bottom.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <Accelerate/Accelerate.h>

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  apple-bottom ZHERK Benchmark                                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    if (ab_init() != AB_OK) { fprintf(stderr, "Init failed\n"); return 1; }
    printf("Device: %s\n\n", ab_device_name());
    
    int sizes[] = {256, 512, 1024, 2048};
    int K = 256;
    
    printf("  N       │  AMX GFLOP/s │  GPU GFLOP/s │  Speedup  │ Status\n");
    printf("──────────┼──────────────┼──────────────┼───────────┼────────\n");
    
    for (int s = 0; s < 4; s++) {
        int N = sizes[s];
        size_t countA = (size_t)N * K;
        size_t countC = (size_t)N * N;
        double flops = 4.0 * N * N * K;  // Complex symmetric
        int iters = (N <= 1024) ? 5 : 3;
        
        double* Ar = malloc(countA * sizeof(double));
        double* Ai = malloc(countA * sizeof(double));
        double* Cr_gpu = malloc(countC * sizeof(double));
        double* Ci_gpu = malloc(countC * sizeof(double));
        
        double complex* A_ref = malloc(countA * sizeof(double complex));
        double complex* C_ref = malloc(countC * sizeof(double complex));
        
        srand48(42);
        for (size_t i = 0; i < countA; i++) {
            Ar[i] = drand48() * 2 - 1;
            Ai[i] = drand48() * 2 - 1;
            A_ref[i] = Ar[i] + I * Ai[i];
        }
        memset(C_ref, 0, countC * sizeof(double complex));
        
        // AMX benchmark
        cblas_zherk(CblasRowMajor, CblasUpper, CblasNoTrans, N, K, 1.0, A_ref, K, 0.0, C_ref, N);
        double t0 = get_time_sec();
        for (int i = 0; i < iters; i++)
            cblas_zherk(CblasRowMajor, CblasUpper, CblasNoTrans, N, K, 1.0, A_ref, K, 0.0, C_ref, N);
        double amx_gflops = flops * iters / ((get_time_sec() - t0) * 1e9);
        
        // GPU benchmark
        ABMatrix mAr = ab_matrix_create(N, K);
        ABMatrix mAi = ab_matrix_create(N, K);
        ABMatrix mCr = ab_matrix_create(N, N);
        ABMatrix mCi = ab_matrix_create(N, N);
        ab_matrix_upload(mAr, Ar, true);
        ab_matrix_upload(mAi, Ai, true);
        ab_zherk(mAr, mAi, mCr, mCi); // warmup
        
        t0 = get_time_sec();
        for (int i = 0; i < iters; i++)
            ab_zherk(mAr, mAi, mCr, mCi);
        double gpu_gflops = flops * iters / ((get_time_sec() - t0) * 1e9);
        
        ab_matrix_download(mCr, Cr_gpu, true);
        ab_matrix_download(mCi, Ci_gpu, true);
        
        // Precision check (upper triangle)
        double max_err = 0;
        for (int i = 0; i < N; i++) {
            for (int j = i; j < N; j++) {
                double err_r = fabs(Cr_gpu[i * N + j] - creal(C_ref[i * N + j]));
                double err_i = fabs(Ci_gpu[i * N + j] - cimag(C_ref[i * N + j]));
                if (err_r > max_err) max_err = err_r;
                if (err_i > max_err) max_err = err_i;
            }
        }
        
        double speedup = gpu_gflops / amx_gflops;
        printf("  %4d    │    %6.0f    │    %6.0f    │  %5.2fx %s │  %s\n",
               N, amx_gflops, gpu_gflops, speedup, speedup >= 1.0 ? "✓" : " ",
               max_err < 1e-10 ? "✓ PASS" : "✗ FAIL");
        
        ab_matrix_destroy(mAr); ab_matrix_destroy(mAi);
        ab_matrix_destroy(mCr); ab_matrix_destroy(mCi);
        free(Ar); free(Ai); free(Cr_gpu); free(Ci_gpu);
        free(A_ref); free(C_ref);
    }
    
    printf("\n");
    printf("Note: K fixed at %d\n", K);
    
    ab_shutdown();
    return 0;
}
